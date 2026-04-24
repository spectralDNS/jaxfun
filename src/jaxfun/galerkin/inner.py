import importlib
from typing import Any, Literal, TypeGuard, cast, overload

import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun.galerkin import JAXFunction
from jaxfun.la import DiaMatrix, Matrix, MatrixProtocol
from jaxfun.typing import TrialSpaceType
from jaxfun.utils.common import lambdify, matmat, tosparse

from .arguments import (
    ArgumentTag,
    TestFunction,
    TrialFunction,
    evaluate_jaxfunction_expr_quad,
    get_arg,
)
from .composite import BCGeneric, Composite, DirectSum
from .forms import (
    _has_functionspace,
    _has_globalindex,
    _has_testspace,
    get_basisfunctions,
    get_jaxfunctions,
    split,
    split_coeff,
)
from .orthogonal import OrthogonalSpace
from .tensorproductspace import (
    BlockTPMatrix,
    DirectSumTPS,
    TensorMatrix,
    TensorProductSpace,
    TPMatrix,
    VectorTensorProductSpace,
)


def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    return_all_items: bool = False,
    num_quad_points: int | tuple[int, ...] | None = None,
):
    r"""Assemble Galerkin inner products (bilinear / linear forms).

    Supports expressions of the forms:
        a(u, v) - L(v)
        a(u, v)
        L(v)

    Finds test / trial functions, splits expression into coefficients and
    separated coordinate factors, constructs (tensor) matrices and load
    vectors. Handles:
      * Composite / BCGeneric spaces (boundary constraints / lifting)
      * Direct sums of tensor product spaces
      * Multivariable non-separable coefficients (symbolic factor kept)
      * JAXFunction (produces linear contributions)

    Args:
        expr: SymPy expression containing TestFunction (mandatory) and
            optionally TrialFunction, JAXFunction, scalar
            coordinate-dependent factors.
        sparse: If True, sparsify (1D) matrix/tensor factors (DiaMatrix).
        sparse_tol: Zero tolerance (integer multiple of ulp) for sparsify.
        return_all_items: If True return raw list(s) of matrices / vectors
            before summation (always for >1D tensor product).
        num_quad_points: Number of quadrature points to use for evaluating
            the inner products. Can be an integer (1D) or a tuple of integers
            for each dimension. If None, the default number of quadrature
            points is used. This is sufficient for bilinear forms with
            constant coefficients, but for nonlinear forms or forms with
            non-constant coefficients, the number of quadrature points may need
            to be increased for exact integration.
            Note that this offers the possibility to use de-aliasing for
            nonlinear terms, but the user is responsible for ensuring that
            the number of quadrature points is sufficient. For the 3/2 rule,
            use tuple(int(1.5 * n) for n in test_space.num_quad_points).

    Returns:
        Depending on content:
          * Matrix (bilinear only, 1D)
          * Vector (linear only)
          * (Matrix, Vector) tuple
          * Lists / TensorMatrix / TPMatrix objects for >1D
    """  # noqa: E501
    V, U = get_basisfunctions(expr)
    assert V is not None, "No TestFunction found in expression"
    assert _has_testspace(V), "TestFunction has no associated function space"
    test_space = V.functionspace
    measure = test_space.system.sg
    allforms = split(expr * measure)
    a_forms = allforms["bilinear"]
    b_forms = allforms["linear"]
    aresults = []
    bresults = []
    all_linear = not (
        len(a_forms) > 0
        and jnp.any(
            jnp.array(["bilinear" in split_coeff(c0["coeff"]) for c0 in a_forms])
        )
    )
    num_quad_points = (
        num_quad_points if num_quad_points is not None else test_space.num_quad_points
    )

    for a0 in a_forms:  # Bilinear form
        # There is one tensor product matrix or just matrix (1D) for each a0
        # If the form contains a JAXFunction, then the matrix assembled will
        # be multiplied with the JAXFunction and a vector will be returned

        mats: list[tuple[tuple[Array, Array] | Matrix, tuple[int, int]]] = []
        # Split coefficients into linear (JAXFunction) and bilinear (constant
        # coefficients) contributions.
        coeffs = split_coeff(a0["coeff"])
        sc = coeffs.get("bilinear", 1)

        trial_space: TrialSpaceType | None = getattr(U, "functionspace", None)
        trial: list[OrthogonalSpace] = []
        has_bcs = False

        for key, ai in a0.items():
            if key in ("coeff", "multivar", "jaxfunction"):
                continue

            assert isinstance(ai, sp.Expr)
            v, u = get_basisfunctions(ai)
            assert v is not None and u is not None, (
                "Both test and trial functions required in bilinear form"
            )
            assert _has_testspace(v), "TestFunction has no associated function space"
            assert _has_functionspace(u), (
                "TrialFunction has no associated function space"
            )
            vf = v.functionspace
            uf = u.functionspace

            assert isinstance(vf, OrthogonalSpace)
            assert isinstance(uf, OrthogonalSpace)
            assert _has_globalindex(v)
            assert _has_globalindex(u)

            # Store the global indices used in each matrix in case of vector-valued
            # space, for assembly of BC contributions and/or multivariable forms
            global_indices = v.global_index, u.global_index

            trial.append(uf)
            if isinstance(uf, BCGeneric):
                has_bcs = True

            is_multivar = "multivar" in a0 or "jaxfunction" in a0
            N = (
                num_quad_points
                if isinstance(num_quad_points, int)
                else num_quad_points[vf.system.base_scalars()[0]._id[0]]
            )

            z = inner_bilinear(ai, vf, uf, sc, is_multivar, N)

            if isinstance(z, tuple):  # multivar
                mats.append((z, global_indices))
                sc = 1
                continue

            if z.size == 0:
                continue
            if isinstance(uf, BCGeneric) and test_space.dims == 1:
                sign = 1 if all_linear else -1
                bresults.append(
                    sign * (z @ jnp.array(uf.bcs.orderedvals(), dtype=float))
                )
                continue
            if "linear" in coeffs and test_space.dims == 1:
                sign = 1 if all_linear else -1
                scale = coeffs["linear"].get("scale", 1) * sign
                bresults.append(scale * (z @ coeffs["linear"]["jaxcoeff"].array))

            sc = 1
            mats.append((z, global_indices))

        if len(mats) == 0:
            pass

        elif test_space.dims == 1 and "bilinear" in coeffs:
            aresults.append(mats[0][0])

        elif isinstance(mats[0][0], tuple):
            # multivariable form, like sqrt(x+y)*u*v, that cannot be separated
            # Multivar only implemented for 2D TensorProductSpace at present
            assert len(mats) == 2
            assert isinstance(test_space, TensorProductSpace)
            mats_ = [
                cast(tuple[Array, Array], mats[0][0]),  # test/trial x-dir
                cast(tuple[Array, Array], mats[1][0]),  # test/trial y-dir
            ]
            gi = [m[1] for m in mats]

            scales = []
            if "multivar" in a0:
                scales.append(a0["multivar"])
            if "jaxfunction" in a0:
                scales.append(
                    evaluate_jaxfunction_expr_quad(a0["jaxfunction"], N=num_quad_points)
                )
            Am = assemble_multivar(mats_, scales, test_space)
            if has_bcs:
                sign = 1 if all_linear else -1
                assert isinstance(
                    trial_space, TensorProductSpace | VectorTensorProductSpace
                )
                res = sign * jnp.einsum(
                    "ikjl,kl->ij", Am, trial_space.bndvals[tuple(trial)]
                )
                global_indices = gi[0]
                bresults.append(vectorize_bresult(res, test_space, gi[0][0]))

            else:
                if "linear" in coeffs:
                    sign = 1 if all_linear else -1
                    res = sign * jnp.einsum(
                        "ikjl,kl->ij", Am, coeffs["linear"]["jaxcoeff"].array
                    )
                    bresults.append(vectorize_bresult(res, test_space, gi[0][0]))

                if "bilinear" in coeffs:
                    assert isinstance(trial_space, TensorProductSpace)
                    aresults.append(TensorMatrix(Am))

        elif len(mats) > 1:  # regular separable multivariable form
            mats_: list[Array] = [cast(Matrix, m[0]).data for m in mats]
            gi = [m[1] for m in mats]

            assert isinstance(test_space, TensorProductSpace | VectorTensorProductSpace)

            if has_bcs:
                assert len(mats_) == 2  # BCs only implemented for 2D at present
                if trial_space is not None:
                    if isinstance(trial_space, TensorProductSpace):
                        fun = trial_space.bndvals[tuple(trial)]
                    elif isinstance(trial_space, VectorTensorProductSpace):
                        fun = trial_space[gi[1][1]].bndvals[tuple(trial)]
                    else:
                        raise NotImplementedError(
                            "BCs only implemented for TensorProductSpace and VectorTensorProductSpace"  # noqa: E501
                        )
                else:
                    jfs = coeffs["linear"]["jaxcoeff"].functionspace
                    assert isinstance(jfs, DirectSumTPS | VectorTensorProductSpace)
                    if isinstance(jfs, DirectSumTPS):
                        fun = jfs.bndvals[tuple(trial)]
                    else:
                        dsspace = jfs.tensorspaces[gi[1][1]]
                        assert isinstance(dsspace, DirectSumTPS)
                        fun = dsspace.bndvals[tuple(trial)]

                sign = 1 if all_linear else -1
                res = sign * (mats_[0] @ fun @ mats_[1].T)
                global_indices = gi[0]
                bresults.append(vectorize_bresult(res, test_space, global_indices[0]))

            else:
                if "linear" in coeffs:
                    sign = 1 if all_linear else -1
                    if test_space.dims == 2:
                        res = (sign * coeffs["linear"]["scale"]) * (
                            mats_[0] @ coeffs["linear"]["jaxcoeff"].array @ mats_[1].T
                        )
                    else:
                        res = sign * jnp.einsum(
                            "il,jm,kn,lmn->ijk",
                            mats_[0],
                            mats_[1],
                            mats_[2],
                            coeffs["linear"]["jaxcoeff"].array,
                        )
                    bresults.append(vectorize_bresult(res, test_space, gi[0][0]))

                if "bilinear" in coeffs:
                    assert isinstance(
                        test_space, TensorProductSpace | VectorTensorProductSpace
                    )
                    assert isinstance(
                        trial_space, TensorProductSpace | VectorTensorProductSpace
                    )
                    aresults.append(
                        TPMatrix(
                            [cast(MatrixProtocol, m[0]) for m in mats],
                            1,
                            global_indices=gi[0],
                        )
                    )

    # Pure linear forms
    for b0 in b_forms:
        sc = sp.sympify(b0["coeff"])
        sc = float(sc) if sc.is_real else complex(sc)
        if len(a_forms) > 0:
            sc = sc * (-1)

        bs = []

        for key, bi in b0.items():
            if key in ("coeff", "multivar", "jaxfunction"):
                continue

            assert isinstance(bi, sp.Expr)
            v, _ = get_basisfunctions(bi)
            assert v is not None, "Test function required in linear form"
            assert _has_functionspace(v)
            vf = v.functionspace
            assert isinstance(vf, OrthogonalSpace)
            assert _has_globalindex(v)
            global_index = v.global_index
            is_multivar = "multivar" in b0 or "jaxfunction" in b0
            z = inner_linear(
                bi,
                vf,
                sc,
                is_multivar,
                num_quad_points
                if isinstance(num_quad_points, int)
                else num_quad_points[vf.system.base_scalars()[0]._id[0]],
            )
            sc = 1
            bs.append(z)

        if isinstance(test_space, OrthogonalSpace):
            bresults.append(bs[0])
        elif (
            isinstance(test_space, TensorProductSpace | VectorTensorProductSpace)
            and len(test_space) == 2
        ):
            if isinstance(bs[0], tuple):
                assert isinstance(num_quad_points, tuple)
                # multivar or JAXFunction
                uj = jnp.array(1.0)
                if "multivar" in b0:
                    s = test_space.system.base_scalars()
                    uj *= lambdify(s, b0["multivar"], modules="jax")(
                        *test_space.mesh(N=num_quad_points)
                    )
                if "jaxfunction" in b0:
                    uj *= evaluate_jaxfunction_expr_quad(
                        b0["jaxfunction"], N=num_quad_points
                    )
                if "jaxfunction" not in b0 and "multivar" not in b0:
                    raise ValueError("Expected multivar or jaxfunction key in b0")
                res = bs[0][0].T @ uj @ bs[1][0]
                bresults.append(vectorize_bresult(res, test_space, global_index))

            else:
                res = jnp.multiply.outer(bs[0], bs[1])
                bresults.append(vectorize_bresult(res, test_space, global_index))

        elif (
            isinstance(test_space, TensorProductSpace | VectorTensorProductSpace)
            and len(test_space) == 3
        ):
            if isinstance(bs[0], tuple):
                assert isinstance(num_quad_points, tuple)
                # multivar or JAXFunction
                if "multivar" in b0:
                    s = test_space.system.base_scalars()
                    uj = lambdify(s, b0["multivar"], modules="jax")(
                        *test_space.mesh(N=num_quad_points)
                    )
                elif "jaxfunction" in b0:
                    uj = evaluate_jaxfunction_expr_quad(
                        b0["jaxfunction"], N=num_quad_points
                    )
                else:
                    raise ValueError("Expected multivar or jaxfunction key in b0")
                res = jnp.einsum("il,jm,kn,ijk->lmn", bs[0][0], bs[1][0], bs[2][0], uj)
                bresults.append(vectorize_bresult(res, test_space, global_index))
            else:
                res = jnp.multiply.outer(jnp.multiply.outer(bs[0], bs[1]), bs[2])
                bresults.append(vectorize_bresult(res, test_space, global_index))

    return process_results(
        aresults, bresults, return_all_items, test_space.dims, sparse, sparse_tol
    )


def vectorize_bresult(
    res: Array, space: TensorProductSpace | VectorTensorProductSpace, global_index: int
) -> Array:
    if not isinstance(space, VectorTensorProductSpace):
        return res
    out = jnp.zeros((space.dims,) + res.shape, dtype=res.dtype)
    return out.at[global_index].set(res)


@overload
def process_results(
    aresults: list[Matrix | TPMatrix | TensorMatrix],
    bresults: list[Array],
    return_all_items: Literal[True],
    dims: int,
    sparse: bool,
    sparse_tol: int,
) -> tuple[list[Matrix | TPMatrix | TensorMatrix], list[Array]]: ...
@overload
def process_results(
    aresults: list[Matrix | TPMatrix | TensorMatrix],
    bresults: list[Array],
    return_all_items: Literal[False],
    dims: int,
    sparse: Literal[False],
    sparse_tol: int,
) -> (
    Matrix
    | TPMatrix
    | TensorMatrix
    | Array
    | tuple[Matrix | TPMatrix | TensorMatrix, Array]
): ...
@overload
def process_results(
    aresults: list[Matrix | TPMatrix | TensorMatrix],
    bresults: list[Array],
    return_all_items: Literal[False],
    dims: int,
    sparse: Literal[True],
    sparse_tol: int,
) -> (
    DiaMatrix
    | TPMatrix
    | TensorMatrix
    | Array
    | tuple[DiaMatrix | TPMatrix | TensorMatrix, Array]
): ...
def process_results(
    aresults: list[Matrix | TPMatrix | TensorMatrix],
    bresults: list[Array],
    return_all_items: bool,
    dims: int,
    sparse: bool,
    sparse_tol: int,
) -> Any:
    """Finalize assembly results (sum terms, optional sparsify).

    Args:
        aresults: List of bilinear matrices (dense or structured holders).
        bresults: List of load vectors / tensors.
        return_all_items: If True skip summation (raw lists returned).
        dims: Spatial dimension (1 => sum; >1 keep tensor structure).
        sparse: If True, sparsify (only when applicable).
        sparse_tol: Zero tolerance for sparsification.

    Returns:
        Matrix, vector, (matrix, vector) or lists depending on inputs.
    """
    if return_all_items:
        return aresults, bresults
    if len(aresults) > 0 and dims == 1:
        aresults: Matrix = Matrix(
            jnp.sum(jnp.array([cast(Matrix, a).data for a in aresults]), axis=0)
        )
        if sparse:
            aresults: DiaMatrix = tosparse(aresults.data, tol=sparse_tol)

    if len(aresults) > 0 and dims > 1 and sparse:
        aresults: list[TPMatrix] = cast(list[TPMatrix], aresults)
        for a0 in aresults:
            if isinstance(a0, TPMatrix):
                a0.mats: list[DiaMatrix] = [
                    tosparse(cast(Matrix, a0.mats[i]).data, sparse_tol)
                    for i in range(a0.dims)
                ]

    if len(bresults) > 0:
        bresults: Array = jnp.sum(jnp.array(bresults), axis=0)

    # Return just the one matrix/vector if 1D and only bilinear or linear forms
    if len(aresults) > 0 and len(bresults) == 0:
        return aresults

    if len(aresults) == 0 and len(bresults) > 0:
        return bresults

    return aresults, bresults


@overload
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: Literal[False],
    num_quad_points: int,
) -> Matrix: ...
@overload
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: Literal[True],
    num_quad_points: int,
) -> tuple[Array, Array]: ...
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: bool,
    num_quad_points: int,
) -> Matrix | tuple[Array, Array]:
    """Assemble single bilinear form contribution term.

    Detects derivative orders on test/trial factors, applies optional
    symbolic coefficient (possibly sampled at quadrature points) and
    returns dense matrix or tuple (multivar separated factors).

    Args:
        ai: SymPy sub-expression containing basis factors.
        v: Test function space (Orthogonal or Composite).
        u: Trial function space.
        sc: Scalar bilinear coefficient (after linear split).
        multivar: True if coefficient not separable (handled upstream).
        num_quad_points: Number of quadrature points.
    Returns:
        Matrix, or (Pi, Pj) tuple for multivar separation.
    """
    vo = v.orthogonal
    uo = u.orthogonal
    N = num_quad_points
    xj, wj = vo.quad_points_and_weights(N=N)
    df = float(vo.domain_factor)
    i, j = 0, 0
    scale = jnp.array([sc])
    for aii in ai.args:
        found_basis = False
        for p in sp.core.traversal.preorder_traversal(aii):
            if get_arg(p) in (ArgumentTag.TEST, ArgumentTag.TRIAL):
                found_basis = True
                break
        if found_basis:
            if isinstance(aii, sp.Derivative):
                arg = get_arg(aii.args[0])
                if arg is ArgumentTag.TEST:
                    assert i == 0
                    i = int(aii.derivative_count)
                elif arg is ArgumentTag.TRIAL:
                    assert j == 0
                    j = int(aii.derivative_count)
            continue

        jaxfunction = get_jaxfunctions(aii)
        if len(jaxfunction) == 1:
            scale *= evaluate_jaxfunction_expr_quad(aii, jaxfunction.pop(), N=N)
            continue
        elif len(jaxfunction) > 1:
            raise ValueError("Multiple JAXFunctions found in single bilinear form")

        if len(aii.free_symbols) > 0:
            s = aii.free_symbols.pop()
            scale *= lambdify(s, uo.map_expr_true_domain(aii), modules="jax")(xj)
        else:
            scale *= float(aii)  # ty:ignore[invalid-argument-type]

    z = None
    if len(scale) == 1 and not multivar:
        # Look up matrix
        mod = importlib.import_module(vo.__class__.__module__)
        z = mod.matrices((vo, i), (uo, j))
        if z is None:
            pass
        else:
            s = scale * df ** (i + j - 1)
            if s.item() != 1:
                z.data = z.data * s
            z = z.todense()

    if z is None:
        w = wj * df ** (i + j - 1) * scale
        Pi = vo.evaluate_basis_derivative(xj, k=i)
        Pj = uo.evaluate_basis_derivative(xj, k=j)

        if multivar:
            multi: Array = v.apply_stencil_right(w[:, None] * Pi)
            multj: Array = u.apply_stencil_right(jnp.conj(Pj))
            return multi, multj

        z = matmat(Pi.T * w[None, :], jnp.conj(Pj))

    if u == v and isinstance(u, Composite):
        z = v.apply_stencil_galerkin(z)
    elif isinstance(v, Composite) and isinstance(u, Composite):
        z = v.apply_stencils_petrovgalerkin(z, u.S)
    elif isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    elif isinstance(u, Composite):
        z = u.apply_stencil_right(z)

    return Matrix(z)


def inner_linear(
    bi: sp.Expr,
    v: OrthogonalSpace,
    sc: float | complex,
    multivar: bool,
    num_quad_points: int,
) -> Array | tuple[Array]:
    """Assemble single linear form contribution.

    Args:
        bi: SymPy term with one test function (possibly derivative).
        v: Test function space.
        sc: Scalar coefficient (sign-adjusted if from a(u,v)-L(v)).
        global_index: Global index for vector-valued spaces.
        multivar: True if non-separable scaling (return tuple form).
        num_quad_points: Number of quadrature points to use
            (de-aliasing for nonlinear / non-constant coeffs).

    Returns:
        Vector (1D), tuple (Pi,) for multivar, or projected result.
    """
    vo = v.orthogonal
    N = num_quad_points
    xj, wj = vo.quad_points_and_weights(N=N)
    df = float(vo.domain_factor)
    i = 0
    uj = jnp.array([sc])  # incorporate scalar coefficient into first vector

    if get_arg(bi) is ArgumentTag.TEST:
        pass
    elif isinstance(bi, sp.Derivative):
        i = int(bi.derivative_count)
    else:
        for bii in bi.args:
            found_basis = False
            for p in sp.core.traversal.preorder_traversal(bii):
                if get_arg(p) is ArgumentTag.TEST:
                    found_basis = True
                    break
            if found_basis:
                if isinstance(bii, sp.Derivative):
                    assert i == 0
                    i = int(bii.derivative_count)
                continue

            jaxfunction = get_jaxfunctions(bii)
            if len(jaxfunction) == 1:
                jaxf = jaxfunction.pop()
                assert jaxf.functionspace.orthogonal.__class__ == vo.__class__, (
                    "JAXFunction space must match test function space"
                )
                uj *= evaluate_jaxfunction_expr_quad(bii, jaxf, N=N)
                continue
            # bii contains coordinates in the domain of v, e.g., (r, theta) for polar
            # Need to compute bii as bii(x(X)), since we use quadrature points
            if len(bii.free_symbols) > 0:
                s = bii.free_symbols.pop()
                uj *= lambdify(s, vo.map_expr_true_domain(bii), modules="jax")(xj)
            else:
                uj *= float(bii)  # ty:ignore[invalid-argument-type]

    Pi = vo.evaluate_basis_derivative(xj, k=i)
    w = wj * df ** (i - 1)  # Account for domain different from reference
    if multivar:
        return (v.apply_stencil_right((uj * w)[:, None] * jnp.conj(Pi)),)

    z = matmat(uj * w, jnp.conj(Pi))
    if isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    return z


def contains_sympy_symbols(obj: object) -> TypeGuard[sp.Expr]:
    return len(sp.sympify(obj).free_symbols) > 0


def assemble_multivar(
    mats: list[tuple[Array, Array]],
    scale: sp.Expr | Array | list[sp.Expr | Array],
    test_space: TensorProductSpace,
) -> Array:
    """Contract separated multivariable factors into global matrix.

    Handles expressions like sqrt(x+y)*u*v where coefficient cannot be
    separated into pure x / y factors. Performs batched contraction.

    Args:
        mats: List [(P0,P1),(P2,P3)] of separated directional factors.
        scale: SymPy expression or numeric scalar/array.
        test_space: Tensor product space (for mesh / variable order).

    Returns:
        Dense matrix of shape (i, k, j, l) assembled from factors.
    """
    P0, P1 = mats[0]
    P2, P3 = mats[1]
    sci = jnp.array([1.0])
    for sc in scale if isinstance(scale, list) else [scale]:
        if not isinstance(sc, Array) and contains_sympy_symbols(sc):
            s = test_space.system.base_scalars()
            xj = test_space.mesh(N=(P0.shape[0], P1.shape[0]))
            sc = lambdify(s, sc, modules="jax")(*xj)
        elif isinstance(sc, float | complex):
            sc = jnp.ones((P0.shape[0], P1.shape[0])) * sc
        else:
            sc = cast(Array, sc)
        sci = sci * sc

    a = jnp.einsum("pi,pk,qj,ql,pq->ikjl", P0, P1, P2, P3, sci)
    return a


def project1D(ue: sp.Expr, V: OrthogonalSpace | Composite | DirectSum) -> Array:
    """Project scalar expression ue onto 1D space V.

    Args:
        ue: SymPy expression in physical coordinate.
        V: Orthogonal / Composite / DirectSum space.

    Returns:
        Coefficient vector uh.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    M, b = inner(v * (u - ue))
    uh = M.solve(b)
    return uh


def project(ue: sp.Expr, V: TrialSpaceType) -> Array:
    """Project expression onto (possibly tensor) space V.

    Args:
        ue: SymPy expression.
        V: Function space (may be tensor product/direct sum).

    Returns:
        Coefficient array shaped to V.num_dofs.
    """
    from scipy import sparse as scipy_sparse

    from jaxfun.operators import Dot

    if V.dims == 1:
        assert isinstance(V, OrthogonalSpace | Composite | DirectSum)
        return project1D(ue, V)

    if len(get_jaxfunctions(ue)) == 0:
        assert not isinstance(V, OrthogonalSpace | Composite | DirectSum)
        if V.rank == 0:
            uj = lambdify(V.system.base_scalars(), ue, modules="jax")(*V.mesh())
            uj = jnp.broadcast_to(uj, V.num_quad_points)
        elif V.rank == 1:
            assert isinstance(V, VectorTensorProductSpace)
            s = V.system.base_scalars()
            bv = V.system.base_vectors()
            uj = (lambdify(s, Dot(ue, n).doit())(*V.mesh()) for n in bv)
            uj = jnp.stack(
                [jnp.broadcast_to(ui, V.tensorspaces[0].num_quad_points) for ui in uj],
                axis=0,
            )
        return V.forward(uj)

    u = TrialFunction(V)
    v = TestFunction(V)
    if V.rank == 0:
        M, b = inner(v * (u - ue))
        uh = jnp.linalg.solve(M[0].mat, b.flatten()).reshape(V.num_dofs)

    elif V.rank == 1:
        assert isinstance(ue, sp.Mul | sp.Add | JAXFunction), (
            "Projection requires unevaluated expressions"
        )  # noqa: E501
        assert isinstance(V, VectorTensorProductSpace)
        M, b = inner(Dot(v, (u - ue)))
        A = BlockTPMatrix(M, V, V)
        C = A.block_array()
        uh = jnp.array(scipy_sparse.linalg.spsolve(C, b.ravel()).reshape(b.shape))

    return uh
