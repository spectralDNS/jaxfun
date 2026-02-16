import importlib
from typing import Literal, TypeGuard, cast, overload

import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental.sparse import BCOO
from sympy.core.function import AppliedUndef

from jaxfun.typing import TrialSpaceType
from jaxfun.utils.common import lambdify, matmat, tosparse

from .arguments import TestFunction, TrialFunction, evaluate_jaxfunction_expr
from .composite import BCGeneric, Composite, DirectSum
from .forms import (
    _has_functionspace,
    _has_globalindex,
    _has_testspace,
    get_basisfunctions,
    get_jaxarrays,
    split,
    split_coeff,
    split_linear_coeff,
)
from .orthogonal import OrthogonalSpace
from .tensorproductspace import (
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
      * JAXArray factors in linear forms

    Args:
        expr: SymPy expression containing TestFunction (mandatory) and
            optionally TrialFunction, JAXFunction, JAXArray, scalar
            coordinate-dependent factors.
        sparse: If True, sparsify (1D) matrix/tensor factors (BCOO).
        sparse_tol: Zero tolerance (integer multiple of ulp) for sparsify.
        return_all_items: If True return raw list(s) of matrices / vectors
            before summation (always for >1D tensor product).

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

    for a0 in a_forms:  # Bilinear form
        # There is one tensor product matrix or just matrix (1D) for each a0
        # If the form contains a JAXFunction, then the matrix assembled will
        # be multiplied with the JAXFunction and a vector will be returned

        mats: list[tuple[tuple[Array, Array] | Array, tuple[int, int]]] = []
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

            z = inner_bilinear(ai, vf, uf, sc, "multivar" in a0 or "jaxfunction" in a0)

            if isinstance(z, tuple):  # multivar
                mats.append((z, global_indices))
                continue

            if z.size == 0:
                continue
            if isinstance(uf, BCGeneric) and test_space.dims == 1:
                sign = 1 if all_linear else -1
                bresults.append(sign * (z @ jnp.array(uf.bcs.orderedvals())))
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
                    evaluate_jaxfunction_expr(a0["jaxfunction"], test_space.mesh())
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
                    assert coeffs["bilinear"] == 1
                    assert isinstance(trial_space, TensorProductSpace)
                    aresults.append(
                        TensorMatrix(
                            Am,
                            test_space,
                            trial_space.tpspaces[tuple(trial)]
                            if isinstance(trial_space, DirectSumTPS)
                            else trial_space,
                        )
                    )

        elif len(mats) > 1:  # regular separable multivariable form
            mats_ = [cast(Array, m[0]) for m in mats]
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
                    res = (sign * coeffs["linear"]["scale"]) * (
                        mats_[0] @ coeffs["linear"]["jaxcoeff"].array @ mats_[1].T
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
                            mats_,
                            coeffs["bilinear"],
                            test_space,
                            trial_space.tpspaces[tuple(trial)]
                            if isinstance(trial_space, DirectSumTPS)
                            else trial_space,
                            global_indices=gi[0],
                        )
                    )

    # Pure linear forms: (no JAXFunctions, but could be unseparable in coefficients and
    # contain JAXArrays)
    for b0 in b_forms:
        jaxarrays = get_jaxarrays(b0["coeff"])
        if len(jaxarrays) == 1:
            jaxarrays = split_linear_coeff(b0["coeff"])
            b0["coeff"] = 1
        elif len(jaxarrays) > 1:
            raise NotImplementedError(
                "Only one JAXArray allowed in linear form at present"
            )

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

            z = inner_linear(
                bi,
                vf,
                sc,
                "multivar" in b0
                or isinstance(jaxarrays, Array)
                or ("jaxfunction" in b0),
            )
            sc = 1
            bs.append(z)
        if isinstance(test_space, OrthogonalSpace):
            if isinstance(bs[0], tuple):
                # JAXArray (no multivar)
                Pi = bs[0][0]
                bresults.append(Pi.T @ jaxarrays)
            else:
                bresults.append(bs[0])
        elif (
            isinstance(test_space, TensorProductSpace | VectorTensorProductSpace)
            and len(test_space) == 2
        ):
            if isinstance(bs[0], tuple):
                # multivar or JAXArray
                if "multivar" in b0:
                    s = test_space.system.base_scalars()
                    xj = test_space.mesh()
                    uj = lambdify(s, b0["multivar"], modules="jax")(*xj)
                elif "jaxfunction" in b0:
                    uj = evaluate_jaxfunction_expr(b0["jaxfunction"], test_space.mesh())
                else:
                    uj = jaxarrays
                res = bs[0][0].T @ uj @ bs[1][0]
                bresults.append(vectorize_bresult(res, test_space, global_index))

            else:
                res = jnp.multiply.outer(bs[0], bs[1])
                bresults.append(vectorize_bresult(res, test_space, global_index))

        elif (
            isinstance(test_space, TensorProductSpace | VectorTensorProductSpace)
            and len(test_space) == 3
        ):
            res = jnp.multiply.outer(jnp.multiply.outer(bs[0], bs[1]), bs[2])
            bresults.append(vectorize_bresult(res, test_space, global_index))

    return process_results(
        aresults, bresults, return_all_items, test_space.dims, sparse, sparse_tol
    )


def toBCOO(z: Array, sparse_tol: int) -> BCOO:
    """Convert dense operator to BCOO.

    Args:
        z: Dense matrix with attached attributes (test/trial data).
        sparse_tol: Zero tolerance passed to tosparse.

    Returns:
        BCOO sparse matrix.
    """
    return tosparse(z, tol=sparse_tol)


def vectorize_bresult(
    res: Array, space: TensorProductSpace | VectorTensorProductSpace, global_index: int
) -> Array:
    if not isinstance(space, VectorTensorProductSpace):
        return res
    out = jnp.zeros((space.dims,) + res.shape, dtype=res.dtype)
    return out.at[global_index].set(res)


def process_results(
    aresults: list[Array],
    bresults: list[Array],
    return_all_items: bool,
    dims: int,
    sparse: bool,
    sparse_tol: int,
) -> (
    Array | list[Array] | BCOO | tuple[list[Array] | Array | BCOO, list[Array] | Array]
):
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
        aresults: Array = jnp.sum(jnp.array(aresults), axis=0)
        if sparse:
            aresults: BCOO = tosparse(aresults, tol=sparse_tol)

    if len(aresults) > 0 and dims > 1 and sparse:
        for a0 in aresults:
            assert isinstance(a0, TPMatrix)
            a0.mats: list[BCOO] = [
                toBCOO(
                    a0.mats[i],
                    sparse_tol,
                )
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
) -> Array: ...
@overload
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: Literal[True],
) -> tuple[Array, Array]: ...
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: bool,
) -> Array | tuple[Array, Array]:
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

    Returns:
        Dense matrix, or (Pi, Pj) tuple for multivar separation.
    """
    vo = v.orthogonal
    uo = u.orthogonal
    xj, wj = vo.quad_points_and_weights()
    df = float(vo.domain_factor)
    i, j = 0, 0
    scale = jnp.array([sc])
    for aii in ai.args:
        found_basis = False
        for p in sp.core.traversal.preorder_traversal(aii):
            if getattr(p, "argument", -1) in (0, 1):
                found_basis = True
                break
        if found_basis:
            if isinstance(aii, sp.Derivative):
                if getattr(aii.args[0], "argument", -1) == 0:
                    assert i == 0
                    i = int(aii.derivative_count)
                elif getattr(aii.args[0], "argument", -1) == 1:
                    assert j == 0
                    j = int(aii.derivative_count)
            continue
        jaxfunction = None
        for p in sp.core.traversal.preorder_traversal(aii):
            if getattr(p, "argument", -1) == 2:  # JAXFunction->AppliedUndef
                jaxfunction = p
                break
        if jaxfunction:
            scale *= evaluate_jaxfunction_expr(aii, xj, cast(AppliedUndef, jaxfunction))
            continue
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

    return z


def inner_linear(
    bi: sp.Expr,
    v: OrthogonalSpace,
    sc: float | complex,
    multivar: bool,
) -> Array | tuple[Array]:
    """Assemble single linear form contribution.

    Args:
        bi: SymPy term with one test function (possibly derivative).
        v: Test function space.
        sc: Scalar coefficient (sign-adjusted if from a(u,v)-L(v)).
        global_index: Global index for vector-valued spaces.
        multivar: True if non-separable scaling (return tuple form).

    Returns:
        Vector (1D), tuple (Pi,) for multivar, or projected result.
    """
    vo = v.orthogonal
    xj, wj = vo.quad_points_and_weights()
    df = float(vo.domain_factor)
    i = 0
    uj = jnp.array([sc])  # incorporate scalar coefficient into first vector
    if getattr(bi, "argument", -1) == 0:
        pass
    elif isinstance(bi, sp.Derivative):
        i = int(bi.derivative_count)
    else:
        for bii in bi.args:
            found_basis = False
            for p in sp.core.traversal.preorder_traversal(bii):
                if getattr(p, "argument", -1) == 0:
                    found_basis = True
                    break
            if found_basis:
                if isinstance(bii, sp.Derivative):
                    assert i == 0
                    i = int(bii.derivative_count)
                continue

            jaxfunction = None
            for p in sp.core.traversal.preorder_traversal(bii):
                if getattr(p, "argument", -1) == 2:  # JAXFunction->AppliedUndef
                    jaxfunction = cast(AppliedUndef, p)
                    break
            if jaxfunction:
                uj *= evaluate_jaxfunction_expr(bii, xj, jaxfunction)
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
        Dense matrix of shape (i*j, k*l) assembled from factors.
    """
    P0, P1 = mats[0]
    P2, P3 = mats[1]
    sci = jnp.array([1.0])
    for sc in scale if isinstance(scale, list) else [scale]:
        if not isinstance(sc, Array) and contains_sympy_symbols(sc):
            s = test_space.system.base_scalars()
            xj = test_space.mesh()
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
    uh = jnp.linalg.solve(M, b)
    return uh


def project(ue: sp.Basic, V: OrthogonalSpace | TensorProductSpace) -> Array:
    """Project expression onto (possibly tensor) space V.

    Args:
        ue: SymPy expression.
        V: Function space (may be tensor product/direct sum).

    Returns:
        Coefficient array shaped to V.num_dofs.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    M, b = inner(v * (u - ue))
    uh = jnp.linalg.solve(M[0].mat, b.flatten()).reshape(V.num_dofs)
    return uh
