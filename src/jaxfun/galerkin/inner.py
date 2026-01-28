import importlib
from typing import Any, TypeGuard

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental.sparse import BCOO

from jaxfun.utils.common import lambdify, matmat, tosparse

from .arguments import TestFunction, TrialFunction
from .composite import BCGeneric, Composite, DirectSum
from .forms import (
    _has_functionspace,
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
)


def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    return_all_items: bool = False,
) -> Any:
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
    assert _has_functionspace(V), "TestFunction has no associated function space"
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
        mats = []
        # Split coefficients into linear (JAXFunction) and bilinear (constant
        # coefficients) contributions.
        coeffs = split_coeff(a0["coeff"])
        sc = coeffs.get("bilinear", 1)

        trial_space = getattr(U, "functionspace", None)
        trial = []
        has_bcs = False

        for key, ai in a0.items():
            if key in ("coeff", "multivar"):
                continue

            v, u = get_basisfunctions(ai)
            assert v is not None and u is not None, (
                "Both test and trial functions required in bilinear form"
            )
            assert _has_functionspace(v), (
                "TestFunction has no associated function space"
            )
            assert _has_functionspace(u), (
                "TrialFunction has no associated function space"
            )
            vf = v.functionspace
            uf = u.functionspace
            assert isinstance(vf, OrthogonalSpace)
            assert isinstance(uf, OrthogonalSpace)

            trial.append(uf)
            if isinstance(uf, BCGeneric):
                has_bcs = True

            z = inner_bilinear(ai, vf, uf, sc, "multivar" in a0)

            if isinstance(z, tuple):  # multivar
                mats.append(z)
                continue
            if z.size == 0:
                continue
            if isinstance(uf, BCGeneric) and test_space.dims == 1:
                bresults.append(-(z @ jnp.array(uf.bcs.orderedvals())))
                continue
            if "linear" in coeffs and test_space.dims == 1:
                sign = 1 if all_linear else -1
                scale = coeffs["linear"].get("scale", 1) * sign
                bresults.append(scale * (z @ coeffs["linear"]["jaxfunction"].array))

            sc = 1
            mats.append(z)

        if len(mats) == 0:
            pass
        elif test_space.dims == 1 and "bilinear" in coeffs:
            aresults.append(mats[0])

        elif isinstance(mats[0], tuple):
            # multivariable form, like sqrt(x+y)*u*v, that cannot be separated
            assert isinstance(test_space, TensorProductSpace)
            Am = assemble_multivar(mats, a0["multivar"], test_space)

            if has_bcs:
                assert trial_space is not None
                bresults.append(
                    -(Am @ trial_space.bndvals[(tuple(trial))].flatten()).reshape(
                        test_space.num_dofs
                    )
                )
            else:
                if "linear" in coeffs:
                    sign = 1 if all_linear else -1
                    bresults.append(
                        sign
                        * (
                            Am @ coeffs["linear"]["jaxfunction"].array.flatten()
                        ).reshape(test_space.num_dofs)
                    )
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

        else:  # regular separable multivariable form
            if has_bcs:
                fun = (
                    trial_space.bndvals[tuple(trial)]
                    if trial_space
                    else coeffs["linear"]["jaxfunction"].space.bndvals[tuple(trial)]
                )
                bresults.append(-(mats[0] @ fun @ mats[1].T))
            else:
                if "linear" in coeffs:
                    sign = 1 if all_linear else -1
                    bresults.append(
                        (sign * coeffs["linear"]["scale"])
                        * (mats[0] @ coeffs["linear"]["jaxfunction"].array @ mats[1].T)
                    )

                if "bilinear" in coeffs:
                    assert isinstance(test_space, TensorProductSpace)
                    assert isinstance(trial_space, TensorProductSpace)
                    aresults.append(
                        TPMatrix(
                            mats,
                            coeffs["bilinear"],
                            test_space,
                            trial_space.tpspaces[tuple(trial)]
                            if isinstance(trial_space, DirectSumTPS)
                            else trial_space,
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
            if key in ("coeff", "multivar"):
                continue

            v, _ = get_basisfunctions(bi)
            assert v is not None, "Test function required in linear form"
            assert _has_functionspace(v)
            assert isinstance(v.functionspace, OrthogonalSpace)

            z = inner_linear(
                bi,
                v.functionspace,
                sc,
                "multivar" in b0 or isinstance(jaxarrays, Array),
            )

            sc = 1
            bs.append(z)
        if isinstance(test_space, OrthogonalSpace):
            bresults.append(bs[0])
        elif len(test_space) == 2:
            if isinstance(bs[0], tuple):
                # multivar
                if "multivar" in b0:
                    assert isinstance(test_space, TensorProductSpace)
                    s = test_space.system.base_scalars()
                    xj = test_space.mesh()
                    uj = lambdify(s, b0["multivar"], modules="jax")(*xj)
                else:
                    uj = jaxarrays
                bresults.append(bs[0][0].T @ uj @ bs[1][0])
            else:
                bresults.append(jnp.multiply.outer(bs[0], bs[1]))
        elif len(test_space) == 3:
            bresults.append(jnp.multiply.outer(jnp.multiply.outer(bs[0], bs[1]), bs[2]))

    return process_results(
        aresults, bresults, return_all_items, test_space.dims, sparse, sparse_tol
    )


def tosparse_and_attach(z: Array, sparse_tol: int) -> BCOO:
    """Convert dense operator to BCOO and preserve metadata.

    Args:
        z: Dense matrix with attached attributes (test/trial data).
        sparse_tol: Zero tolerance passed to tosparse.

    Returns:
        BCOO sparse matrix with copied metadata attributes.
    """
    a0 = tosparse(z, tol=sparse_tol)
    a0.test_derivatives = z.test_derivatives  # ty:ignore[unresolved-attribute]
    a0.trial_derivatives = z.trial_derivatives  # ty:ignore[unresolved-attribute]
    a0.test_name = z.test_name  # ty:ignore[unresolved-attribute]
    a0.trial_name = z.trial_name  # ty:ignore[unresolved-attribute]
    return a0


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
            a0.mats: list[BCOO] = [
                tosparse_and_attach(
                    a0.mats[i],  # ty:ignore[possibly-missing-attribute]
                    sparse_tol,
                )
                for i in range(a0.dims)  # ty:ignore[possibly-missing-attribute]
            ]

    if len(bresults) > 0:
        bresults: Array = jnp.sum(jnp.array(bresults), axis=0)

    # Return just the one matrix/vector if 1D and only bilinear or linear forms
    if len(aresults) > 0 and len(bresults) == 0:
        return aresults

    if len(aresults) == 0 and len(bresults) > 0:
        return bresults

    return aresults, bresults


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
            if hasattr(p, "argument"):
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
            return (
                v.apply_stencil_right(w[:, None] * Pi),
                u.apply_stencil_right(jnp.conj(Pj)),
            )

        z = matmat(Pi.T * w[None, :], jnp.conj(Pj))

    if u == v and isinstance(u, Composite):
        z = v.apply_stencil_galerkin(z)
    elif isinstance(v, Composite) and isinstance(u, Composite):
        z = v.apply_stencils_petrovgalerkin(z, u.S)
    elif isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    elif isinstance(u, Composite):
        z = u.apply_stencil_right(z)

    # Attach some attributes to the matrix such that it can be easily recognized
    z.test_derivatives = i
    z.trial_derivatives = j
    z.test_name = v.name
    z.trial_name = u.name
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
    scale: sp.Expr | Array,
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
    i, k = P0.shape[1], P1.shape[1]
    j, l = P2.shape[1], P3.shape[1]  # noqa: E741
    if contains_sympy_symbols(scale):
        s = test_space.system.base_scalars()
        xj = test_space.mesh()
        scale = lambdify(s, scale, modules="jax")(*xj)
    else:
        scale = jnp.ones((P0.shape[0], P1.shape[0])) * scale

    def fun(p0, p1, p2, p3):
        ph0 = p0[:, None] * p1
        ph1 = p2[:, None] * p3
        return ph0.T @ scale @ ph1

    a = jax.vmap(
        jax.vmap(fun, in_axes=(None, None, 1, None)), in_axes=(1, None, None, None)
    )(P0, P1, P2, P3)

    return a.reshape((i * j, k * l))


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
