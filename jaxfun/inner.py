import importlib
from typing import Any

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental.sparse import BCOO

from jaxfun.arguments import (
    BasisFunction,
    TestFunction,
    TrialFunction,
    test,
    trial,
)
from jaxfun.Basespace import BaseSpace
from jaxfun.composite import BCGeneric, Composite
from jaxfun.coordinates import CoordSys
from jaxfun.forms import get_basisfunctions, split, split_coeff
from jaxfun.tensorproductspace import (
    DirectSumTPS,
    TensorMatrix,
    TensorProductSpace,
    TPMatrix,
)
from jaxfun.utils.common import lambdify, matmat, tosparse


def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    return_all_items: bool = False,
) -> Array | list[Array]:
    r"""Compute inner products

    Assemble bilinear and linear forms. The expr input needs to represent one of the following
    three combinations

        a(u, v) - L(v)
        a(u, v)
        L(v)

    where a(u, v) and L(v) are bilinear and linear forms, respectively. In addition to test and
    trial functions, the forms may also contain regular spectral functions (:class:`.JAXFunction`)
    and Sympy functions of spatial coordinates (e.g., :class:`.ScalarFunction`).

    For a(u, v) - L(v) we return both matrices and a vector, where the vector represents the right
    hand side of the linear system Ax = b. If only a(u, v), then we return only matrices unless there
    are non-zero boundary conditions, in which case a right-hand side vector is returned as well. If
    only L(v), then only a vector is returned.

    If `return_all_items=True`, then we return all computed matrices and vectors, without adding them
    together first.

    Parameters
    ----------
    expr : Sympy Expr
        An expression containing :class:`.TestFunction` and optionally :class:`.TrialFunction`
        or :class:`.JAXFunction`. May also contain any Sympy function of spatial coordinates.
    sparse : bool
        if True, then sparsify the matrices before returning
    sparse_tol : int
        An integer multiple of one ulp. The tolereance for something being zero is
        determined based on the absolute value of the number being less than
        sparse_tol*ulp.
    return_all_items : bool
        Whether to return just one matrix/vector, or whether to return all computed matrices/vectors.
        Note that one expr may maintain any number of terms leading to many matrices/vectors.
        This parameter is only relevant for 1D problems. Multidimensional problems always
        returns all tensor product matrices.

    """  # noqa: E501
    V, U = get_basisfunctions(expr)
    test_space = V.functionspace
    trial_space = U.functionspace if U is not None else None
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

        trial = []
        has_bcs = False

        for key, ai in a0.items():
            if key in ("coeff", "multivar"):
                continue

            v, u = get_basisfunctions(ai)
            vf = v.functionspace
            uf = u.functionspace
            trial.append(uf)
            if isinstance(uf, BCGeneric):
                has_bcs = True

            z = inner_bilinear(ai, vf, uf, sc, "multivar" in a0)

            if isinstance(z, tuple):  # multivar
                mats.append(z)
                continue
            if z.size == 0:
                continue
            if has_bcs and test_space.dims == 1:
                bresults.append(-(z @ jnp.array(uf.bcs.orderedvals())))
                continue
            if "linear" in coeffs and test_space.dims == 1:
                sign = 1 if all_linear else -1
                bresults.append(sign * (z @ coeffs["linear"]["jaxfunction"].array))

            sc = 1
            mats.append(z)

        if len(mats) == 0:
            pass
        elif test_space.dims == 1 and "bilinear" in coeffs:
            aresults.append(mats[0])

        elif isinstance(mats[0], tuple):
            # multivariable form, like sqrt(x+y)*u*v, that cannot be separated
            Am = assemble_multivar(mats, a0["multivar"], test_space)

            if has_bcs:
                bresults.append(
                    -(Am @ trial_space.bndvals[(tuple(trial))].flatten()).reshape(
                        test_space.dim()
                    )
                )
            else:
                if "linear" in coeffs:
                    sign = 1 if all_linear else -1
                    bresults.append(
                        sign
                        * (
                            Am @ coeffs["linear"]["jaxfunction"].array.flatten()
                        ).reshape(test_space.dim())
                    )
                if "bilinear" in coeffs:
                    assert coeffs["bilinear"] == 1
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

    # Pure linear forms: (no JAXFunctions, but could be unseparable in coefficients)
    for b0 in b_forms:
        sc = sp.sympify(b0["coeff"])
        sc = float(sc) if sc.is_real else complex(sc)
        if len(a_forms) > 0:
            sc = sc * (-1)

        bs = []
        for key, bi in b0.items():
            if key in ("coeff", "multivar"):
                continue

            v, _ = get_basisfunctions(bi)

            z = inner_linear(bi, v.functionspace, sc, "multivar" in b0)

            sc = 1
            bs.append(z)
        if isinstance(test_space, BaseSpace):
            bresults.append(bs[0])
        elif len(test_space) == 2:
            if isinstance(bs[0], tuple):
                # multivar
                s = test_space.system.base_scalars()
                xj = test_space.mesh()
                uj = lambdify(s, b0["multivar"], modules="jax")(*xj)
                bresults.append(bs[0][0].T @ uj @ bs[1][0])
            else:
                bresults.append(jnp.multiply.outer(bs[0], bs[1]))
        elif len(test_space) == 3:
            bresults.append(jnp.multiply.outer(jnp.multiply.outer(bs[0], bs[1]), bs[2]))

    return process_results(
        aresults, bresults, return_all_items, test_space.dims, sparse, sparse_tol
    )


def tosparse_and_attach(z: Array, sparse_tol: int) -> BCOO:
    a0 = tosparse(z, tol=sparse_tol)
    a0.test_derivatives = z.test_derivatives
    a0.trial_derivatives = z.trial_derivatives
    a0.test_name = z.test_name
    a0.trial_name = z.trial_name
    return a0


def process_results(
    aresults: list[Array],
    bresults: list[Array],
    return_all_items: bool,
    dims: int,
    sparse: bool,
    sparse_tol: int,
) -> Array | list[Array] | BCOO | list[BCOO] | tuple[BCOO | Array, Array]:
    if return_all_items:
        return aresults, bresults

    if len(aresults) > 0 and dims == 1:
        aresults = jnp.sum(jnp.array(aresults), axis=0)
        if sparse:
            aresults = tosparse(aresults, tol=sparse_tol)

    if len(aresults) > 0 and dims > 1 and sparse:
        for a0 in aresults:
            a0.mats = [
                tosparse_and_attach(
                    a0.mats[i],
                    sparse_tol,
                )
                for i in range(a0.dims)
            ]

    if len(bresults) > 0:
        bresults = jnp.sum(jnp.array(bresults), axis=0)

    # Return just the one matrix/vector if 1D and only bilinear or linear forms
    if len(aresults) > 0 and len(bresults) == 0:
        return aresults

    if len(aresults) == 0 and len(bresults) > 0:
        return bresults

    return aresults, bresults


def inner_bilinear(
    ai: sp.Expr, v: BaseSpace, u: BaseSpace, sc: float | complex, multivar: bool
) -> Array:
    vo = v.orthogonal
    uo = u.orthogonal
    xj, wj = vo.quad_points_and_weights()
    df = float(vo.domain_factor)
    i, j = 0, 0
    scale = jnp.array([sc])
    for aii in ai.args:
        found_basis = False
        for p in sp.core.traversal.preorder_traversal(aii):
            if isinstance(p, BasisFunction):
                found_basis = True
                break
        if found_basis:
            if isinstance(aii, sp.Derivative):
                if isinstance(aii.args[0], test):
                    assert i == 0
                    i = int(aii.derivative_count)
                elif isinstance(aii.args[0], trial):
                    assert j == 0
                    j = int(aii.derivative_count)
            continue
        if len(aii.free_symbols) > 0:
            s = aii.free_symbols.pop()
            scale *= lambdify(s, uo.map_expr_true_domain(aii), modules="jax")(xj)
        else:
            scale *= float(aii)

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
    v: BaseSpace,
    sc: float | complex,
    multivar: bool,
) -> Array:
    vo = v.orthogonal
    xj, wj = vo.quad_points_and_weights()
    df = float(vo.domain_factor)
    i = 0
    uj = jnp.array([sc])  # incorporate scalar coefficient into first vector
    if isinstance(bi, test):
        pass
    elif isinstance(bi, sp.Derivative):
        i = int(bi.derivative_count)
    else:
        for bii in bi.args:
            found_basis = False
            for p in sp.core.traversal.preorder_traversal(bii):
                if isinstance(p, BasisFunction):
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
                uj *= float(bii)

    Pi = vo.evaluate_basis_derivative(xj, k=i)
    w = wj * df ** (i - 1)  # Account for domain different from reference
    if multivar:
        return (v.apply_stencil_right((uj * w)[:, None] * jnp.conj(Pi)),)

    z = matmat(uj * w, jnp.conj(Pi))
    if isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    return z


def assemble_multivar(
    mats: list[tuple[Array, Array]], scale: Array, test_space: TensorProductSpace
) -> Array:
    P0, P1 = mats[0]
    P2, P3 = mats[1]
    i, k = P0.shape[1], P1.shape[1]
    j, l = P2.shape[1], P3.shape[1]
    if len(sp.sympify(scale).free_symbols) > 0:
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


def project1D(ue: sp.Expr, V: BaseSpace) -> Array:
    u = TrialFunction(V)
    v = TestFunction(V)
    M, b = inner(v * (u - ue))
    uh = jnp.linalg.solve(M, b)
    return uh


def project(ue: sp.Expr, V: BaseSpace) -> Array:
    u = TrialFunction(V)
    v = TestFunction(V)
    M, b = inner(v * (u - ue))
    uh = jnp.linalg.solve(M[0].mat, b.flatten()).reshape(V.dim())
    return uh


# Experimental measure
class Measure:
    sparse = True
    return_all_items = False
    sparse_tol = 100

    def __init__(self, system: CoordSys, **kwargs: dict[Any]) -> None:
        self.system = system
        self.__dict__.update(kwargs)

    def __rmul__(self, expr: sp.Expr) -> Array | list[Array]:
        return inner(
            expr * self.system.sg,
            sparse=self.sparse,
            return_all_items=self.return_all_items,
            sparse_tol=self.sparse_tol,
        )
