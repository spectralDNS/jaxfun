import importlib
from typing import Any

import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun.arguments import BasisFunction, TestFunction, TrialFunction, test, trial
from jaxfun.Basespace import BaseSpace
from jaxfun.composite import BCGeneric, Composite
from jaxfun.coordinates import CoordSys
from jaxfun.forms import get_basisfunctions, split
from jaxfun.tensorproductspace import DirectSumTPS, TPMatrix
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

    where a(u, v) and L(v) are bilinear and linear forms, respectively.

    For a(u, v) - L(v) we return both matrices and a vector, where the vector represents the right
    hand side of the linear system Ax = b. If only a(u, v), then we return only matrices unless there
    are non-zero boundary conditions, in which case a right-hand side vector is returned as well. If
    only L(v), then only a vector is returned.

    If `return_all_items=True`, then we return all computed matrices and vectors, without adding them
    together first.

    Parameters
    ----------
    expr : Sympy Expr
        An expression containing :class:`.TestFunction` and optionally:class:`.TrialFunction`.
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
    forms = split(expr * measure)
    a_forms = forms["bilinear"]
    b_forms = forms["linear"]
    aresults = []
    bresults = []

    for a0 in a_forms:  # Bilinear form
        # There is one tensor product matrix or just matrix (1D) for each a0
        mats = []
        # one scalar coefficient to all the matrices
        sc = sp.sympify(a0["coeff"])
        sc = float(sc) if sc.is_real else complex(sc)
        trial = []
        has_bcs = False
        for key, ai in a0.items():
            if key == "coeff":
                continue

            v, u = get_basisfunctions(ai)
            vf = v.functionspace
            uf = u.functionspace
            trial.append(uf)

            z = inner_bilinear(ai, vf, uf, sc)

            if z.size == 0:
                continue
            if isinstance(uf, BCGeneric) and test_space.dims == 1:
                bresults.append(-(z @ jnp.array(uf.bcs.orderedvals())))
                continue
            if isinstance(uf, BCGeneric):
                has_bcs = True
            sc = 1
            mats.append(z)

        if len(mats) == 0:
            pass
        elif test_space.dims == 1:
            aresults.append(mats[0])
        else:
            if has_bcs:
                bresults.append(
                    -(mats[0] @ trial_space.bndvals[tuple(trial)] @ mats[1].T)
                )
            else:
                aresults.append(
                    TPMatrix(
                        mats,
                        1.0,
                        test_space,
                        trial_space.tpspaces[tuple(trial)]
                        if isinstance(trial_space, DirectSumTPS)
                        else trial_space,
                    )
                )

    # Linear form
    
    for b0 in b_forms:
        bs = []
        sc = sp.sympify(b0["coeff"])
        sc = float(sc) if sc.is_real else complex(sc) 
        if len(a_forms) > 0:
            sc = sc*(-1)
        for key, bi in b0.items():
            if key == "coeff":
                continue

            v, _ = get_basisfunctions(bi)
            z = inner_linear(bi, v.functionspace, sc)
            sc = 1
            bs.append(z)
        if isinstance(test_space, BaseSpace):
            bresults.append(bs[0])
        elif len(test_space) == 2:
            bresults.append(jnp.multiply.outer(bs[0], bs[1]))
        elif len(test_space) == 3:
            bresults.append(jnp.multiply.outer(jnp.multiply.outer(bs[0], bs[1]), bs[2]))

    return process_results(
        aresults, bresults, return_all_items, test_space.dims, sparse, sparse_tol
    )


def process_results(
    aresults, bresults, return_all_items, dims, sparse, sparse_tol
) -> Array | list[Array]:
    if return_all_items:
        return aresults, bresults

    if len(aresults) > 0 and dims == 1:
        aresults = jnp.sum(jnp.array(aresults), axis=0)
        if sparse:
            aresults = tosparse(aresults, tol=sparse_tol)

    if len(bresults) > 0:
        bresults = jnp.sum(jnp.array(bresults), axis=0)

    # Return just the one matrix/vector if 1D and only bilinear or linear forms
    if len(aresults) > 0 and len(bresults) == 0:
        return aresults

    if len(aresults) == 0 and len(bresults) > 0:
        return bresults

    return aresults, bresults


def inner_bilinear(
    ai: sp.Expr, v: BaseSpace, u: BaseSpace, sc: float | complex
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
    if len(scale) == 1:
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


def inner_linear(bi: sp.Expr, v: BaseSpace, sc: float | complex) -> Array:
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
    w = wj * df ** (i - 1)   # Account for domain different from reference
    z = matmat(uj * w, jnp.conj(Pi))
    if isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    return z


def project1D(ue: sp.Expr, V: BaseSpace) -> Array:
    u = TrialFunction(V)
    v = TestFunction(V)
    M, b = inner(v * u - v * ue)
    uh = jnp.linalg.solve(M, b)
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
