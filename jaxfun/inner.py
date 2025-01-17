from typing import Union, Any
from jax import Array
import jax.numpy as jnp
from jaxfun.arguments import test, trial, BasisFunction
from jaxfun.coordinates import CoordSys
from jaxfun.forms import get_basisfunctions, split
from jaxfun.utils.common import matmat, tosparse
from jaxfun.composite import Composite
from jaxfun.Basespace import BaseSpace
from jaxfun.utils.common import lambdify
from jaxfun.tensorproductspace import TensorProductSpace, TPMatrix
import sympy as sp


def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    return_all_items: bool = False,
) -> Union[Array, list[Array]]:
    r"""Compute 1D inner products

    Parameters
    ----------
    expr : Sympy Expr
        An expression containing :class:`.TestFunction` and optionally :class:`.TrialFunction`.
    sparse : bool
        if True, then sparsify the matrices before returning
    sparse_tol : int
        An integer multiple of one ulp. The tolereance for something being zero is determined based
        on the absolute value of the number being less than sparse_tol*ulp.
    return_all_items : bool
        Whether to return just one matrix/vector, or whether to return all computed matrices/vectors.
        Note that one expr may maintain any number of terms leading to many matrices/vectors.
        This parameter is only relevant for bilinear 1D problems. Multidimensional problems always 
        returns all matrices.

    """
    V, U = get_basisfunctions(expr)
    test_space = V.functionspace
    trial_space = U.functionspace if U is not None else None
    measure = test_space.system.sg
    forms = split(expr * measure)
    a_forms = forms["bilinear"]
    b_forms = forms["linear"]
    aresults = []
    bresults = []
    bilinear_return_all_items = True if isinstance(test_space, TensorProductSpace) else return_all_items

    for a0 in a_forms:  # Bilinear form
        # There is one tensor product matrix or just matrix (1D) for each a0
        mats = []
        sc: float = float(a0["coeff"])  # one scalar coefficient to all the matrices
        for key, ai in a0.items():
            if key == "coeff":
                continue

            if isinstance(test_space, TensorProductSpace):
                v, u = test_space.spacemap[key], trial_space.spacemap[key]
            else:
                v, u = test_space, trial_space

            z = inner_bilinear(ai, v, u, float(sc))
            sc = 1
            if sparse and bilinear_return_all_items:
                z = tosparse(z, tol=sparse_tol)
            mats.append(z)

        if isinstance(test_space, TensorProductSpace):
            aresults.append(TPMatrix(mats, 1.0, test_space, trial_space))
        else:
            aresults.append(mats[0])

    if len(aresults) > 0:
        if (not bilinear_return_all_items) and isinstance(test_space, BaseSpace):
            aresults = jnp.sum(jnp.array(aresults), axis=0)
            if sparse:
                aresults = tosparse(aresults, tol=sparse_tol)
        
    # Linear form
    for b0 in b_forms:
        bs = []
        sc: float = float(b0["coeff"])  # one scalar coefficient to all the vectors
        for key, bi in b0.items():
            if key == "coeff":
                continue

            v = (
                test_space.spacemap[key]
                if isinstance(test_space, TensorProductSpace)
                else test_space
            )
            z = inner_linear(bi, v, sc)
            sc = 1
            bs.append(z)
        if isinstance(test_space, BaseSpace):
            bresults.append(bs[0])
        elif len(test_space) == 2:
            bresults.append(jnp.multiply.outer(bs[0], bs[1]))
        elif len(test_space) == 3:
            bresults.append(jnp.multiply.outer(jnp.multiply.outer(bs[0], bs[1]), bs[2])) 
            
    if (not return_all_items) and len(bresults) > 0:
        bresults = jnp.sum(jnp.array(bresults), axis=0)

    # Return just the one matrix/vector if 1D and only bilinear or linear forms
    if len(aresults) > 0 and len(bresults) == 0:
        return aresults
    if len(aresults) == 0 and len(bresults) > 0:
        return bresults
    if len(aresults) > 0 and len(bresults) > 0:
        return aresults, bresults
    return aresults, bresults


def inner_bilinear(ai: sp.Expr, v: BaseSpace, u: BaseSpace, sc: float) -> Array:
    vo = v.orthogonal
    uo = u.orthogonal
    xj, wj = vo.quad_points_and_weights()
    df = float(vo.domain_factor)
    i, j = 0, 0
    scale = jnp.array([sc])
    sc = 1
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
    w = wj * df ** (i + j - 1) * scale  # Account for domain different from reference
    Pi = vo.evaluate_basis_derivative(xj, k=i)
    Pj = uo.evaluate_basis_derivative(xj, k=j)
    z = matmat(Pi.T * w[None, :], Pj)
    if u == v and isinstance(u, Composite):
        z = v.apply_stencil_galerkin(z)
    elif isinstance(v, Composite) and isinstance(u, Composite):
        z = v.apply_stencils_petrovgalerkin(z, u.S)
    elif isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    elif isinstance(u, Composite):
        z = u.apply_stencil_right(z)
    return z


def inner_linear(bi: sp.Expr, v: BaseSpace, sc: float) -> Array:
    vo = v.orthogonal
    xj, wj = vo.quad_points_and_weights()
    df = float(vo.domain_factor)
    i = 0
    uj = jnp.array([sc])  # incorporate scalar coefficient into first matrix
    sc = 1
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
            # Need to map bii to reference domains since we use quadrature points
            if len(bii.free_symbols) > 0:
                s = bii.free_symbols.pop()
                uj *= lambdify(s, vo.map_expr_true_domain(bii), modules="jax")(xj)
            else:
                uj *= float(bii)
    Pi = vo.evaluate_basis_derivative(xj, k=i)
    w = wj * df ** (i - 1) * sc  # Account for domain different from reference
    z = matmat(uj * w, Pi)
    if isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    return z


class Measure:
    sparse = True
    return_all_items = False
    sparse_tol = 100

    def __init__(self, system: CoordSys, **kwargs: dict[Any]) -> None:
        self.system = system
        self.__dict__.update(kwargs)

    def __rmul__(self, expr: sp.Expr):
        return inner(
            expr * self.system.sg,
            sparse=self.sparse,
            return_all_items=self.return_all_items,
            sparse_tol=self.sparse_tol,
        )
