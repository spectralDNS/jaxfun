from typing import Union
from jax import Array
import numpy as np
import jax.numpy as jnp
from jaxfun.arguments import test, trial, BasisFunction
from jaxfun.forms import get_basisfunctions, inspect_form
from jaxfun.utils.common import matmat, tosparse
from jaxfun.composite import Composite
from jaxfun.utils.common import lambdify
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
        If it contains a :class:`.TrialFunction`, then this function returns matrices, whereas
        if it only contains a :class:`.TestFunction` a vector is returned.
    sparse : bool
        if True, then sparsify the matrices before returning
    sparse_tol : int
        An integer multiple of one ulp. The tolereance for something being zero is determined based
        on the absolute value of the number being less than sparse_tol*ulp.
    return_all_items : bool
        Whether to return just one matrix/vector, or whether to return all computed matrices/vectors.
        None that one expr may maintain any number of terms leading to many matrices/vectors

    """
    v, u = get_basisfunctions(expr)
    a_forms, b_forms = inspect_form(expr)
    A, b = [], []
    vo = v.functionspace.orthogonal
    xj, wj = vo.quad_points_and_weights()
    df = vo.domain_factor

    for ai in a_forms:  # Bilinear form
        uo = u.functionspace.orthogonal

        i, j = 0, 0
        scale = jnp.ones(1)
        for aii in ai.args:
            if isinstance(aii, sp.Derivative):
                if isinstance(aii.args[0], test):
                    i = int(aii.derivative_count)
                elif isinstance(aii.args[0], trial):
                    j = int(aii.derivative_count)
            
            found_basis = False
            for p in sp.core.traversal.preorder_traversal(aii):
                if isinstance(p, BasisFunction):
                    found_basis = True
                    break
            if found_basis:
                continue
            
            if len(aii.free_symbols) > 0:
                s = aii.free_symbols.pop()
                scale *= lambdify(s, aii, modules="jax")(xj)
            else:
                scale *= float(aii)
        w = wj * df**(i+j-1) * scale # Account for domain different from reference
        
        Pi = vo.evaluate_basis_derivative(xj, k=i)
        Pj = uo.evaluate_basis_derivative(xj, k=j)
        z = matmat(Pi.T * w[None, :], Pj)
        if u.functionspace == v.functionspace and isinstance(u.functionspace, Composite):
            z = v.functionspace.apply_stencil_galerkin(z)
        elif isinstance(v.functionspace, Composite) and isinstance(u.functionspace, Composite):
            z = v.functionspace.apply_stencils_petrovgalerkin(z, u.functionspace.S)
        elif isinstance(v.functionspace, Composite):
            z = v.functionspace.apply_stencil_left(z)
        elif isinstance(u.functionspace, Composite):
            z = u.functionspace.apply_stencil_right(z)
        A.append(z)

    if len(A) > 0:
        if not return_all_items:
            A = jnp.sum(jnp.array(A), axis=0)
            if sparse:
                A = tosparse(A, tol=sparse_tol)
        else:
            if sparse:
                A = [tosparse(a, tol=sparse_tol) for a in A]
        return A

    # Linear form
    for bi in b_forms:
        i = 0    
        uj = jnp.ones(1)
        for bii in bi.args:
            if isinstance(bii, sp.Derivative):
                i = int(bii.derivative_count)
        
            found_basis = False
            for p in sp.core.traversal.preorder_traversal(bii):
                if isinstance(p, BasisFunction):
                    found_basis = True
                    break
            if found_basis:
                continue
                
            if len(bii.free_symbols) > 0:
                s = bii.free_symbols.pop()
                uj *= lambdify(s, bii, modules="jax")(xj)
            else:
                uj *= float(bii)

        Pi = vo.evaluate_basis_derivative(xj, k=i)        
        w = wj * df**(i-1) # Account for domain different from reference
        z = matmat(uj * w, Pi)
        if isinstance(v.functionspace, Composite):
            z = v.functionspace.apply_stencil_left(z)
        b.append(z)

    if return_all_items:
        return b
    return jnp.sum(jnp.array(b), axis=0)
