from typing import Union
from jax import Array
import jax.numpy as jnp
from jaxfun.arguments import TestFunction
from jaxfun.forms import get_basisfunctions, inspect_form
from jaxfun.utils.common import matmat, from_dense
from jaxfun.composite import Composite
import sympy as sp


def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 10,
    return_all_items: bool = False,
) -> Union[Array, list[Array]]:
    r"""Compute coefficient matrix

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

    for ai in a_forms:  # Bilinear form
        uo = u.functionspace.orthogonal

        i, j = 0, 0
        for aii in ai.args:
            if isinstance(aii, sp.Derivative):
                if isinstance(aii.args[0], TestFunction):
                    i = aii.derivative_count
                else:
                    j = aii.derivative_count
        w = wj
        if len(ai.args) == 3:
            scale = ai.args[0]
            if len(scale.free_symbols) > 0:
                s = scale.free_symbols.pop()
                scale = sp.lambdify(s, scale, modules="jax")(xj)
            else:
                scale = float(scale)
            w = w * scale

        Pi = vo.evaluate_basis_derivative(xj, k=i)
        Pj = uo.evaluate_basis_derivative(xj, k=j)
        z = matmat(Pi.T * w[None, :], Pj)
        if u.functionspace == v.functionspace and isinstance(u.functionspace, Composite):
            z = v.functionspace.apply_stencil_galerkin(z)
        elif isinstance(u.functionspace, Composite):
            z = v.functionspace.apply_stencils_petrovgalerkin(z, u.functionspace.S)
        A.append(z)

    if len(A) > 0:
        if not return_all_items:
            A = jnp.sum(jnp.array(A), axis=0)
            if sparse:
                A = from_dense(A, sparse_tol=sparse_tol)
        else:
            if sparse:
                A = [from_dense(a, sparse_tol=sparse_tol) for a in A]
        return A

    # Linear form
    for bi in b_forms:
        i = 0
        for bii in bi.args:
            if isinstance(bii, sp.Derivative):
                i = bii.derivative_count

        Pi = vo.evaluate_basis_derivative(xj, k=i)
        if len(bi.args) > 1:
            uj = bi.args[0]
            for bb in bi.args[1:]:
                if not isinstance(bb, TestFunction):
                    uj = uj * bb
            if len(uj.free_symbols) > 0:
                s = uj.free_symbols.pop()
                uj = sp.lambdify(s, uj, modules="jax")(xj)
            else:
                uj = float(uj)
        b.append(v.functionspace.apply_stencil_left(matmat(uj * wj, Pi)))

    if return_all_items:
        return b
    return jnp.sum(jnp.array(b), axis=0)
