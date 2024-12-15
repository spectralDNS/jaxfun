from typing import Union
from jax import Array
from jaxfun.composite import (
    Composite,
    apply_stencil_galerkin,
    apply_stencil_petrovgalerkin,
    apply_stencil_linear,
)
from jaxfun.utils.common import matmat, from_dense
import sympy as sp


def inner(
    test: tuple[Composite, int], trial: Union[tuple[Composite, int], sp.Expr], sparse: bool = False
) -> Array:
    r"""Compute coefficient matrix

    .. math::
        (\frac{\partial^{i}\psi_m}{\partialx^{i}}, \frac{\partial^{j}\phi_n}{\partialx^{j}})

    where :math:`\psi_j` is the j'th basis function of the trial space and
    :math:`\phi_i` is the j'th basis function of the test space
    """
    v, i = test
    if isinstance(trial, tuple):  # Bilinear form
        u, j = trial
        x, w = v.orthogonal.quad_points_and_weights()
        Pi = v.orthogonal.evaluate_basis_derivative(x, k=i)
        Pj = u.orthogonal.evaluate_basis_derivative(x, k=j)
        z = matmat(Pi.T * w[None, :], Pj)
        if u == v:
            z = apply_stencil_galerkin(v.S, z)
        else:
            z = apply_stencil_petrovgalerkin(v.S, z, u.S)
        return from_dense(z) if sparse else z

    # Linear form
    x, w = v.orthogonal.quad_points_and_weights()
    Pi = v.orthogonal.evaluate_basis_derivative(x, k=i)
    try:
        s = trial.free_symbols.pop()
        uj = sp.lambdify(s, trial, modules=["jax"])(x)
    except AttributeError:
        # constant function
        uj = trial
    
    return apply_stencil_linear(v.S, matmat(uj * w, Pi))
