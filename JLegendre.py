import jax 
import jax.numpy as jnp 
from jax import Array 

jax.config.update("jax_enable_x64", True)

@jax.jit
def legval(x : float, c : Array) -> float:
    """
    Evaluate a Legendre series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    Parameters
    ----------
    x : array_like, compatible object
    c : array_like
    
    Returns
    -------
    values : Array

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    
    if len(c) == 1:
        c0 : float = c[0]
        c1 : float = 0
    elif len(c) == 2:
        c0 : float = c[0]
        c1 : float = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x
