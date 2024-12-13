import timeit

import jax
import jax.numpy as jnp
import numpy as np

from jaxfun import Chebyshev
from jaxfun import Legendre

jnp.set_printoptions(4)
jax.config.update("jax_enable_x64", True)
print("JAX running on", jax.devices()[0].platform.upper())

N = 100
M = 1000
C = 1000
k = 2
x = jnp.linspace(-1, 1, N)
xn = np.array(x)
c = jnp.ones(C)
cn = np.array(c)


def run_vandermonde(space) -> None:
    family = space.__name__.split('.')[-1]
    print(f"{family} - Vandermonde")
    space.vandermonde(x, N)

    time_jax = timeit.timeit(
        f"{family}.vandermonde(x, {N})",
        number=M,
        setup=f"from __main__ import {family}, x",
    )
    print(f'{"Jax":20s} {time_jax:.4e}')
    npfun = {
        'Legendre': np.polynomial.legendre.legvander,
        'Chebyshev': np.polynomial.chebyshev.chebvander
    }[family]
    time_np = timeit.timeit(
        f"npfun(xn, {N - 1})",
        number=M,
        globals={**globals(), **locals()}
    )
    print(f'{"Numpy":20s} {time_np:.4e}')

    assert (
        jnp.linalg.norm(
            jnp.array(npfun(xn, N - 1))
            - space.vandermonde(x, N)
        )
        < 1e-8
    )


def run_evaluate(space) -> None:
    family = space.__name__.split('.')[-1]
    print(f"{family} - evaluate")

    space.evaluate(x, c)
    time_jax = timeit.timeit(
        "space.evaluate(x, c)",
        number=M,
        globals={**globals(), **locals()}
    )
    print(f'{"Jax":20s} {time_jax:.4e}')

    npfun = {
        'Legendre': np.polynomial.legendre.legval,
        'Chebyshev': np.polynomial.chebyshev.chebval
    }[family]
    time_np = timeit.timeit(
        "npfun(xn, cn)",
        number=M,
        globals={**globals(), **locals()}
    )
    print(f'{"Numpy":20s} {time_np:.4e}')


def run_evaluate_basis_derivative(space) -> None:
    family = space.__name__.split('.')[-1]
    print(f"{family} - evaluate_basis_derivative")
    space.evaluate_basis_derivative(x, N, k)
    time_jax = timeit.timeit(
        f"{family}.evaluate_basis_derivative(x, {N}, {k})",
        number=M,
        setup=f"from __main__ import {family}, x",
    )
    print(f'{"Jax":20s} {time_jax:.4e}')

    npfuns = {
        'Legendre': (np.polynomial.legendre.legvander, np.polynomial.legendre.legder),
        'Chebyshev': (np.polynomial.chebyshev.chebvander, np.polynomial.chebyshev.chebder)
    }[family]
    time_np = timeit.timeit(
        f"""
np_res = npfuns[0](xn, {N} - 1)
P = np_res.shape[-1]
D = np.zeros((P, P))
D[:-{k}] = npfuns[1](np.eye(P, P), {k})
np_res = np.dot(np_res, D)    
        """,
        number=M,
        globals={**globals(), **locals()}
    )
    print(f'{"Numpy":20s} {time_np:.4e}')


if __name__ == "__main__":
    for space in (Chebyshev, Legendre):
        run_vandermonde(space)
        run_evaluate(space)
        run_evaluate_basis_derivative(space)
    