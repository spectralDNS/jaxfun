import timeit

import jax
import jax.numpy as jnp
import numpy as np

import Chebyshev
import Legendre

jnp.set_printoptions(4)
jax.config.update("jax_enable_x64", True)

N = 1000
M = 1000
C = 1000
x = jnp.linspace(-1, 1, N)
xn = np.array(x)
c = jnp.ones(C)
cn = np.array(c)


def run_legvander() -> None:
    print("Legendre - Vandermonde")
    Legendre.legvander(x, N)

    time_jax = timeit.timeit(
        f"Legendre.legvander(x, {N})",
        number=M,
        setup="from __main__ import Legendre, x",
    )
    print(f'{"legvander":20s} {time_jax:.4e}')
    time_np = timeit.timeit(
        f"np.polynomial.legendre.legvander(xn, {N - 1})",
        number=M,
        setup="from __main__ import np, xn",
    )
    print(f'{"np.legvander":20s} {time_np:.4e}')

    assert (
        jnp.linalg.norm(
            jnp.array(np.polynomial.legendre.legvander(xn, N - 1))
            - Legendre.legvander(x, N)
        )
        < 1e-8
    )


def run_legval() -> None:
    print("Legendre - evaluate")

    Legendre.legval(x, c)
    time_jax = timeit.timeit(
        "Legendre.legval(x, c)",
        number=M,
        setup="from __main__ import Legendre, x, c",
    )
    print(f'{"legval":20s} {time_jax:.4e}')

    time_np = timeit.timeit(
        "np.polynomial.legendre.legval(xn, cn)",
        number=M,
        setup="from __main__ import np, xn, cn",
    )
    print(f'{"np.legval":20s} {time_np:.4e}')


def run_chebvander() -> None:
    print("Chebyshev - Vandermonde")
    Chebyshev.chebvander(x, N)
    time_jax = timeit.timeit(
        f"Chebyshev.chebvander(x, {N})",
        number=M,
        setup="from __main__ import Chebyshev, x",
    )
    print(f'{"chebvander":20s} {time_jax:.4e}')

    Chebyshev.chebvander(x, N)
    time_np = timeit.timeit(
        f"np.polynomial.chebyshev.chebvander(xn, {N - 1})",
        number=M,
        setup="from __main__ import np, xn",
    )
    print(f'{"np.chebvander":20s} {time_np:.4e}')

    assert (
        jnp.linalg.norm(
            jnp.array(np.polynomial.chebyshev.chebvander(x, N - 1))
            - Chebyshev.chebvander(x, N)
        )
        < 1e-8
    )


def run_chebval() -> None:
    print("Chebyshev - evaluate")
    Chebyshev.chebval(x, c)
    time_jax = timeit.timeit(
        "Chebyshev.chebval(x, c)",
        number=M,
        setup="from __main__ import Chebyshev, x, c",
    )
    print(f'{"chebval":20s} {time_jax:.4e}')

    time_np = timeit.timeit(
        "np.polynomial.chebyshev.chebval(xn, cn)",
        number=M,
        setup="from __main__ import np, xn, cn",
    )
    print(f'{"np.chebval":20s} {time_np:.4e}')


if __name__ == "__main__":
    run_legvander()
    run_legval()
    run_chebvander()
    run_chebval()
