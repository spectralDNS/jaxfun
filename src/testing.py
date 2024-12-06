import jax 
import numpy as np
import jax.numpy as jnp
import timeit 
import Legendre 
import Chebyshev 
jnp.set_printoptions(4)
jax.config.update("jax_enable_x64", True)

N = 10
M = 1000
C = 10
x = jnp.linspace(-1, 1, N)
xn = np.array(x)
c = jnp.ones(C)
cn = np.array(c)
print('Legendre - Vandermonde')
Legendre.legvander(x, N)
print(f'{"legvander":20s} {timeit.timeit(f"Legendre.legvander(x, {N})", number=M, setup="from __main__ import Legendre, x"):.4e}')
print(f'{"np.legvander":20s} {timeit.timeit(f"np.polynomial.legendre.legvander(xn, {N-1})", number=M, setup="from __main__ import np, xn"):.4e}')
assert jnp.linalg.norm(jnp.array(np.polynomial.legendre.legvander(xn, N-1))-Legendre.legvander(x, N)) < 1e-8

print('Legendre - evaluate')
Legendre.legval(x, c)
print(f'{"legval":20s} {timeit.timeit("Legendre.legval(x, c)", number=M, setup="from __main__ import Legendre, x, c"):.4e}')
print(f'{"np.legval":20s} {timeit.timeit("np.polynomial.legendre.legval(xn, cn)", number=M, setup="from __main__ import np, xn, cn"):.4e}')

print('Chebyshev - Vandermonde')
Chebyshev.chebvander(x, N)
print(f'{"chebvander":20s} {timeit.timeit(f"Chebyshev.chebvander(x, {N})", number=M, setup="from __main__ import Chebyshev, x"):.4e}')
Chebyshev.chebvander(x, N)
print(f'{"np.chebvander":20s} {timeit.timeit(f"np.polynomial.chebyshev.chebvander(xn, {N-1})", number=M, setup="from __main__ import np, xn"):.4e}')
assert jnp.linalg.norm(jnp.array(np.polynomial.chebyshev.chebvander(x, N-1))-Chebyshev.chebvander(x, N)) < 1e-8

print('Chebyshev - evaluate')
Chebyshev.chebval(x, c)
print(f'{"chebval":20s} {timeit.timeit("Chebyshev.chebval(x, c)", number=M, setup="from __main__ import Chebyshev, x, c"):.4e}')
print(f'{"np.chebval":20s} {timeit.timeit("np.polynomial.chebyshev.chebval(xn, cn)", number=M, setup="from __main__ import np, xn, cn"):.4e}')
