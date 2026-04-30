import jax
import jax.numpy as jnp
from jax import Array, lax
from jax._src.lax import control_flow


@jax.jit
def TDMA_Fwd2(a: Array, b: Array, c: Array):
    # for i in range(1, N):
    #    ld[i-1] = ld[i-1]/d[i-1]
    #    d[i] = d[i] - ld[i-1]*ud[i-1]

    def fwd1(dm1, x):
        ldm1, d, udm1 = x
        ld = ldm1 / dm1
        d = d - ld * udm1
        return d, (ld, d)

    ld, d = lax.scan(fwd1, b[0], (a, b[1:], c))[1]
    return ld, jnp.hstack((b[0], d))


@jax.jit
def TDMA_BSolve2(u0: Array, a: Array, b: Array, c: Array) -> Array:
    # for i in range(1, n):
    #    u[i] -= ld[i-1]*u[i-1]
    # u[n-1] = u[n-1]/d[n-1]
    # for i in range(n-2, -1, -1):
    #    u[i] = (u[i] - ud[i]*u[i+1])/d[i]

    def fwd(um1, x):
        u, ldm1 = x
        u -= ldm1 * um1
        return u, u
    u_ = lax.scan(fwd, 0.0, (u0, jnp.hstack((0.0, a))))[1]

    def bwd(up1, x):
        u, d, ud = x
        u = (u - ud * up1) / d
        return u, u

    return lax.scan(bwd, 0.0, (u_, b, jnp.hstack((c, 0.0))), reverse=True)[1]

@jax.jit
def TDMA_Sol2(u0: Array, a: Array, b: Array, c: Array):
    _a, _b = TDMA_Fwd2(a, b, c)
    return TDMA_BSolve2(u0, _a, _b, c)


def TDMA_LU(ld: Array, d: Array, ud: Array, N: int):
    def body_fun(i, carry):
        ld, d = carry
        ld = ld.at[i - 2].set(ld[i - 2] / d[i - 2])  # Update ld[i-2]
        d = d.at[i].set(d[i] - ld[i - 2] * ud[i - 2])  # Update d[i]
        return ld, d

    # Loop from 2 to N as specified in the original function
    ld, d = jax.lax.fori_loop(2, N, body_fun, (ld, d))
    return ld, d


@jax.jit
def TDMA_solve(u: Array, ld: Array, d: Array, ud: Array, n: int) -> Array:
    def forward_body(i, u):
        return u.at[i].set(u[i] - ld[i - 2] * u[i - 2])

    u = jax.lax.fori_loop(2, n, forward_body, u)

    u = u.at[n - 1].set(u[n - 1] / d[n - 1])
    u = u.at[n - 2].set(u[n - 2] / d[n - 2])

    def backward_body(i, u):
        return u.at[i].set((u[i] - ud[i] * u[i + 2]) / d[i])

    u = jax.lax.fori_loop(0, n - 2, lambda i, u: backward_body(n - 3 - i, u), u)

    return u


@jax.jit(static_argnums=3)
def TDMA_O_LU(ld: Array, d: Array, ud: Array, N: int):
    # for i in range(1, N):
    #    ld[i-1] = ld[i-1]/d[i-1]
    #    d[i] = d[i] - ld[i-1]*ud[i-1]

    def body_fun(i, carry):
        ld, d = carry
        ld = ld.at[i - 1].set(ld[i - 1] / d[i - 1])  # Update ld[i-1]
        d = d.at[i].set(d[i] - ld[i - 1] * ud[i - 1])  # Update d[i]
        return ld, d

    # Loop from 2 to N as specified in the original function
    ld, d = jax.lax.fori_loop(1, N, body_fun, (ld, d))
    return ld, d


@jax.jit(static_argnums=4)
def TDMA_O_solve(u: Array, ld: Array, d: Array, ud: Array, n: int) -> Array:
    # for i in range(1, n):
    #    u[i] -= ld[i-1]*u[i-1]
    # u[n-1] = u[n-1]/d[n-1]
    # for i in range(n-2, -1, -1):
    #    u[i] = (u[i] - ud[i]*u[i+1])/d[i]
    def forward_body(i, u):
        return u.at[i].set(u[i] - ld[i - 1] * u[i - 1])

    u = jax.lax.fori_loop(1, n, forward_body, u)

    u = u.at[n - 1].set(u[n - 1] / d[n - 1])

    def backward_body(i, u):
        return u.at[i].set((u[i] - ud[i] * u[i + 1]) / d[i])

    u = jax.lax.fori_loop(0, n - 1, lambda i, u: backward_body(n - 2 - i, u), u)

    return u


@jax.jit
def TDMA_O_Solve(u: Array, ld: Array, d: Array, ud: Array) -> Array:

    N: int = u.shape[0]

    def lu(i: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        ld, d = carry
        ld = ld.at[i - 1].set(ld[i - 1] / d[i - 1])
        d = d.at[i].set(d[i] - ld[i - 1] * ud[i - 1])
        return ld, d

    ld, d = jax.lax.fori_loop(1, N, lu, (ld, d))

    def fwd(i: int, u: Array) -> Array:
        return u.at[i].set(u[i] - ld[i - 1] * u[i - 1])

    u = jax.lax.fori_loop(1, N, fwd, u)

    u = u.at[N - 1].set(u[N - 1] / d[N - 1])

    def bwd(n: int, u: Array) -> Array:
        i = N - 2 - n
        return u.at[i].set((u[i] - ud[i] * u[i + 1]) / d[i])

    u = jax.lax.fori_loop(0, N - 1, bwd, u)

    return u


@jax.jit
def TDMA_Fwd(a: Array, b: Array, c: Array):
    def fwd1(t_: Array, x: Array) -> tuple[Array, Array]:
        t = x[2] / (x[1] - x[0] * t_)
        return t, t

    return lax.scan(fwd1, jnp.array(0.0), (a, b, c))[1]


@jax.jit
def TDMA_BSolve(u0: Array, a: Array, b: Array, c_: Array):

    def fwd2(b_: Array, x: Array) -> tuple[Array, Array]:
        t = (x[0] - x[3] * b_) / (x[1] - x[3] * x[2])
        return t, t

    def bwd1(x_: Array, x: Array) -> tuple[Array, Array]:
        t = x[0] - x[1] * x_
        return t, t

    _c = jnp.concatenate((jnp.zeros(1), c_[:-1]))
    b_ = lax.scan(fwd2, u0[0] / b[0:1], (u0, b, _c, a))[1]
    x_ = lax.scan(bwd1, b_[-1], (b_, c_), reverse=True)[1]
    return x_


@jax.jit
def TDMA_Sol(u0: Array, a: Array, b: Array, c: Array):
    c_ = TDMA_Fwd(a, b, c)
    return TDMA_BSolve(u0, a, b, c_)


@jax.jit
def tridiagonal_solve_jax(dl, d, du, b):
    def fwd(carry, args):
        cp, dp = carry
        a, b, c, d = args
        cp_next = c / (b - a * cp)
        dp_next = (d - a * dp) / (b - a * cp)
        return (cp_next, dp_next), (cp, dp)

    (_, final), (cp, dp) = control_flow.scan(
        fwd, (du[0] / d[0], b[0] / d[0]), (dl[1:], d[1:], du[1:], b[1:, :])
    )

    def bwd(xn, args):
        cp, dp = args
        x = dp - cp * xn
        return x, xn

    end, ans = control_flow.scan(bwd, final, (cp, dp), reverse=True)
    return lax.concatenate((end[None], ans), 0)
