from typing import Callable, NamedTuple, Union
from functools import partial
from numbers import Number
import copy
import jax
from jax import Array
import jax.numpy as jnp
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse
import sympy as sp


n = sp.Symbol("n", real=True, integer=True, positive=True)


def diff(
    fun: Callable[[float, Array], float], k: int = 1
) -> Callable[[Array, Array], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.jit(jax.vmap(fun, in_axes=(0, None)))


def diffx(
    fun: Callable[[float, int], float], k: int = 1
) -> Callable[[Array, int], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.vmap(fun, in_axes=(0, None))


def jacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    for _ in range(k):
        fun = jax.jacfwd(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0)


@partial(jax.jit, static_argnums=(0, 3))
def evaluate(
    fun: Callable[[tuple[float], Array, tuple[int]], Array],
    x: tuple[float],
    c: Array,
    axes: tuple[int] = (0,),
) -> Array:
    assert len(x) == len(axes)
    dim: int = len(c.shape)
    for xi, ax in zip(x, axes):
        axi: int = dim - 1 - ax
        c = jax.vmap(fun, in_axes=(None, axi), out_axes=axi)(xi, c)
    return c


@jax.jit
def matmat(a: Union[Array, BCOO], b: Union[Array, BCOO]) -> Array:
    return a @ b


@jax.jit
def matmat_bcoo(a: BCOO, b: BCOO) -> BCOO:
    return a @ b


def to_sparse(a: Array, tol: float) -> sparse.BCOO:
    b: float = jnp.linalg.norm(a)
    a = jnp.choose(jnp.array(jnp.abs(a) > tol * b, dtype=int), (jnp.zeros_like(a), a))
    return sparse.BCOO.fromdense(a)


def from_dense(a: Array, tol: float = 1e-10) -> sparse.BCOO:
    z = jnp.where(jnp.abs(a) < tol, jnp.zeros(a.shape), a)
    return sparse.BCOO.from_scipy_sparse(scipy_sparse.csr_matrix(z))


class Domain(NamedTuple):
    lower: Number
    upper: Number


class BoundaryConditions(dict):
    """Boundary conditions as a dictionary"""

    def __init__(self, bc: dict, domain: Domain = None):
        bcs = {"left": {}, "right": {}}
        bcs.update(copy.deepcopy(bc))
        dict.__init__(self, bcs)

    def orderednames(self) -> list[str]:
        return ["L" + bci for bci in sorted(self["left"].keys())] + [
            "R" + bci for bci in sorted(self["right"].keys())
        ]

    def orderedvals(self) -> list[Number]:
        ls = []
        for lr in ("left", "right"):
            for key in sorted(self[lr].keys()):
                val = self[lr][key]
                ls.append(val[1] if isinstance(val, (tuple, list)) else val)
        return ls

    def num_bcs(self) -> int:
        return len(self.orderedvals())


def bnd_values(
    alpha: Number,
    beta: Number,
    k: int = 0,
    gn: Callable[[Number, Number, int], sp.Expr] = 1,
) -> tuple[
    Callable[[Union[int, sp.Symbol]], sp.Expr],
    Callable[[Union[int, sp.Symbol]], sp.Expr],
]:
    """Return lambda function for computing boundary values

    See if this could be done better
    """
    if gn == 1:
        gn = lambda a, b, n: 1

    def gam(i: int) -> sp.Expr:
        return sp.rf(i + alpha + beta + 1, k) * sp.Rational(1, 2 ** (k))

    return (
        lambda i: gn(alpha, beta, i)
        * (-1) ** ((k + i) % 2)
        * gam(i)
        * sp.binomial(i + beta, i - k),
        lambda i: gn(alpha, beta, i) * gam(i) * sp.binomial(i + alpha, i - k),
    )


def get_stencil_matrix(
    bcs,
    family: str,
    alpha: Number = None,
    beta: Number = None,
    gn: Callable[[Number, Number, int], sp.Expr] = 1,
) -> dict:
    """Return stencil matrix as dictionary of keys, values being diagonals and sympy expressions"""
    global bnd_values
    bnd_values = partial(bnd_values, alpha=alpha, beta=beta, gn=gn)
    bcs = BoundaryConditions(bcs)
    bc = {"D": 0, "N": 1, "N2": 2, "N3": 3, "N4": 4}
    lr = {"L": 0, "R": 1}
    lra = {"L": "left", "R": "right"}
    s = []
    r = []
    for key in bcs.orderednames():
        k, v = key[0], key[1:]
        if v in "WR":  # Robin conditions
            k0 = 0 if v == "R" else 1
            alfa = bcs[lra[k]][v][0]
            f = [bnd_values(k=k0)[lr[k]], bnd_values(k=k0 + 1)[lr[k]]]
            s.append(
                [
                    sp.simplify(f[0](n + j) + alfa * f[1](n + j))
                    for j in range(1, 1 + bcs.num_bcs())
                ]
            )
            r.append(-sp.simplify(f[0](n) + alfa * f[1](n)))
        else:
            f = bnd_values(k=bc[v])[lr[k]]
            s.append([sp.simplify(f(n + j)) for j in range(1, 1 + bcs.num_bcs())])
            r.append(-sp.simplify(f(n)))
    A = sp.Matrix(s)
    b = sp.Matrix(r)
    M = sp.simplify(A.solve(b))
    d = {0: 1}
    for i, s in enumerate(M):
        if not s == 0:
            d[i + 1] = s
    return d
