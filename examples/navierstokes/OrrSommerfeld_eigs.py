"""
Solve the Orr-Sommerfeld eigenvalue problem

Using Shen's biharmonic basis

"""

# ruff: noqa: E402
import jax

# Only as a script. ChannelFlow2D.py imports this module for its eigenmode, and
# by then jaxfun has been imported and traced: flipping the global precision at
# that point leaves float32 constants (jaxfun's Gauss-Legendre tables) in the
# same graph as float64 ones. The eigenvalue assertion below needs float64, so
# the entry point sets it and an import does not.
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

jax.config.update("jax_default_matmul_precision", "highest")

from functools import cached_property
from typing import cast

import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sympy as sp
from flax import nnx
from jax import Array
from matplotlib import pyplot as plt

from jaxfun.galerkin import (
    Composite,
    FunctionSpace,
    JAXFunction,
    Legendre,
    TestFunction,
    TrialFunction,
    inner,
    project,
)
from jaxfun.la import BaseMatrix

x = sp.Symbol("x", real=True)
n = sp.Symbol("n", integer=True)


class _ShiftInvert(scipy.sparse.linalg.LinearOperator):
    """The operator (A - sigma B)^-1 B, with a sparse LU of A - sigma B."""

    def __init__(self, A: np.ndarray, B: np.ndarray, sigma: complex) -> None:
        super().__init__(dtype=np.dtype(complex), shape=A.shape)
        self.B = scipy.sparse.csc_matrix(B)
        self.lu = scipy.sparse.linalg.splu(
            (scipy.sparse.csc_matrix(A) - sigma * self.B).tocsc()
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self.lu.solve(self.B @ x)


@nnx.dataclass
class OrrSommerfeld:
    alfa: float = 1.0
    Re: float = 8000.0
    N: int = 80
    verbose: bool = False

    def interp(
        self, y: Array, eigvals: Array, eigvectors: Array, eigval: int = 1
    ) -> tuple[int, Array, Array]:
        """Interpolate solution eigenvector and its derivative onto y

        Args:
            y : Interpolation points
            eigvals : All computed eigenvalues
            eigvectors : All computed eigenvectors
            eigval : The chosen eigenvalue, ranked with descending imaginary
                part. The largest imaginary part is 1, the second
                largest is 2, etc.
        """
        z = self.get_eigval(eigval, eigvals, self.verbose)
        nx, eigval = z[0].item(), z[1].item()
        SB = self.functionspace
        x = SB.system.x
        ev = jnp.squeeze(eigvectors[:, nx])
        phi_hat = JAXFunction(ev, SB, "phi")
        phi: Array = phi_hat(y)
        P = SB.get_orthogonal()
        dphi = project(phi_hat.diff(x), P)
        dphidy = P.evaluate(y, dphi)
        return eigval, phi, dphidy

    @cached_property
    def functionspace(self) -> Composite:
        bcs = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
        V = FunctionSpace(
            self.N,
            Legendre.Legendre,
            bcs=bcs,
            scaling=sp.sqrt(2 * (2 * n + 5)) * (2 * n + 3),
            name="V",
        )
        return cast(Composite, V)

    def assemble(self) -> tuple[Array, Array]:
        V = self.functionspace
        v = TestFunction(V)
        u = TrialFunction(V)

        Re = self.Re
        a = self.alfa
        x = V.system.x

        def mat(form: sp.Expr) -> BaseMatrix:
            """`inner` of a purely bilinear form is a matrix; say so."""
            return cast(BaseMatrix, inner(form, sparse=True))

        # (u'', v)_w
        K = mat(v * u.diff(x, 2))

        # ((1-x**2)u, v)_w
        K1 = mat(v * (1 - x**2) * u)

        # ((1-x**2)u'', v)_w
        K2 = mat(v * (1 - x**2) * u.diff(x, 2))

        # (u'''', v)_w
        Q = mat(v * u.diff(x, 4))

        # (u, v)_w
        M = mat(v * u)

        B = -Re * a * 1j * (K - a**2 * M)
        A = (
            Q
            - 2 * a**2 * K
            + (a**4 - 2 * a * Re * 1j) * M
            - 1j * a * Re * (K2 - a**2 * K1)
        )

        return A.todense(), B.todense()

    def solve(self, sigma: complex | None = None, k: int = 6) -> tuple[Array, Array]:
        """Solve the Orr-Sommerfeld eigenvalue problem A x = c B x.

        Args:
            sigma: If None, return all eigenvalues and eigenvectors. Otherwise
                return only the `k` eigenvalues nearest to `sigma`, which is
                much cheaper for large N.
            k: Number of eigenvalues returned when `sigma` is given.
        """
        if self.verbose:
            print("Solving the Orr-Sommerfeld eigenvalue problem...")
            print("Re = " + str(self.Re) + " and alfa = " + str(self.alfa))
        A, B = (np.asarray(m) for m in self.assemble())
        if sigma is None:
            # QZ on the pencil, rather than eig(inv(B) @ A): cond(B) grows like
            # N^3, and forming the inverse loses digits that QZ keeps.
            vals, vecs = scipy.linalg.eig(A, B)
        else:
            # Shift-invert Arnoldi: the eigenvalues mu of (A - sigma B)^-1 B are
            # 1 / (c - sigma), largest for c nearest sigma. The pencil is banded,
            # so the sparse LU is cheap.
            mu, vecs = scipy.sparse.linalg.eigs(
                _ShiftInvert(A, B, sigma), k=k, which="LM"
            )
            vals = sigma + 1 / mu
        # Unit norm with the largest component real, as numpy's eig returns them,
        # so that solves at different N give the same eigenfunction, phase and all.
        # QZ fixes only the magnitude.
        vecs = vecs / np.linalg.norm(vecs, axis=0)
        top = vecs[np.argmax(np.abs(vecs), axis=0), np.arange(vecs.shape[1])]
        vecs = vecs * (np.abs(top) / top)
        return jnp.asarray(vals), jnp.asarray(vecs)

    @staticmethod
    def get_eigval(
        nx: int, eigvals: Array, verbose: bool = False
    ) -> tuple[Array, Array]:
        """Get the chosen eigenvalue

        Parameters
        ----------
            nx : The chosen eigenvalue. nx=1 corresponds to the one with the
                largest imaginary part, nx=2 the second largest etc.
            eigvals : Computed eigenvalues
            verbose : Print the value of the chosen eigenvalue. Default is False.

        """
        nx: Array = jnp.atleast_1d(jnp.array(nx))
        indices = jnp.argsort(jnp.imag(eigvals))
        indi = indices[-1 * nx]
        eigval = eigvals[indi]
        if verbose:
            ev = list(eigval) if jnp.ndim(eigval) else [eigval]
            indv = list(indi) if jnp.ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indv)):
                print(f"Eigenvalue {i + 1} ({v}) = {e:2.16e}")
        return indi, eigval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orr Sommerfeld parameters")
    parser.add_argument(
        "--N", type=int, default=100, help="Number of discretization points"
    )
    parser.add_argument("--Re", default=8000.0, type=float, help="Reynolds number")
    parser.add_argument("--alfa", default=1.0, type=float, help="Parameter")
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Print results"
    )
    parser.add_argument(
        "--plot", dest="plot", action="store_true", help="Plot eigenvalues"
    )
    parser.set_defaults(verbose=False)
    parser.set_defaults(plot=False)
    args = parser.parse_args()
    plot = args.__dict__.pop("plot")
    z = OrrSommerfeld(**vars(args))
    evals, evectors = z.solve()
    d = z.get_eigval(1, evals, args.verbose)

    if args.Re == 8000.0 and args.alfa == 1.0 and args.N > 80:
        assert abs(d[1] - (0.24707506017499 + 0.002664410371114j)) < 1e-12
    if args.Re == 10000.0 and args.alfa == 1.0 and args.N > 80:
        # Mode A1 of Giannakis, Fischer and Rosner, J. Comput. Phys. 228 (2009)
        # 1188-1233, Table B.1, computed with 500 modes.
        assert abs(d[1] - (0.2375264888204708 + 0.003739670622977800j)) < 1e-13
        near = z.get_eigval(1, z.solve(sigma=0.24 + 0.0j, k=4)[0])[1]
        assert abs(near - d[1]) < 1e-13

    if plot:
        plt.figure()
        evi = evals * z.alfa
        plt.plot(evi.real, evi.imag, "o")
        plt.axis((0, 1, -10, 0.1))
        plt.show()
