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
import sympy as sp
from flax import nnx
from jax import Array
from matplotlib import pyplot as plt

from jaxfun.galerkin import (
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


@nnx.dataclass
class OrrSommerfeld:
    alfa: float = 1.0
    Re: float = 8000.0
    N: int = 80
    verbose: bool = False

    def interp(self, y, eigvals, eigvectors, eigval=1):
        """Interpolate solution eigenvector and it's derivative onto y

        Args:
            y : array
                Interpolation points
            eigvals : array
                All computed eigenvalues
            eigvectors : array
                All computed eigenvectors
            eigval : int, optional
                The chosen eigenvalue, ranked with descending imaginary
                part. The largest imaginary part is 1, the second
                largest is 2, etc.
        """
        nx, eigval = self.get_eigval(eigval, eigvals, self.verbose)
        SB = self.functionspace
        x = SB.system.x
        ev = jnp.squeeze(eigvectors[:, nx])
        phi_hat = JAXFunction(ev, SB, "phi")
        phi = phi_hat(y)
        P = SB.get_orthogonal()
        dphi = project(phi_hat.diff(x), P)
        dphidy = P.evaluate(y, dphi)
        return eigval, phi, dphidy

    @cached_property
    def functionspace(self):
        bcs = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
        return FunctionSpace(
            self.N,
            Legendre.Legendre,
            bcs=bcs,
            scaling=sp.sqrt(2 * (2 * n + 5)) * (2 * n + 3),
            name="V",
        )

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

    def solve(self):
        """Solve the Orr-Sommerfeld eigenvalue problem"""
        if self.verbose:
            print("Solving the Orr-Sommerfeld eigenvalue problem...")
            print("Re = " + str(self.Re) + " and alfa = " + str(self.alfa))
        A, B = self.assemble()
        return jnp.linalg.eig(jnp.linalg.inv(B) @ A)

    @staticmethod
    def get_eigval(nx, eigvals, verbose=False):
        """Get the chosen eigenvalue

        Parameters
        ----------
            nx : int
                The chosen eigenvalue. nx=1 corresponds to the one with the
                largest imaginary part, nx=2 the second largest etc.
            eigvals : array
                Computed eigenvalues
            verbose : bool, optional
                Print the value of the chosen eigenvalue. Default is False.

        """
        indices = jnp.argsort(jnp.imag(eigvals))
        indi = indices[-1 * jnp.array(nx)]
        eigval = eigvals[indi]
        if verbose:
            ev = list(eigval) if jnp.ndim(eigval) else [eigval]
            indi = list(indi) if jnp.ndim(indi) else [indi]
            for i, (e, v) in enumerate(zip(ev, indi)):
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
        assert abs(d[1] - (0.24707506017508621 + 0.0026644103710965817j)) < 1e-12

    if plot:
        plt.figure()
        evi = evals * z.alfa
        plt.plot(evi.real, evi.imag, "o")
        plt.axis((0, 1, -10, 0.1))
        plt.show()
