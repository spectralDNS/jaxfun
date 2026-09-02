# Incompressible Navier-Stokes in a periodic channel, two-dimensional
#
#   u_t + Grad(u)⋅u = -Grad(p) + nu*Div(Grad(u)) + f*i
#   Div(u)          = 0
#
# on (x, y) in [0, Lx] x (-1, 1), periodic in x, no-slip walls, with a constant
# streamwise body force f standing in for a driving pressure gradient. Fourier
# along axis 0 and the polynomial (wall-normal) direction along axis 1, which is
# jaxfun's convention and the order the multi-device sharding path requires.
#
# The class is named KMM2D after the method rather than after the equations,
# which is what distinguishes it: eliminating the pressure into a fourth-order
# equation for the wall-normal velocity, carrying the k=0 mean flow as its own
# one-dimensional problem, and recovering the streamwise velocity from
# continuity. That formulation is
#
#   Kim J, Moin P, Moser R. Turbulence statistics in fully developed channel
#   flow at low Reynolds number. Journal of Fluid Mechanics. 1987;177:133-166.
#   doi:10.1017/S0022112087000892
#
# The 2D is part of the method, not just the geometry: in two dimensions the
# elimination closes on the wall-normal velocity alone, where in three it also
# needs an equation for the wall-normal vorticity.
#
# This is the velocity-only core of shenfun's Rayleigh-Benard formulation,
# https://shenfun.readthedocs.io/en/latest/rayleighbenard.html. RayleighBenard.py
# subclasses it and adds the temperature equation and its buoyancy coupling.
#
# PRESSURE ELIMINATION
#
# Applying Div(Grad(.)) to the y-momentum equation and substituting
# Div(Grad(p)) = -Div(H), where H = Grad(u)⋅u, removes the pressure:
#
#   (Div(Grad(v)))_t = nu*Div(Grad(Div(Grad(v)))) + H_x,xy - H_y,xx
#
# with v = v_y = 0 at the walls -- a fourth-order equation whose mass operator is
# the Laplacian. H is taken in rotational form H = (-v*w, u*w), w = v_x - u_y,
# which leaves a Grad(|u|^2/2) absorbed into the (eliminated) pressure. A subclass
# may add further explicit forcing to this equation -- buoyancy, in the Rayleigh-Benard
# case.
#
# THE UNKNOWNS
#
#   v    wall-normal velocity   VB = F x B, biharmonic     transient, mass = Lap
#   u    streamwise velocity    VD = F x D                 algebraic, continuity
#   u0   mean flow, k = 0 only  1-D D1                     transient
#
# plus one transported scalar per subclass extension. Continuity
# i*k*u_hat + v_hat_y = 0 determines u for every k except k = 0, where it
# degenerates. There the x-momentum equation loses its pressure gradient outright
# (dp/dx -> i*k*p_hat = 0) and closes the system on its own:
#
#   u0_t = nu*u0_yy - <H_x>_x + f
#
# u0 carries the entire mean profile: the Poiseuille base flow of a driven
# channel, and in Rayleigh-Benard the "wind" driven from zero by the Reynolds
# stress d<u v>/dy.
#
# WHY THIS IS A TAILORED SOLVER
#
# The u equation is algebraic for k != 0 and transient for k = 0, while
# SystemIntegrator classifies a whole equation as one or the other. No symbolic
# weak form can express the split, because a Fourier multiplier that is 1 at
# k = 0 and 0 elsewhere is not a differential operator. So the stage loop is
# written out here -- but composed from framework pieces rather than hand-rolled:
# TimeStepper supplies the batched step driver, and one IMEXRungeKutta per
# transient equation supplies the mass/stiffness split, the stage operators
# (cached per distinct Butcher diagonal and factorized outside jit) and the
# Butcher accumulation. Their weak forms carry no nonlinear terms at all; every
# explicit term is computed here and handed to `stage()`, which takes its
# nonlinear and linear caches as plain arrays.
#
# TWO PROPERTIES THAT COME OUT FOR FREE
#
# v_hat[k=0] stays exactly zero: every term on the v-equation right-hand side
# carries at least one d/dx, and d/dx is a diagonal multiply by i*k that is
# exactly 0 there. This is shenfun's u_hat[k,0] = 0 with no special handling.
#
# The continuity solve is pointwise exact, not merely a Galerkin projection:
# v in VB makes v_y vanish at both walls, so v_y lies in VD exactly and the weak
# equation has the pointwise solution as its unique Galerkin solution.
#
# THE k = 0 PIN
#
# The continuity operator is a single TPMatrix, diag(2*pi*i*k) x M_y, whose
# Fourier entry at k = 0 is exactly zero. Pinning that row to the identity makes
# the operator non-singular *and* turns the k = 0 row into a free slot: the solve
# then returns u_hat[0] = M_y^-1 rhs[0], so setting rhs[0] = M_y @ u0 injects the
# mean flow and leaves every other wavenumber bit-identical. Pinning the M x M
# Fourier factor keeps the fast per-wavenumber banded solve; pinning the
# flattened Kronecker matrix would destroy it.
#
# DEALIASING: 3/2 IN FOURIER, NONE IN THE WALL-NORMAL
#
# The matrices are assembled exactly from the precomputed composite stencils, so
# quadrature error can only enter through the transform pair around the pointwise
# products. Both directions alias there. A quadratic product of two fields that
# fill the grid has twice the bandwidth, and neither the M-point FFT nor the
# N-point Gauss quadrature can carry the excess; it folds back onto the retained
# modes. Measured at N=32, as the amplitude with which an unrepresentable mode m
# lands on a retained one:
#
#   Fourier     exact fold at amplitude 1: mode k1+k2 lands on k1+k2-M
#   Chebyshev   exact fold at amplitude 1: T_{2N-j} is -T_j on the Gauss points
#   Legendre    no exact fold: 0.97 at m=N+1, falling to 0.1-0.2 and spread over
#               several modes by m=2N
#
# So this is not a Fourier-only problem, and the wall-normal direction is not
# spared by using a Gauss quadrature rather than an FFT -- under Chebyshev the
# fold is exactly as clean and exactly as full-amplitude as Fourier's. Nor is the
# 3/2 rule mandatory in the streamwise direction: in every direction the size of
# the error is set by how much of the product's spectrum actually reaches the
# fold, so a run with nothing left at the top of either spectrum aliases in
# neither, padded or not.
#
# What the asymmetric setting here rests on is measurement and cost, not
# principle. Measured on a developed Ra=1e6-family field at 128 x 64 under
# Legendre -- the change in the nonlinear terms from dropping the wall-normal
# padding, relative to their own size, against the fraction of the temperature
# spectrum left in the top third:
#
#   Ra      spectrum tail    d(NL_v)     d(NL_u0)    d(NL_T)
#   1e4     3.8e-07          5.3e-11     1.5e-11     3.6e-12
#   1e5     1.2e-04          3.7e-04     2.6e-08     4.5e-06
#   1e6     9.9e-04          1.0e-02     1.1e-04     1.9e-03
#
# Dropping the wall-normal padding buys about 23% of the runtime and costs 5e-11
# once the run is resolved, which is why it is off here. It is not free at the
# margin: at Ra=1e6 on this grid the run is only marginally resolved and dropping
# it perturbs the v nonlinear term by 1%. The streamwise 3/2 rule is kept because
# it is cheap, not because a resolved run needs it. Either way the guard is the
# `tail` diagnostic -- below ~1e-5 padding is pointless in both directions, at
# 1e-3 it is not; past that, raise M and N rather than trusting the answer. Those
# numbers are Legendre, and have not been remeasured for Chebyshev.
#
# THE FIELDS ARE REAL, SO HALF THE SPECTRUM IS REDUNDANT
#
# Every physical field here is real, so its spectrum is Hermitian: the negative
# wavenumbers are the conjugates of the positive ones and carry no information.
# The streamwise direction is therefore built with `TensorProduct(..., real=True)`,
# which stores only k = 0, ..., M/2 and transforms with rfft/irfft. Nothing is
# approximated -- the equations for -k are the conjugates of those for +k -- but
# everything downstream runs on half the data: the banded per-wavenumber solves,
# the Butcher accumulation, the stencils, and the wall-normal matrix products
# that dominate the nonlinear term.
#
# The Nyquist mode is the one place a half spectrum is not simply the same thing
# written down once. A real field cannot carry a phase there, d/dx of it is not
# representable, and the operator matrices use the raw wavenumber while the
# transforms zero it -- so the two conventions disagree unless it vanishes.
# `_zero_nyquist` holds it at zero at every stage, and nothing is lost.
#
# CHOICE OF BASIS AND TEST SPACE
#
# `polynomial` picks the wall-normal basis and `kind` picks how it is tested.
# The two are not independent: what matters is whether the resulting operators
# stay banded, because every implicit solve here is a banded LU, and its cost
# grows sharply with the bandwidth -- the forward/backward substitution runs as a
# `lax.scan` whose carry widens with the band. Dropping the continuity operator
# from four y-diagonals to three, by testing it against the Galerkin space, was
# worth 1.18x on the whole solver. Widest band in the v equation, measured:
#
#                            N=24   N=48   N=96
#   Legendre   Galerkin         5      5      5     <- the default
#   Legendre   Petrov-Galerkin  7      7      7
#   Chebyshev  Galerkin        10     24     87
#   Chebyshev  Petrov-Galerkin  7      7      7     <- the fast pairing
#   ChebyshevU Galerkin        10     22     74
#
# So pair LEGENDRE with GALERKIN and CHEBYSHEV with PETROV_GALERKIN. The other
# two cells of that square are legal and give the same answer, but neither is
# ever the right choice:
#
#   Legendre + PG      works, but Legendre's Galerkin operators are already
#                      banded, and PG only widens them (7 against 5). PG buys
#                      sparsity that Legendre does not need.
#   Chebyshev + G      works, but differentiating a Chebyshev expansion in
#                      coefficient space is dense upper triangular -- T_n' spreads
#                      over every lower T_k of the same parity -- where Legendre's
#                      weight of 1 lets integration by parts collapse the same
#                      operators to a few diagonals. So the band grows like N.
#                      This is exactly what PG exists to avoid: the ChebPhi test
#                      functions restore a fixed bandwidth.
#
# ChebyshevU has no PG test space implemented, so it is Galerkin only, and
# Galerkin leaves it dense -- asking for PG raises NotImplementedError from the
# space rather than falling back. It is here for completeness.
#
# Which of the two is faster is a performance question, not a correctness one,
# and the answer is machine-dependent -- measure it rather than trusting a
# number written here. Chebyshev transforms are DCTs, O(N log N), against
# Legendre's cached Vandermonde matrix-multiply, O(N^2), while Legendre's
# operators are narrower; so Chebyshev gains as N grows, and gains as M shrinks
# because the streamwise FFT work is identical for both and dilutes the
# difference. Where that leaves a given (M, N) on a given machine is not
# predictable from here.
#
# Accuracy is the more portable reason to prefer Chebyshev: its transform
# round-trips at machine epsilon where the dense Legendre Vandermonde
# accumulates a few hundred ulp.
#
# Three traps if you do measure. `solve()` carries a fixed cost of order half a
# second per call, so time `_step_impl` in a `fori_loop` instead. Machines drift
# by a few percent between processes, so time both bases in one process,
# alternating. And a diverged run is *faster* than a healthy one, since a
# resolved spectral tail reaches denormals while NaN arithmetic does not: check
# `jnp.isfinite`, and that no coefficient is denormal, before believing anything.
#
# RESOLUTION AND STEP SIZE ARE COUPLED
#
# Only the diffusive terms are implicit, so dt is limited by the advective
# Courant number and has to come down roughly in step with the resolution -- and
# it is the *wall-normal* resolution that binds, the Gauss node spacing collapsing
# like 1/N^2 at the walls for every basis offered here. The `courant` diagnostic
# reports where a run stands. It reads ~1 at the stability boundary -- measured
# on Orr-Sommerfeld, 0.76 runs and 1.15 diverges -- but that threshold is
# calibrated to the 3/2 padding and to ARS443 rather than being a property of
# the method; see the diagnostic's own docstring.
#
# CHOICE OF TABLEAU
#
# Any globally stiffly accurate IMEX tableau works -- ARS443, ARS222 and
# IMEX_EULER; the Kennedy-Carpenter ARK family is only implicitly stiffly
# accurate and is rejected. But advection is handled by the *explicit* table, so
# the explicit part's imaginary-axis stability decides whether a scheme is usable
# at all. Measured on the Orr-Sommerfeld growth rate of OrrSommerfeld.py
# (Re=8000, T=50), as relative error against linear theory:
#
#   dt            0.05        0.02        0.01
#   ARS443        7.6e-06     5.1e-07     6.6e-08     3rd order
#   ARS222        unstable    1.3e-05     3.0e-06     2nd order
#   IMEX_EULER    diverges    diverges    diverges
#
# IMEX_EULER cannot integrate this problem at any step size: its explicit half is
# forward Euler, whose stability region touches the imaginary axis only at the
# origin, so pure advection is unconditionally unstable. ARS443 is the default.
#
# VERIFICATION (not here -- this module has no entry point of its own)
#
# Two demos drive this solver and check it against theory: OrrSommerfeld.py
# evolves an eigenmode on plane Poiseuille flow and compares its growth rate
# with linear stability theory, which exercises advection, the biharmonic
# operator, continuity and the mean flow at once; RayleighBenard.py subclasses
# it and checks the onset of convection.
#
# Spatial discretization: Fourier x (Legendre Galerkin | Chebyshev Petrov-Galerkin)
# Time discretization: any globally stiffly accurate IMEX Runge-Kutta tableau
# ruff: noqa: E402
from enum import StrEnum
from functools import partial
from typing import Any, Literal, cast, overload

import jax

# Before any jaxfun import, so nothing is built at the wrong precision. Kept here
# rather than left to the demos because an importer that forgets gets float32
# silently: OrrSommerfeld.py seeds its eigenmode at amplitude 1e-7 on a base flow
# of order 1, and in float32 the growth rate it measures comes out negative
# (-5.5e-04 against a theoretical +2.7e-03).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import sympy as sp
from flax import nnx
from jax import shard_map

from jaxfun import galerkin
from jaxfun.galerkin import (
    Fourier,
    FunctionSpace,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.composite import PGComposite
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.integrators import ARS443, IMEXRungeKutta, IMEXTableau
from jaxfun.integrators.base import TimeStepper
from jaxfun.la import BaseMatrix, DiaMatrix, TPMatrix
from jaxfun.operators import Constant, Div, Grad
from jaxfun.sharding import (
    batched_physical_sharding,
    batched_spectral_sharding,
)
from jaxfun.typing import Array, PolynomialKind, ScalarPadding, TestSpaceKind
from jaxfun.utils.common import Domain
from jaxfun.utils.operator_tools import assemble_linear_term

SOLVE: dict[str, Any] = {"auto_threshold": 100000}
ASSEMBLE: dict[str, Any] = {"sparse": True, "sparse_tol": 1000}

# The wall-normal bases this solver accepts, mapped to their space class.
POLYNOMIALS: dict[PolynomialKind, type[OrthogonalSpace]] = {
    PolynomialKind.LEGENDRE: galerkin.Legendre.Legendre,
    PolynomialKind.CHEBYSHEV: galerkin.Chebyshev.Chebyshev,
    PolynomialKind.CHEBYSHEVU: galerkin.ChebyshevU.ChebyshevU,
}


def linear_operator(expr: sp.Expr) -> BaseMatrix:
    """Assemble one linear weak form into its operator."""
    A = assemble_linear_term(expr, **ASSEMBLE)[0]
    assert A is not None, f"expected a non-empty linear form, got {expr}"
    return A


class VelocityKind(StrEnum):
    SPECTRAL = "spectral"
    PHYSICAL = "physical"
    BOTH = "both"


def snapshot_times(dt: float, steps: int, n_batches: int) -> Array:
    """Return the times at which `TimeStepper.solve` records its snapshots.

    Mirrors the batching in `TimeStepper.solve`: one snapshot at t=0, one per
    completed batch, and -- when `n_batches` does not divide `steps` -- a final
    one after the shorter remainder chunk. That last interval is *not* the same
    length as the others, so a `linspace` over the snapshot count silently
    mislabels the time axis and biases anything fitted against it.
    """
    count = min(n_batches, steps)
    batch_len = steps // count
    times = [i * batch_len * dt for i in range(count + 1)]
    if steps - count * batch_len:
        times.append(steps * dt)
    return jnp.asarray(times)


def growth_rate_of(values: Array, times: Array) -> float:
    """Return the exponential growth rate fitted over the second half of a run.

    `times` is truncated to the number of samples actually recorded, because
    `TimeStepper.solve` stops early once the state goes non-finite -- a diverged
    run therefore returns fewer snapshots than the batching predicts, and the
    fit below then reports nan rather than a spurious rate.
    """
    times = times[: values.shape[0]]
    half = values.shape[0] // 2
    return float(jnp.polyfit(times[half:], jnp.log(values[half:]), 1)[0])


class KMM2D(TimeStepper[tuple[Array, ...]]):
    """Kim-Moin-Moser channel flow in two dimensions.

    Incompressible Navier-Stokes in a periodic channel, pressure eliminated into
    a fourth-order equation for the wall-normal velocity after Kim, Moin & Moser
    (JFM 177:133-166, 1987).

    The state is `(v_hat, u0, *scalars)`: the wall-normal velocity, the mean
    streamwise profile, and one array per transported scalar contributed by a
    subclass. The streamwise velocity is never stored -- it is recomputed from v
    and the mean flow at every stage by `velocity`, so it cannot drift out of
    sync with them.

    Subclasses extend the system through four hooks: `scalar_integrators`,
    `scalar_initial`, `scalar_terms` and `buoyancy`. Everything else -- the
    spaces, the velocity equations, the stage loop -- is shared.
    """

    def __init__(
        self,
        M: int,
        N: int,
        Lx: float,
        nu: float,
        *,
        body_force: float = 0.0,
        tableau: IMEXTableau = ARS443,
        time: tuple[float, float] | None = None,
        padding: tuple[int, int] | None = None,
        polynomial: PolynomialKind = PolynomialKind.LEGENDRE,
        kind: TestSpaceKind = TestSpaceKind.GALERKIN,
    ) -> None:
        """Assemble the velocity spaces, operators and sub-integrators.

        Args:
            M: Number of Fourier modes along the periodic direction.
            N: Number of modes along the wall-normal direction.
            Lx: Width of the periodic box.
            nu: Kinematic viscosity.
            body_force: Constant streamwise forcing, standing in for a driving
                pressure gradient. Being constant it has only a k=0 Fourier
                component, so it enters the mean-flow equation alone.
            tableau: Any *globally* stiffly accurate IMEX Runge-Kutta tableau,
                so that the last stage is the accepted solution.
            time: Optional default integration interval.
            padding: Shape of real space. Only required if padding is used,
                otherwise real shape defaults to M, N
            polynomial: Polynomial basis for the wall-normal direction, one
                of the keys of `POLYNOMIALS`, by member name or short form.
                See "CHOICE OF BASIS AND TEST SPACE" in the header: pair
                LEGENDRE with GALERKIN and CHEBYSHEV with PETROV_GALERKIN.
            kind: Test space kind, either GALERKIN or PETROV_GALERKIN. Short
                forms G or PG. CHEBYSHEVU has no PG test space yet and
                raises NotImplementedError if asked for one.
        """
        if not tableau.is_stiffly_accurate:
            raise ValueError(
                "This solver takes the last stage as the accepted solution, so it "
                "needs a globally stiffly accurate tableau (both the explicit and "
                "the implicit table satisfying A[-1] == b). Try ARS443, ARS222 or "
                "IMEX_EULER; the Kennedy-Carpenter ARK schemes are only implicitly "
                "stiffly accurate and would need the final recombination."
            )
        polynomial = PolynomialKind.coerce(polynomial)
        kind = TestSpaceKind.coerce(kind)
        PG = kind is TestSpaceKind.PETROV_GALERKIN
        if polynomial not in POLYNOMIALS:
            raise NotImplementedError(
                f"{polynomial.name} is not available here; pick one of "
                f"{', '.join(p.name for p in POLYNOMIALS)}."
            )
        polspace = POLYNOMIALS[polynomial]

        self.time = time
        self.tableau = nnx.static(tableau)
        self.nu, self.Lx = nnx.static(nu), nnx.static(Lx)
        self.nyquist = nnx.static(M // 2)
        self.pad = nnx.static((M, N) if padding is None else padding)
        hom = {"left": {"D": 0}, "right": {"D": 0}}
        bih = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
        F = FunctionSpace(M, Fourier.Fourier, domain=Domain(0, Lx), name="F")
        D = FunctionSpace(N, polspace, bcs=hom, name="D")
        B = FunctionSpace(N, polspace, bcs=bih, name="B")
        VD = TensorProduct(F, D, name="VD", real=True)
        VB = TensorProduct(F, B, name="VB", real=True)
        # `real=True` substituted the half spectrum, which stores M/2 + 1
        # wavenumbers rather than M, plus whatever padding the device count
        # needs to split them.
        F = cast(Fourier.RFourier, VD.basespaces[0])

        # Whether the transforms below take their distributed path. Decided
        # once, from sizes rather than from any array's placement, so the
        # single-device path is chosen at construction and the code that runs
        # there is unchanged. Both conditions are what `lax.all_to_all(tiled=
        # True)` needs of the two axes it swaps: the stored wavenumber count on
        # the way out, the dealiased quadrature count on the way in. The first
        # is `RFourier`'s job and always holds; the second is this solver's, and
        # a padding that fails it keeps the local path -- losing parallelism,
        # not correctness.
        n_dev = len(jax.devices())
        self.sharded = nnx.static(
            n_dev > 1 and F.N % n_dev == 0 and self.pad[1] % n_dev == 0
        )

        # Convection H and the scalar fluxes satisfy no boundary conditions, so they
        # live in the orthogonal space.
        Wo = VD.get_orthogonal()
        # The mean flow is genuinely one-dimensional: it is the k=0 mode alone,
        # and at k=0 every 2-D operator reduces to its y-factor exactly.
        D1 = VD.basespaces[1]

        if PG:
            BP = B.get_testspace("PG", name="BP")
            PB = TensorProduct(F, BP, name="PB")
            P1 = cast(PGComposite, D1).get_testspace("PG", name="P1")

        else:
            PB = VB
            P1 = D1

        self.F = nnx.static(F)
        # d/dx in coefficient space: the same multiplier `Fourier.derivative_coeffs`
        # applies for a first derivative, kept here so the vorticity can be formed
        # in coefficient space and ride along in the batched transform below.
        self.ikx = nnx.data(
            1j * F.wavenumbers(eliminate_highest_freq=True) * float(F.domain_factor)
        )
        self.system = nnx.static(VD.system)
        self.VD, self.VB = nnx.static(VD), nnx.static(VB)
        self.Wo, self.D1 = nnx.static(Wo), nnx.static(D1)
        self.PB, self.P1 = nnx.static(PB), nnx.static(P1)
        self.polspace = nnx.static(polspace)
        self.testkind = nnx.static(kind)

        x, y = VD.system.base_scalars()
        t = VD.system.base_time()
        nu_c = Constant("nu", nu)

        u = TrialFunction(VD, name="u")
        w = TestFunction(VD, name="w")
        v = TrialFunction(VB, name="v", transient=True)
        q = TestFunction(PB, name="q")
        g = TrialFunction(Wo, name="g")
        u1 = TrialFunction(D1, name="u1", transient=True)
        w1 = TestFunction(P1, name="w1")

        # Purely linear weak forms: every explicit term is supplied by `step`.
        # This is necessary for highly optimized solvers that performs several
        # tricks for efficiency.
        eq_v = ((Div(Grad(v))).diff(t) - nu_c * Div(Grad(Div(Grad(v))))) * q
        eq_0 = (u1.diff(t) - nu_c * u1.diff(y, 2)) * w1
        opts: dict[str, Any] = {**ASSEMBLE, "solver_options": SOLVE, "tableau": tableau}
        self.gv = nnx.data(
            IMEXRungeKutta(eq_v, initial=jnp.zeros(VB.num_dofs, dtype=complex), **opts)
        )
        self.g0 = nnx.data(IMEXRungeKutta(eq_0, initial=jnp.zeros(D1.num_dofs), **opts))

        A_div = linear_operator(u.diff(x, 1) * w)
        assert isinstance(A_div, TPMatrix)

        A_kx = A_div.mats[0]
        assert isinstance(A_kx, DiaMatrix)
        assert float(jnp.abs(A_kx.diagonal(0)[0])) == 0.0, (
            "the k=0 row of the continuity operator must be singular"
        )
        self.A_pin = nnx.data(
            TPMatrix(
                [A_kx.pin({0: 1.0}).matrix, A_div.mats[1]],
                A_div.coefficient,
                A_div.global_indices,
            )
        )
        # Note that the global matrix A_pin quite remarkably leads to a distributed
        # solve, with no communication between the devices. The solve is done in
        # parallel, with each device solving its local part of the system independently.

        self.My = nnx.data(A_div.mats[1])
        self.C_v = nnx.data(linear_operator(v.diff(y, 1) * w))
        # v equation: + H_x,xy - H_y,xx
        self.C_hx = nnx.data(linear_operator(g.diff(x, 1).diff(y, 1) * q))
        self.C_hy = nnx.data(linear_operator(-g.diff(x, 2) * q))

        h1 = TrialFunction(D1.get_orthogonal(), name="h1")
        self.G = nnx.data(linear_operator(-w1 * h1))

        # A constant force has only a k=0 component, so it lands entirely on u0.
        self.f0 = nnx.data(
            body_force * P1.scalar_product(jnp.ones(P1.shape[0]))
            if body_force
            else None
        )
        # Volume of the domain in scalar-product units, for exact averages.
        self.vol = nnx.static(float(Wo.scalar_product(jnp.ones(Wo.shape))[0, 0].real))

    # -- extension points --------------------------------------------------

    @property
    def scalar_integrators(self) -> tuple[IMEXRungeKutta, ...]:
        """Sub-integrators for transported scalars, appended to the state."""
        return ()

    def scalar_initial(self) -> tuple[Array, ...]:
        """Initial coefficients for each transported scalar."""
        return ()

    def scalar_terms(
        self, u_p: Array, v_p: Array, scalars: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        """Explicit right-hand side of each transported scalar equation."""
        return ()

    def buoyancy(self, scalars: tuple[Array, ...]) -> Array | None:
        """Extra explicit forcing on the wall-normal momentum equation."""
        return None

    def extra_diagnostics(self, state: tuple[Array, ...]) -> dict[str, float]:
        """Diagnostics contributed by a subclass."""
        return {}

    # -- fields ------------------------------------------------------------

    @property
    def integrators(self) -> tuple[IMEXRungeKutta, ...]:
        """Every sub-integrator, in state order."""
        return (self.gv, self.g0) + self.scalar_integrators

    def velocity(self, v_hat: Array, u0: Array) -> Array:
        """Return the streamwise velocity: continuity for k != 0, u0 at k = 0."""
        rhs = (-(self.C_v @ v_hat)).at[0].set(self.My @ (u0 + 0j))
        return self.A_pin.solve(rhs)

    @overload
    def velocity_from_state(
        self,
        state: tuple[Array, ...],
        pad: tuple[int, int] | None = None,
        kind: Literal[VelocityKind.SPECTRAL, VelocityKind.PHYSICAL] = ...,
    ) -> tuple[Array, Array]: ...
    @overload
    def velocity_from_state(
        self,
        state: tuple[Array, ...],
        pad: tuple[int, int] | None = None,
        *,
        kind: Literal[VelocityKind.BOTH],
    ) -> tuple[Array, Array, Array, Array]: ...
    def velocity_from_state(
        self,
        state: tuple[Array, ...],
        pad: tuple[int, int] | None = None,
        kind: VelocityKind = VelocityKind.PHYSICAL,
    ) -> tuple[Array, Array] | tuple[Array, Array, Array, Array]:
        v_hat, u0 = state[0], state[1]
        u_hat = self.velocity(v_hat, u0)
        if kind == VelocityKind.SPECTRAL:
            return u_hat, v_hat
        u_p = self.VD.backward(u_hat, N=pad)
        v_p = self.VB.backward(v_hat, N=pad)
        if kind == VelocityKind.PHYSICAL:
            return u_p, v_p
        return u_hat, v_hat, u_p, v_p

    # The transform is split into its two directions so that the vorticity can be
    # assembled in between: v_x and u_y need different wall-normal matrices, but
    # they are only ever used as a difference, so subtracting them while x is
    # still in coefficient space turns two streamwise transforms into one.
    #
    # That is the only reason these are not
    #
    #   u_p, v_p, vx_p = self.Wo.backward_batch(stack, N=self.pad)
    #   uy_p           = self.Wo.backward_primitive(cu, k=(0, 1), N=self.pad)
    #
    # which is otherwise the same work in three lines. Measured on the
    # Rayleigh-Benard configuration (128 x 64, 3/2 padded), that costs 2.85-3.04
    # ms per step against 2.77 ms here -- the extra padded streamwise transform,
    # 3-10%. Each half below is the two bases' own 1-D transform applied
    # along its axis and batched over the fields, which is what
    # `TensorProductSpace` would do internally, in the same axis order:
    # wall-normal first, so the streamwise padding never inflates the matrix
    # products.

    # HOW THE THREE ARE DISTRIBUTED
    #
    # Each carries its own sharding, and they compose into the two-phase shape a
    # separable transform wants -- the wall-normal direction evaluated while the
    # wavenumbers are split, the streamwise direction while the *quadrature
    # points* are split instead, and one `all_to_all` to swap which is which:
    #
    #   _wall_normal   k-sharded -> k-sharded    no communication
    #   _streamwise    k-sharded -> y-sharded    one all_to_all
    #   _forward       y-sharded -> k-sharded    one all_to_all
    #
    # The pointwise products in between are elementwise on y-sharded arrays, so
    # they need nothing. That is also what lets `scalar_terms` stay exactly as it
    # is: it is handed y-sharded physical fields and calls these same three, so a
    # subclass never sees the distinction.
    #
    # The alternative -- letting the compiler partition these automatically --
    # does not work, because `_streamwise` and `_forward` transform *along* the
    # split axis. GSPMD resolves that by gathering the whole Fourier axis onto
    # every device at every stage, which is not a distribution of the work at
    # all: measured at M=34, N=128 it left 8 all-gathers in one step and ran 2.1x
    # slower than one device.

    def _wall_normal(self, *coeffs: Array, ky: int = 0) -> Array:
        """Evaluate the wall-normal direction, leaving x in coefficient space.

        Batched over `coeffs`: every field goes through the same Vandermonde
        (for Legendre) -- `ky` picks which derivative of it -- so the matrix
        product runs once on the stacked fields rather than once each. Fields
        wanting a different `ky` need their own call; that is the only constraint
        on what can share a batch.

        Communicates nothing -- every wavenumber's wall-normal transform is
        independent -- but still goes through `shard_map` when distributed. The
        batching is why: folding the fields into the wavenumber axis is a
        concatenation *along the split axis*, which the compiler would have to
        resolve by redistributing. Inside the kernel the same concatenation is
        local, and the arithmetic is the one big matrix product it is on one
        device.
        """
        Nq = self.pad[1]
        yspace = self.Wo.basespaces[1]
        yback = jax.vmap(partial(yspace.backward_primitive, k=ky, N=Nq))
        nf = len(coeffs)

        def local(c: Array) -> Array:
            nk = c.shape[1]
            return yback(c.reshape(nf * nk, -1)).reshape(nf, nk, Nq)

        stacked = jnp.stack(coeffs)
        if not self.sharded:
            return local(stacked)
        return shard_map(
            local,
            mesh=batched_spectral_sharding.mesh,
            in_specs=(batched_spectral_sharding.spec,),
            out_specs=batched_spectral_sharding.spec,
            check_vma=False,
        )(stacked)

    def _streamwise(self, *rows: Array) -> Array:
        """Evaluate the streamwise direction: half spectrum -> real padded field.

        Where the layout turns over when distributed. The `all_to_all` trades
        this device's share of the wavenumbers for its share of the quadrature
        points, after which the wavenumber axis is complete and the inverse
        transform along it is an ordinary local one. The result is split along
        the wall-normal direction instead, which is what the pointwise products
        downstream want and what `_forward` expects back.
        """
        xback = partial(self.F.backward, N=self.pad[0])
        local = jax.vmap(jax.vmap(xback, in_axes=1, out_axes=1))
        stacked = jnp.stack(rows)
        if not self.sharded:
            return local(stacked)

        def kernel(c: Array) -> Array:
            # (nf, k_local, Nq) -> (nf, k, Nq_local)
            c = jax.lax.all_to_all(
                c, axis_name="k", split_axis=2, concat_axis=1, tiled=True
            )
            return local(c)

        return shard_map(
            kernel,
            mesh=batched_spectral_sharding.mesh,
            in_specs=(batched_spectral_sharding.spec,),
            out_specs=batched_physical_sharding.spec,
            check_vma=False,
        )(stacked)

    def _forward(self, *fields: Array) -> Array:
        """Transform padded real fields back to orthogonal coefficient arrays.

        The inverse of `_wall_normal` + `_streamwise`, batched the same way and
        for the same reason, and turning the layout back over the same way: the
        streamwise transform first, while its axis is still whole, then one
        `all_to_all` back to split wavenumbers, then the wall-normal transform,
        local again.
        """
        yspace = self.Wo.basespaces[1]
        xforward = jax.vmap(jax.vmap(self.F.forward, in_axes=1, out_axes=1))
        yforward = jax.vmap(yspace.forward)
        nf = len(fields)

        def to_coefficients(half: Array) -> Array:
            nk = half.shape[1]
            return yforward(half.reshape(nf * nk, -1)).reshape(nf, nk, -1)

        stacked = jnp.stack(fields)
        if not self.sharded:
            return to_coefficients(xforward(stacked))

        def kernel(c: Array) -> Array:
            half = xforward(c)  # (nf, Mx, Nq_local) -> (nf, k, Nq_local)
            half = jax.lax.all_to_all(
                half, axis_name="k", split_axis=1, concat_axis=2, tiled=True
            )  # -> (nf, k_local, Nq)
            return to_coefficients(half)

        return shard_map(
            kernel,
            mesh=batched_physical_sharding.mesh,
            in_specs=(batched_physical_sharding.spec,),
            out_specs=batched_spectral_sharding.spec,
            check_vma=False,
        )(stacked)

    def explicit_terms(
        self, u_hat: Array, v_hat: Array, scalars: tuple[Array, ...]
    ) -> tuple[Array, Array, tuple[Array, ...]]:
        """Return the explicit right-hand side of every transient equation.

        Everything is evaluated on the padded mesh and truncated back by the
        forward transforms.

        The fields are mapped to the *orthogonal* basis first. A `Composite`
        transform is a banded stencil followed by the orthogonal Vandermonde, so
        doing the stencils here leaves u, v and the vorticity sharing one
        wall-normal matrix product despite living in three different composite
        spaces -- which is what lets `_wall_normal` batch them.
        """
        cu = self.VD.to_orthogonal(u_hat)
        cv = self.VB.to_orthogonal(v_hat)
        u_c, v_c, vx_c = self._wall_normal(cu, cv, self.ikx[:, None] * cv)
        (uy_c,) = self._wall_normal(cu, ky=1)
        u_p, v_p, om = self._streamwise(u_c, v_c, vx_c - uy_c)
        Hx, Hy = -v_p * om, u_p * om
        hx, hy = self._forward(Hx, Hy)
        NL_v = self.C_hx @ hx + self.C_hy @ hy
        buoyancy = self.buoyancy(scalars)
        if buoyancy is not None:
            NL_v = NL_v + buoyancy
        NL_0 = self.G @ hx.real[0]  # Scalar product y-dir (-w1, hx.real[0])
        if self.f0 is not None:
            NL_0 = NL_0 + jnp.asarray(self.f0)
        return NL_v, NL_0, self.scalar_terms(u_p, v_p, scalars)

    # -- stepping ----------------------------------------------------------

    def _step_impl(
        self,
        state: tuple[Array, ...],
        dt: float,
        N: ScalarPadding = None,
        /,
    ) -> tuple[Array, ...]:
        """Advance one IMEX Runge-Kutta step.

        Each field's stage comes from its own `IMEXRungeKutta.stage`, which takes
        the nonlinear and linear caches as plain arrays -- so the explicit terms
        are computed here, in rotational form. The constraint is solved between
        the implicit solves and the explicit evaluation, so the streamwise velocity
        is never lagged: at every stage it is the exact solution of continuity for
        that stage's v and u0.

        The tableau is globally stiffly accurate (enforced in `__init__`), so the
        last stage is the accepted solution and no final recombination is needed.
        """
        integrators = self.integrators
        m = tuple(g.apply_mass(s) for g, s in zip(integrators, state, strict=True))
        nl: list[list[Array | None]] = [[] for _ in integrators]
        li: list[list[Array | None]] = [[] for _ in integrators]
        stage = tuple(state)

        for i in range(self.tableau.stages):
            stage = self._zero_nyquist(
                tuple(
                    g.stage(i, m[k], dt, nl[k], li[k], g.linear_forcing)
                    for k, g in enumerate(integrators)
                )
            )
            vi, u0i, scalars = stage[0], stage[1], stage[2:]
            ui = self.velocity(vi, u0i)
            NL_v, NL_0, NL_s = self.explicit_terms(ui, vi, scalars)
            for k, (g, value) in enumerate(
                zip(integrators, (NL_v, NL_0) + NL_s, strict=True)
            ):
                nl[k].append(value)
                li[k].append(g.linear_operator @ stage[k])

        return stage

    def _zero_nyquist(self, state: tuple[Array, ...]) -> tuple[Array, ...]:
        """Return `state` with the Nyquist Fourier mode cleared.

        The operators use the raw wavenumbers while `backward_primitive` zeroes
        the Nyquist for odd derivatives, so leaving it populated would make the
        two disagree there. A real field cannot carry a phase on that mode
        either, so nothing is lost -- and with it held at zero the non-negative
        wavenumbers determine the field outright, which is what `_wall_normal`
        and `_forward` rely on.

        Applied at every stage rather than once at the end of the step, so that
        no stage is ever *evaluated* with a mode the two conventions disagree
        about. u0 is 1-D and has no Fourier direction.
        """
        ny = self.nyquist
        return (state[0].at[ny].set(0.0), state[1]) + tuple(
            s.at[ny].set(0.0) for s in state[2:]
        )

    def _setup_impl(self, dt: float) -> None:
        """Factorize every operator before time stepping starts.

        The continuity operator has to be warmed here for the same reason the
        stage operators do: the solver picks its path by inspecting concrete
        matrix values, and inside the jitted step those are tracers.
        """
        for g in self.integrators:
            g.setup(dt)
        self.A_pin.solve(jnp.zeros(self.VD.num_dofs, dtype=complex))

    def initial_coefficients(self, initial=None) -> tuple[Array, ...]:
        """Return the state at rest, plus whatever the subclass contributes."""
        if initial is not None:
            return self._coerce_state(initial)
        return (
            jnp.zeros(self.VB.num_dofs, dtype=complex),
            jnp.zeros(self.D1.num_dofs),
        ) + self.scalar_initial()

    def _coerce_state(self, state0) -> tuple[Array, ...]:
        """Coerce a restart state into one coefficient array per field."""
        v_hat, u0, *scalars = state0
        return (
            jnp.asarray(v_hat).reshape(self.VB.num_dofs).astype(complex),
            jnp.asarray(u0).reshape(self.D1.num_dofs).real,
        ) + tuple(
            jnp.asarray(s).reshape(g.trialspace.num_dofs).astype(complex)
            for g, s in zip(self.scalar_integrators, scalars, strict=True)
        )

    # -- diagnostics -------------------------------------------------------

    def average(self, f: Array) -> Array:
        """Return the exact volume average of a padded physical field."""
        return self.Wo.scalar_product(f)[0, 0].real / self.vol

    def courant(self, state: tuple[Array, ...], dt: float) -> float:
        """Return the advective Courant number on the padded mesh.

        Only the diffusive terms are implicit, so the step size is limited by
        advection alone. This is the finite-difference form, dt*(|u|/dx +
        |v|/dy), and neither term is the spectral criterion:

        Along Fourier the operator is i*k*u, so what binds is dt*|u|*k_max,
        larger than dt*|u|/dx by k_max*dx = 2*pi/3 under the 3/2 padding. ARS443's
        explicit half is stable to about 2 on the imaginary axis, so the two
        factors very nearly cancel and this number reads ~1 at the boundary
        (0.76 runs, 1.15 diverges). Drop the padding and dx becomes Lx/M, the
        factor becomes pi, and the threshold moves to ~0.6: it is calibrated to
        this configuration, not derived.

        Along the wall-normal direction 1/dy understates the stiffness rather
        than overstating it -- the derivative matrix has norm 3.1x (Legendre) to
        4.6x (Chebyshev) the reciprocal of the smallest spacing, the point
        clustering not being what sets it. What keeps that direction from
        binding is that |v| is small where the points are dense, which is a
        property of the flow and not of the discretization.
        """
        u_p, v_p = self.velocity_from_state(
            state, pad=self.pad, kind=VelocityKind.PHYSICAL
        )

        xm, ym = self.VD.mesh(N=self.pad, broadcast=False)
        dx = float(self.Lx) / xm.shape[0]
        dy = jnp.abs(jnp.asarray(jnp.gradient(ym)))[None, :]
        return float(dt * (jnp.abs(u_p) / dx + jnp.abs(v_p) / dy).max())

    def diagnostics(self, state: tuple[Array, ...]) -> dict[str, float]:
        """Return the structural checks, plus any the subclass adds."""
        u0 = state[1]
        pad = self.pad
        u_hat, v_hat, u_p, v_p = self.velocity_from_state(
            state, pad=pad, kind=VelocityKind.BOTH
        )
        div = self.VD.backward_primitive(
            u_hat, k=(1, 0), N=pad
        ) + self.VB.backward_primitive(v_hat, k=(0, 1), N=pad)
        scale = max(float(jnp.abs(u_p).max()), float(jnp.abs(v_p).max()), 1e-300)
        return {
            "div": float(jnp.abs(div).max()) / scale,
            "v[k=0]": float(jnp.abs(v_hat[0]).max()),
            "u[k=0]-u0": float(jnp.abs(u_hat[0].real - u0).max()),
            "max|u|": float(jnp.abs(u_p).max()),
            "max|v|": float(jnp.abs(v_p).max()),
            # Physical, not the coefficient max: for Poiseuille this must read 1.
            "max|u0|": float(jnp.abs(self.D1.backward(u0)).max()),
        } | self.extra_diagnostics(state)
