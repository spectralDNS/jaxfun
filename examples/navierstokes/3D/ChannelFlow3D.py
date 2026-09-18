# Incompressible Navier-Stokes in a periodic channel, three-dimensional
#
#   u_t + Grad(u)⋅u = -Grad(p) + nu*Div(Grad(u)) + f
#   Div(u)          = 0
#
# on (x, y, z) in [0, Lx] x [0, Ly] x (-1, 1), periodic in x and y, no-slip walls,
# with a constant horizontal body force f standing in for a driving pressure
# gradient. Fourier along axes 0 and 1 and the polynomial (wall-normal) direction
# along axis 2, which is jaxfun's convention and the order the multi-device
# sharding path requires. The velocity is (u, v, w): u streamwise, v spanwise,
# w wall-normal.
#
# The class is named KMM3D after the method rather than after the equations,
# which is what distinguishes it: eliminating the pressure into a fourth-order
# equation for the wall-normal velocity, transporting the wall-normal vorticity
# alongside it, carrying the (kx, ky) = (0, 0) mean flow as its own
# one-dimensional problem, and recovering the two horizontal velocities from
# continuity. That formulation is
#
#   Kim J, Moin P, Moser R. Turbulence statistics in fully developed channel
#   flow at low Reynolds number. Journal of Fluid Mechanics. 1987;177:133-166.
#   doi:10.1017/S0022112087000892
#
# This is the three-dimensional counterpart of ChannelFlow2D.py, whose header
# says what changes here: in two dimensions the pressure elimination closes on
# the wall-normal velocity alone, where in three the horizontal plane has a
# second degree of freedom and the elimination needs an equation for the
# wall-normal vorticity to close. Everything else -- the pin, the mean-flow
# split, the basis pairing, the tableau requirement -- carries over, and where a
# 2-D claim needed re-deriving rather than re-indexing this header says so.
#
# PRESSURE ELIMINATION
#
# Write the momentum equation in rotational form,
#
#   u_t + omega x u = -Grad(P) + nu*Div(Grad(u)),   P = p + |u|^2/2
#
# with omega = Curl(u); the gradient part of Grad(u)⋅u is absorbed into P, which
# is eliminated below and never computed. Set H = omega x u. Taking the
# divergence gives Div(Grad(P)) = -Div(H), and applying Div(Grad(.)) to the
# z-momentum equation and substituting removes the pressure:
#
#   (Div(Grad(w)))_t = nu*Div(Grad(Div(Grad(w)))) + H_x,xz + H_y,yz - H_z,xx - H_z,yy
#
# with w = w_z = 0 at the walls -- a fourth-order equation whose mass operator is
# the Laplacian. Dropping y reduces the bracket to H_x,xz - H_z,xx, which is
# ChannelFlow2D.py's H_x,xy - H_y,xx with y renamed: C_hx and C_hy there are the
# same two operators as C_hx and C_hz here. A subclass may add further explicit
# forcing to this equation -- buoyancy, in a Rayleigh-Benard extension.
#
# THE SECOND EQUATION
#
# One scalar equation cannot carry a three-component solenoidal field. Taking the
# curl of the momentum equation and keeping the wall-normal component gives the
# transport equation for g = omega_z = v_x - u_y:
#
#   g_t = nu*Div(Grad(g)) + H_x,y - H_y,x
#
# with g = 0 at the walls, which follows from no-slip: u and v vanish along the
# whole wall, so their horizontal derivatives do too. This is the equation the 2-D
# solver does not need, because there omega_z is the only vorticity component and
# is determined by v alone.
#
# THE UNKNOWNS
#
#   w    wall-normal velocity    VB = Fx x Fy x B, biharmonic  transient, mass = Lap
#   g    wall-normal vorticity   VD = Fx x Fy x D              transient, mass = mass
#   u, v horizontal velocities   VD                            algebraic, recovered
#   u0   mean streamwise, (0,0)  1-D D1                        transient
#   v0   mean spanwise, (0,0)    1-D D1                        transient
#
# plus one transported scalar per subclass extension.
#
# THE 2x2 RECOVERY
#
# Continuity and the definition of g are two equations in the two horizontal
# velocities. Differentiating continuity by x and the definition of g by y and
# subtracting eliminates v; the mirror combination eliminates u:
#
#   -(u_xx + u_yy) = w_zx + g_y
#   -(v_xx + v_yy) = w_zy - g_x
#
# Both are the *same* operator, applied to different right-hand sides. In
# Fourier space its multiplier is kx^2 + ky^2, the determinant of the 2x2
# system, which vanishes only at (kx, ky) = (0, 0).
#
# The recovery is pointwise exact, not merely a Galerkin projection, by the same
# argument the 2-D solver makes for continuity: w in VB makes w_z vanish at both
# walls, so w_z lies in VD exactly, and g lives in VD already. Both right-hand
# sides are therefore represented without truncation and the weak equation has
# the pointwise solution as its unique Galerkin solution. Measured on a
# manufactured mode at 16 x 16 x 24: residual 8.9e-16, divergence 1.3e-15, and
# v_x - u_y - g at 6.7e-16. That last one is why `explicit_terms` reads omega_z
# off g rather than recomputing it from the recovered velocities -- the two are
# the same array to round-off, and one of them is already in hand.
#
# THE RECOVERY IS DIAGONAL
#
# Neither row needs a banded solve at all, and seeing why is worth the paragraph.
#
# Every operator involved is separable as (horizontal diagonal) x (wall-normal
# matrix). The recovery operator is diag(kx^2) x I x M_z plus I x diag(ky^2) x
# M_z, so it factors as W x M_z with W(kx, ky) = kx^2 + ky^2 -- neither term
# differentiates the unknown in z, so both carry a plain mass matrix there. The
# g terms on the right-hand side carry the same M_z for the same reason. The one
# operator that does not is the w term, whose wall-normal factor is <B', D>.
#
# So project that one out first. With f = w_z taken in VD -- one mass solve,
# f_hat = M_z^-1 <B', D> w_hat -- the equations read
#
#   -(u_xx + u_yy) = f_x + g_y      -(v_xx + v_yy) = f_y - g_x
#
# in which *every* wall-normal factor is now M_z, on both sides. It cancels
# outright, leaving
#
#   W u_hat = cx f_hat + cy g_hat   W v_hat = cy f_hat - cx g_hat
#
# with cx, cy and W plain (kx, ky) arrays: two elementwise multiplies and a
# divide per row, and no wall-normal work whatever.
#
# The projection is not extra work bought back elsewhere -- it is work that was
# being done twice. Writing the old form factored, u_hat = (W x M_z)^-1
# (cx x <B',D>) w_hat + ...), the M_z^-1 applied to the w term *is* this
# projection, and it was being recomputed inside each of the two solves while
# the g terms, which never needed it, were dragged through it as well. Hoisting
# it leaves one wall-normal solve per stage where there were two, on one array
# rather than two.
#
# What that remaining projection costs depends on the wall-normal basis, and by
# more than a constant. In Legendre it is not a solve at all: M_z^-1 <B', D>
# collapses to a single subdiagonal, because differentiating that basis stays
# inside it, so the projection is a shift and a multiply. Every other basis
# differentiates out of its own family and the matrix is dense; those take the
# same per-wavenumber banded solver the stage operators use -- O(n_z) per
# wavenumber, where multiplying by the assembled matrix would be O(n_z^2) and
# would have to store it.
#
# THE (0,0) MODE
#
# W vanishes at (kx, ky) = (0, 0) -- it is the determinant of the 2x2 system,
# and that mode is exactly the one continuity says nothing about, which is why
# it gets its own pair of equations (below). So W is pinned to 1 there to keep
# the divide finite, and the mean profiles are written straight into
# u_hat[0,0] and v_hat[0,0] afterwards. Nothing else is touched.
#
# The 2-D solver has to work harder for this. Its continuity operator is a
# single banded solve, so the mean flow can only be injected *through* it, which
# takes a pinned row in the matrix (and a multiplication by 1j, that operator's
# Fourier diagonal being purely imaginary where these are real). Here the
# horizontal operator is a number per wavenumber, so the (0,0) mode is simply an
# array slot and the pin costs an `.at[0, 0].set`.
#
# THREE PROPERTIES THAT COME OUT FOR FREE
#
# w_hat[0,0] stays exactly zero: every term on the w-equation right-hand side
# carries at least one d/dx or d/dy, and both are diagonal multiplies that are
# exactly 0 there.
#
# g_hat[0,0] stays exactly zero for the same reason, which is also the physics --
# the horizontal mean of v_x - u_y is the mean of a horizontal derivative.
#
# So the entire mean flow lives in u0 and v0, and `diagnostics` reports both
# invariants: a run in which either drifts off zero has a bug, not a small error.
#
# THE MEAN FLOW IS TWO ODEs
#
# Continuity determines u and v for every (kx, ky) except (0, 0), where the 2x2
# system degenerates. There both horizontal momentum equations lose their
# pressure gradient outright (dp/dx -> i*kx*p_hat = 0, and likewise in y) and
# close the system on their own:
#
#   u0_t = nu*u0_zz - <H_x> + f_x
#   v0_t = nu*v0_zz - <H_y> + f_y
#
# with <.> the horizontal average, which is the (0,0) Fourier coefficient. u0 and
# v0 carry the entire mean profile: the Poiseuille base flow of a driven channel,
# and in a turbulent run the mean profile driven by the Reynolds stress d<u w>/dz.
#
# `body_force` is therefore a 2-vector, not the scalar it is in 2-D. A bare float
# is read as streamwise-only, which is the usual case; the rotated
# Orr-Sommerfeld verification needs both components, since it tilts the base flow
# in the horizontal plane and its driving force has to tilt with it.
#
# WHY THIS IS A TAILORED SOLVER
#
# Unchanged from 2-D, and for the same reason: the u and v equations are
# algebraic for (kx, ky) != (0, 0) and transient at (0, 0), while
# SystemIntegrator classifies a whole equation as one or the other. No symbolic
# weak form can express the split, because a Fourier multiplier that is 1 at
# (0, 0) and 0 elsewhere is not a differential operator. So the stage loop is
# written out here -- but composed from framework pieces rather than hand-rolled:
# TimeStepper supplies the batched step driver, and one IMEXRungeKutta per
# transient equation supplies the mass/stiffness split, the stage operators
# (cached per distinct Butcher diagonal and factorized outside jit) and the
# Butcher accumulation. Their weak forms carry no nonlinear terms at all; every
# explicit term is computed here and handed to `stage()`, which takes its
# nonlinear and linear caches as plain arrays. There are four such equations here
# rather than two.
#
# DEALIASING: 3/2 IN BOTH FOURIER DIRECTIONS, NONE IN THE WALL-NORMAL
#
# The matrices are assembled exactly from the precomputed composite stencils, so
# quadrature error can only enter through the transform pair around the pointwise
# products. Every direction aliases there. A quadratic product of two fields that
# fill the grid has twice the bandwidth, and neither the FFT nor the Gauss
# quadrature can carry the excess; it folds back onto the retained modes. The
# fold measured in 2-D at N=32, unchanged in kind here:
#
#   Fourier     exact fold at amplitude 1: mode k1+k2 lands on k1+k2-M
#   Chebyshev   exact fold at amplitude 1: T_{2N-j} is -T_j on the Gauss points
#   Legendre    no exact fold: 0.97 at m=N+1, falling to 0.1-0.2 and spread over
#               several modes by m=2N
#
# So this is not a Fourier-only problem, and the wall-normal direction is not
# spared by using a Gauss quadrature rather than an FFT. Nor is the 3/2 rule
# mandatory in either periodic direction: in every direction the size of the
# error is set by how much of the product's spectrum actually reaches the fold,
# so a run with nothing left at the top of a spectrum aliases in that direction
# whether padded or not.
#
# The asymmetric setting here is inherited from 2-D, where it rests on
# measurement: at 128 x 64 under Legendre on a developed Rayleigh-Benard field,
# dropping the wall-normal padding bought about 23% of the runtime and cost 5e-11
# in the nonlinear terms once the run was resolved, rising to 1e-2 at the margin
# of resolution. Those numbers are two-dimensional, Legendre, and that
# configuration; they have NOT been remeasured here, and the third direction
# changes the cost side of the trade in particular. Either way the guard is the
# fraction of the spectrum left in the top third: below ~1e-5 padding is
# pointless in every direction, at 1e-3 it is not; past that, raise the
# resolution rather than trusting the answer.
#
# THE FIELDS ARE REAL, SO HALF OF ONE SPECTRUM IS REDUNDANT
#
# Every physical field here is real, so its spectrum is Hermitian: the mode at
# (-kx, -ky) is the conjugate of the one at (kx, ky) and carries no information.
# The x direction is therefore built with `TensorProduct(..., real=True)`, which
# stores only kx = 0, ..., M/2 and transforms with rfft/irfft. Nothing is
# approximated -- the equations for the reflected modes are the conjugates of
# those kept -- but everything downstream runs on half the data.
#
# Only *one* axis can be halved, and it has to be axis 0: the symmetry is a joint
# reflection of all axes, so halving one already uses it up, and the forward
# transform has to reach the r2c axis while the array is still real.
# `_validate_hermitian_axis` in tensorproductspace.py enforces both. So the
# spanwise direction keeps its full complex spectrum even though its fields are
# just as real.
#
# THE NYQUIST MODE ON BOTH FOURIER AXES
#
# Both are held at zero at every stage, but they are not the same thing and only
# one of them is free.
#
# On the halved x axis, nothing is lost: a real field cannot carry a phase at the
# Nyquist wavenumber, d/dx of it is not representable, and the operator matrices
# use the raw wavenumber while the transforms zero it -- so the two conventions
# disagree unless it vanishes, and holding it at zero is bookkeeping.
#
# On the full spanwise axis the stored wavenumber at index My/2 is -My/2, a mode a
# real field *can* carry, and zeroing it is a genuine one-mode truncation. It is
# done for the same consistency reason -- `wavenumbers(eliminate_highest_freq=
# True)` zeroes it for odd derivatives while the operators use the raw value --
# and it is what every spectral channel code does with that mode. But it is a
# choice, not an identity, so it is stated here rather than hidden.
#
# CHOICE OF BASIS AND TEST SPACE
#
# `polynomial` picks the wall-normal basis and `kind` picks how it is tested. The
# two are not independent: what matters is whether the resulting operators stay
# banded, because every implicit solve here is a banded LU, and its cost grows
# sharply with the bandwidth -- the forward/backward substitution runs as a
# `lax.scan` whose carry widens with the band. Widest band in the w equation,
# measured here at 16 x 16 x 24 and matching the 2-D table exactly (the extra
# Fourier axis is diagonal and changes no bandwidth):
#
#                            N=24   N=48   N=96
#   Legendre   Galerkin         5      5      5     <- the default
#   Legendre   Petrov-Galerkin  7      7      7
#   Chebyshev  Galerkin        10     24     87
#   Chebyshev  Petrov-Galerkin  7      7      7     <- the fast pairing
#   ChebyshevU Galerkin        10     22     74
#
# So pair LEGENDRE with GALERKIN and CHEBYSHEV with PETROV_GALERKIN. The other
# two cells of that square are legal and give the same answer, but Legendre's
# Galerkin operators are already banded and PG only widens them, while Chebyshev
# under plain Galerkin grows like N -- differentiating a Chebyshev expansion in
# coefficient space is dense upper triangular, where Legendre's weight of 1 lets
# integration by parts collapse the same operators to a few diagonals.
# ChebyshevU has no PG test space implemented and is Galerkin only.
#
# Under PG there are *two* Petrov-Galerkin test spaces to build, not one. The w
# equation needs one for the biharmonic space, as in 2-D. The g equation needs
# its own, because it too carries a wall-normal second derivative: tested against
# the Galerkin space its Chebyshev diffusion operator goes dense exactly as the
# w equation's would. The recovery, by contrast, is tested Galerkin in both
# cases -- it carries no wall-normal derivative of its unknown at all, only a
# mass matrix, so there is nothing for PG to sparsify.
#
# Accuracy is the more portable reason to prefer Chebyshev: its transform
# round-trips at machine epsilon where the dense Legendre Vandermonde accumulates
# a few hundred ulp. Which is faster is machine-dependent -- measure it rather
# than trusting a number written here, and read the three traps in
# ChannelFlow2D.py's header before doing so, especially that a diverged run is
# *faster* than a healthy one.
#
# HOW THE NONLINEAR TERM IS EVALUATED
#
# H = omega x u needs all three velocity components and all three vorticity
# components in physical space: six fields, from four coefficient arrays. Two of
# the vorticity components need a wall-normal derivative and two need a
# horizontal one:
#
#   omega_x = w_y - v_z      omega_y = u_z - w_x      omega_z = g
#
# The transform is split into its two directions so that those differences can be
# assembled in between. w_y and v_z need different wall-normal matrices, but they
# are only ever used as a difference, so subtracting them while x and y are still
# in coefficient space turns eight padded horizontal transforms into six. The
# wall-normal direction still runs eight times, but it is the cheap half: it is
# unpadded, and it batches into one matrix product.
#
# That is the only reason these are not
#
#   fields = self.Wo.backward_batch(stack6, N=self.pad)
#   uz, vz = self.Wo.backward_primitive_batch(stack2, k=(0, 0, 1), N=self.pad)
#
# which is otherwise the same work in two lines, and which is what a first
# version should use if these three helpers ever get in the way. The 2-D solver
# measured its own version of this saving at 3-10% of a step for one transform in
# four; two in eight of a larger transform should be worth more, but that has NOT
# been measured here.
#
# HOW THE THREE ARE DISTRIBUTED
#
# Each carries its own sharding, and they compose into the two-phase shape a
# separable transform wants -- the wall-normal direction evaluated while the
# streamwise wavenumbers are split, the horizontal directions while the spanwise
# *quadrature points* are split instead, and one `all_to_all` to swap which is
# which:
#
#   _wall_normal   kx-sharded -> kx-sharded    no communication
#   _horizontal    kx-sharded -> y-sharded     one all_to_all
#   _forward       y-sharded  -> kx-sharded    one all_to_all
#
# The pointwise products in between are elementwise on y-sharded arrays, so they
# need nothing, which is also what lets `scalar_terms` be handed physical fields
# and stay unaware of the distinction.
#
# Two things about this are worth knowing, because both look like they should
# have needed work and did not. The batched shardings are the same two objects
# the 2-D solver uses: a PartitionSpec shorter than the array's rank leaves the
# trailing axes replicated, so `batched_spectral_sharding` splits array axis 1
# and `batched_physical_sharding` array axis 2 at rank 4 just as at rank 3 --
# which is space axis 1 in each layout, the same axis in both dimensions. And for
# the same reason the `split_axis`/`concat_axis` pairs below are the 2-D ones
# unchanged. The spanwise transform goes before the all_to_all rather than after
# precisely so that this stays true.
#
# What does change is which extents have to divide the device count: the stored
# streamwise wavenumber count on the way out, and the *padded spanwise*
# quadrature count on the way in. The first is `RFourier`'s job and always holds;
# the second is this solver's, and a padding that fails it keeps the local path --
# losing parallelism, not correctness. `_use_spmd` in tensorproductspace.py
# applies the same two conditions to the library's own 3-D transforms.
#
# Letting the compiler partition these automatically does not work, for the
# reason the 2-D header gives: `_horizontal` and `_forward` transform *along* the
# split axis, and GSPMD resolves that by gathering the whole axis onto every
# device at every stage, which is not a distribution of the work at all.
#
# This is also the first place in jaxfun where `TPMatricesWavenumberSolver` is
# handed two Fourier axes against one polynomial axis. It was written for that
# case -- it detects Fourier axes as "purely diagonal on that axis for every
# term" and builds its per-wavenumber weight as an outer product over all of them
# -- but nothing had exercised it before, sharded or otherwise.
#
# RESOLUTION AND STEP SIZE ARE COUPLED
#
# Only the diffusive terms are implicit, so dt is limited by the advective
# Courant number and has to come down roughly in step with the resolution -- and
# it is the *wall-normal* resolution that binds, the Gauss node spacing
# collapsing like 1/N^2 at the walls for every basis offered here. The `courant`
# diagnostic reports where a run stands; read its docstring before trusting the
# number, because the 2-D calibration does not transfer.
#
# CHOICE OF TABLEAU
#
# Any globally stiffly accurate IMEX tableau works -- ARS443, ARS222 and
# IMEX_EULER; the Kennedy-Carpenter ARK family is only implicitly stiffly
# accurate and is rejected. But advection is handled by the *explicit* table, so
# the explicit part's imaginary-axis stability decides whether a scheme is usable
# at all. Measured in 2-D on the Orr-Sommerfeld growth rate (Re=8000, T=50), as
# relative error against linear theory:
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
# Two demos drive this solver. Neither is turbulent channel flow, which is the
# eventual application and far too heavy to verify against.
#
#   SquireMode3D.py     Two checks. First, one shot with no time stepping: a
#                       manufactured divergence-free mode, with H differentiated
#                       symbolically, against every assembled forcing operator --
#                       which is what catches a wrong sign or a swapped axis in
#                       omega, H, or the five right-hand sides. Then an exact
#                       decaying Squire mode, which exercises the g equation, the
#                       g branch of the recovery and both FFT axes at a size that
#                       runs in seconds. It does not exercise the biharmonic w
#                       equation or advection.
#
#   OrrSommerfeld3D.py  An Orr-Sommerfeld eigenmode on plane Poiseuille flow,
#                       twice. Embedded with no spanwise dependence it must
#                       reproduce the 2-D solver's growth rate, that slice of this
#                       solver being the 2-D solver. Rotated 45 degrees, with the
#                       base flow and the driving force tilted with it, it has the
#                       same eigenvalue but loads both Fourier axes and the full
#                       2x2 recovery.
#
# One mechanism neither covers: a rotated two-dimensional mode has omega_z
# identically zero, a plane flow having no wall-normal vorticity, so nothing here
# tests the Squire coupling -i*beta*U'*w by which w drives g. Doing that needs a
# genuinely oblique eigenmode, which means solving the coupled Orr-Sommerfeld /
# Squire eigenproblem rather than reusing OrrSommerfeld_eigs.py as both demos do.
#
# Spatial discretization: Fourier x Fourier x (Legendre Galerkin | Chebyshev PG)
# Time discretization: any globally stiffly accurate IMEX Runge-Kutta tableau
# ruff: noqa: E402
from enum import StrEnum
from functools import partial
from typing import Any, Literal, cast, overload

import jax

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
from jaxfun.la.tpmatrix import (
    TPMatrices,
    TPMatricesWavenumberSolver,
    tpmats_wavenumber_factor,
)
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


def along(fn: Any, ax: int, dim: int = 4) -> Any:
    """Return `fn`, a 1-D transform, applied along axis `ax` of a rank-`dim` array.

    One `vmap` per other axis, wrapped smallest-index innermost so that each
    level removes an index larger than any the levels below it refer to, which
    is what keeps their `in_axes` valid without renumbering.

    `jaxfun.sharding._build_local_apply_fn` is the same thing with a `jax.jit`
    around it, which is right where the library calls it -- from outside any
    trace, so the compiled binary is reused. Here every caller is already inside
    the step's own jit, where that wrapper only adds a nested trace per stage,
    so this is the un-jitted form. The 2-D solver spells its equivalent out
    inline for the same reason.
    """
    out = fn
    for other in sorted(set(range(dim)) - {ax}):
        out = jax.vmap(out, in_axes=other, out_axes=other)
    return out


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


class KMM3D(TimeStepper[tuple[Array, ...]]):
    """Kim-Moin-Moser channel flow in three dimensions.

    Incompressible Navier-Stokes in a doubly periodic channel, pressure
    eliminated into a fourth-order equation for the wall-normal velocity and
    closed by a transport equation for the wall-normal vorticity, after Kim,
    Moin & Moser (JFM 177:133-166, 1987).

    The state is `(w_hat, g_hat, u0, v0, *scalars)`: the wall-normal velocity,
    the wall-normal vorticity, the two mean horizontal profiles, and one array
    per transported scalar contributed by a subclass. The horizontal velocities
    are never stored -- they are recomputed from w, g and the mean flow at every
    stage by `velocity`, so they cannot drift out of sync with them.

    Subclasses extend the system through four hooks: `scalar_integrators`,
    `scalar_initial`, `scalar_terms` and `buoyancy`. Everything else -- the
    spaces, the velocity equations, the stage loop -- is shared.
    """

    def __init__(
        self,
        M: int,
        My: int,
        N: int,
        Lx: float,
        Ly: float,
        nu: float,
        *,
        body_force: float | tuple[float, float] = 0.0,
        tableau: IMEXTableau = ARS443,
        time: tuple[float, float] | None = None,
        padding: tuple[int, int, int] | None = None,
        polynomial: PolynomialKind = PolynomialKind.LEGENDRE,
        kind: TestSpaceKind = TestSpaceKind.GALERKIN,
    ) -> None:
        """Assemble the velocity spaces, operators and sub-integrators.

        Args:
            M: Number of Fourier modes along the streamwise direction.
            My: Number of Fourier modes along the spanwise direction.
            N: Number of modes along the wall-normal direction.
            Lx: Streamwise period.
            Ly: Spanwise period.
            nu: Kinematic viscosity.
            body_force: Constant horizontal forcing, standing in for a driving
                pressure gradient. A bare float is streamwise-only. Being
                constant it has only a (0,0) Fourier component, so it enters the
                two mean-flow equations alone.
            tableau: Any *globally* stiffly accurate IMEX Runge-Kutta tableau,
                so that the last stage is the accepted solution.
            time: Optional default integration interval.
            padding: Shape of real space. Only required if padding is used,
                otherwise real shape defaults to M, My, N.
            polynomial: Polynomial basis for the wall-normal direction, one of
                the keys of `POLYNOMIALS`, by member name or short form. See
                "CHOICE OF BASIS AND TEST SPACE" in the header: pair LEGENDRE
                with GALERKIN and CHEBYSHEV with PETROV_GALERKIN.
            kind: Test space kind, either GALERKIN or PETROV_GALERKIN. Short
                forms G or PG. CHEBYSHEVU has no PG test space yet and raises
                NotImplementedError if asked for one.
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
        self.nu = nnx.static(nu)
        self.Lx, self.Ly = nnx.static(Lx), nnx.static(Ly)
        # One per Fourier axis. On the halved streamwise axis this index is the
        # last stored wavenumber; on the full spanwise axis it is the mode whose
        # stored wavenumber is -My/2. See "THE NYQUIST MODE" in the header.
        self.nyquist = nnx.static((M // 2, My // 2))
        self.pad = nnx.static((M, My, N) if padding is None else padding)

        hom = {"left": {"D": 0}, "right": {"D": 0}}
        bih = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
        Fx = FunctionSpace(M, Fourier.Fourier, domain=Domain(0, Lx), name="Fx")
        Fy = FunctionSpace(My, Fourier.Fourier, domain=Domain(0, Ly), name="Fy")
        D = FunctionSpace(N, polspace, bcs=hom, name="D")
        B = FunctionSpace(N, polspace, bcs=bih, name="B")
        VD = TensorProduct(Fx, Fy, D, name="VD", real=True)
        VB = TensorProduct(Fx, Fy, B, name="VB", real=True)
        # `real=True` substituted the half spectrum on axis 0, which stores
        # M/2 + 1 wavenumbers rather than M, plus whatever padding the device
        # count needs to split them. Axis 1 keeps its full complex spectrum.
        Fx = cast(Fourier.RFourier, VD.basespaces[0])
        Fy = cast(Fourier.Fourier, VD.basespaces[1])

        # Whether the transforms below take their distributed path. Decided once,
        # from sizes rather than from any array's placement, so the single-device
        # path is chosen at construction and the code that runs there is
        # unchanged.
        n_dev = len(jax.devices())
        self.sharded = nnx.static(
            n_dev > 1 and VD.num_dofs[0] % n_dev == 0 and self.pad[1] % n_dev == 0
        )

        # Convection H and the scalar fluxes satisfy no boundary conditions, so
        # they live in the orthogonal space.
        Wo = VD.get_orthogonal()
        D1 = VD.basespaces[2]

        if PG:
            PB = TensorProduct(Fx, Fy, B.get_testspace("PG", name="BP"), name="PB")
            PD = TensorProduct(Fx, Fy, D.get_testspace("PG", name="DP"), name="PD")
            P1 = cast(PGComposite, D1).get_testspace("PG", name="P1")
        else:
            PB, PD, P1 = VB, VD, D1

        self.Fx, self.Fy = nnx.static(Fx), nnx.static(Fy)
        self.ikx = nnx.data(
            1j * Fx.wavenumbers(eliminate_highest_freq=True) * float(Fx.domain_factor)
        )
        self.iky = nnx.data(
            1j * Fy.wavenumbers(eliminate_highest_freq=True) * float(Fy.domain_factor)
        )
        self.system = nnx.static(VD.system)
        self.VD, self.VB = nnx.static(VD), nnx.static(VB)
        self.Wo, self.D1 = nnx.static(Wo), nnx.static(D1)
        self.PB, self.PD, self.P1 = (
            nnx.static(PB),
            nnx.static(PD),
            nnx.static(P1),
        )
        self.polspace = nnx.static(polspace)
        self.testkind = nnx.static(kind)

        x, y, z = VD.system.base_scalars()
        t = VD.system.base_time()
        nu_c = Constant("nu", nu)

        u = TrialFunction(VD, name="u")
        wt = TestFunction(VD, name="wt")
        W = TrialFunction(VB, name="W", transient=True)
        q = TestFunction(PB, name="q")
        G = TrialFunction(VD, name="g", transient=True)
        s = TestFunction(PD, name="s")
        h = TrialFunction(Wo, name="h")
        u1 = TrialFunction(D1, name="u1", transient=True)
        w1 = TestFunction(P1, name="w1")

        # Purely linear weak forms: every explicit term is supplied by `step`.
        # This is necessary for highly optimized solvers that perform several
        # tricks for efficiency.
        eq_w = ((Div(Grad(W))).diff(t) - nu_c * Div(Grad(Div(Grad(W))))) * q
        eq_g = (G.diff(t) - nu_c * Div(Grad(G))) * s
        eq_0 = (u1.diff(t) - nu_c * u1.diff(z, 2)) * w1
        opts: dict[str, Any] = {**ASSEMBLE, "solver_options": SOLVE, "tableau": tableau}
        self.gw = nnx.data(
            IMEXRungeKutta(eq_w, initial=jnp.zeros(VB.num_dofs, dtype=complex), **opts)
        )
        self.gg = nnx.data(
            IMEXRungeKutta(eq_g, initial=jnp.zeros(VD.num_dofs, dtype=complex), **opts)
        )
        # One integrator per mean profile. The weak form is the same; they are
        # separate objects because each caches its own stage operators, and they
        # are one-dimensional so that costs nothing.
        self.g0u = nnx.data(
            IMEXRungeKutta(eq_0, initial=jnp.zeros(D1.num_dofs), **opts)
        )
        self.g0v = nnx.data(
            IMEXRungeKutta(eq_0, initial=jnp.zeros(D1.num_dofs), **opts)
        )

        # -- the recovery, reduced to one mass solve and diagonal work -----

        A_h = linear_operator(-(u.diff(x, 2) + u.diff(y, 2)) * wt)
        assert isinstance(A_h, TPMatrices), (
            "the horizontal Laplacian should assemble as one TPMatrix per "
            f"direction, got {type(A_h).__name__}"
        )
        terms = list(A_h.tpmats)
        M_z = cast(DiaMatrix, terms[0].mats[2])

        def horizontal_weight(tp: TPMatrix) -> Array:
            """The (kx, ky) multiplier of a term that is diagonal in both."""
            return jnp.asarray(tp.coefficient) * jnp.outer(
                tp.mats[0].diagonal(0), tp.mats[1].diagonal(0)
            )

        def separable_weight(tp: TPMatrix, axis: int) -> Array:
            """Broadcast a diagonal Fourier matrix in the other two axes"""
            flat = tp.mats[1 - axis].diagonal(0)
            assert jnp.allclose(flat, flat[0]), (
                f"axis {1 - axis} of this term is not a constant diagonal, so "
                "the weight is not separable and must not be stored as a vector"
            )
            w = jnp.asarray(tp.coefficient) * flat[0] * tp.mats[axis].diagonal(0)
            return w.reshape((-1, 1, 1) if axis == 0 else (1, -1, 1))

        weights = sum(
            (horizontal_weight(tp) for tp in terms[1:]),
            horizontal_weight(terms[0]),
        )
        assert weights[0, 0] == 0.0, (
            "the (0,0) block of the recovery operator must be singular, got "
            f"{weights[0, 0]!r}"
        )
        # `u` is the generic VD trial function: the projected w_z and g both
        # live in VD, so one set of operators serves both.
        C_fx = cast(TPMatrix, linear_operator(u.diff(x, 1) * wt))
        C_fy = cast(TPMatrix, linear_operator(u.diff(y, 1) * wt))
        C_wz = cast(TPMatrix, linear_operator(W.diff(z, 1) * wt))
        M_op = cast(TPMatrix, linear_operator(u * wt))
        for name, op in (("d/dx", C_fx), ("d/dy", C_fy), ("mass", M_op)):
            assert isinstance(op, TPMatrix), f"{name} should be one term"
            assert jnp.array_equal(op.mats[2].data, M_z.data), (
                f"the {name} operator's wall-normal factor differs from the "
                "recovery operator's; the cancellation below assumes they are "
                "the same mass matrix"
            )

        self.cx = nnx.data(separable_weight(C_fx, 0))
        self.cy = nnx.data(separable_weight(C_fy, 1))
        self.weights = nnx.data(weights.at[0, 0].set(1.0)[..., None])

        # The projection of w_z into VD, f_hat = M_z^-1 <B', D> w_hat, is the one
        # wall-normal operation the recovery still needs. For Legendre it is a single
        # subdiagonal and a very fast solve. Chebyshev take the wavenumber solver path.
        C_z = cast(DiaMatrix, C_wz.mats[2])
        n_b, n_d = VB.num_dofs[2], VD.num_dofs[2]
        if polynomial is PolynomialKind.LEGENDRE:
            # Read the subdiagonal off the two banded operators instead of
            # forming M_z^-1 <B', D> and taking a diagonal of it. If P is the
            # subdiagonal p then <B', D>[:, k] = p_k M_z[:, k+1] column by
            # column, so one ratio of stored diagonals gives it -- no dense
            # matrix, and none of the round-off a solve would leave off it.
            self.pz_pad = nnx.static((1, n_d - 1 - n_b))
            i_c, i_m = C_z.offsets.index(-1), M_z.offsets.index(0)
            sub = C_z.data[i_c, :n_b] / M_z.data[i_m, 1 : n_b + 1]
            self.pz = nnx.data(jnp.pad(sub, self.pz_pad))
            self.C_wz, self.M_proj = nnx.data(None), nnx.data(None)

        else:
            self.pz, self.pz_pad = nnx.data(None), nnx.static((0, 0))
            self.C_wz = nnx.data(C_wz)
            self.M_proj = nnx.data(tpmats_wavenumber_factor([M_op]))

        # -- w equation: + H_x,xz + H_y,yz - H_z,xx - H_z,yy ---------------
        self.C_hx = nnx.data(linear_operator(h.diff(x, 1).diff(z, 1) * q))
        self.C_hy = nnx.data(linear_operator(h.diff(y, 1).diff(z, 1) * q))
        self.C_hz = nnx.data(linear_operator(-(h.diff(x, 2) + h.diff(y, 2)) * q))

        # -- g equation: + H_x,y - H_y,x -----------------------------------
        self.C_gfx = nnx.data(linear_operator(h.diff(y, 1) * s))
        self.C_gfy = nnx.data(linear_operator(-h.diff(x, 1) * s))

        # -- mean flow: -<H_x> + f_x and -<H_y> + f_y ----------------------
        h1 = TrialFunction(D1.get_orthogonal(), name="h1")
        self.G_mean = nnx.data(linear_operator(-w1 * h1))

        # A constant force has only a (0,0) component, so it lands entirely on
        # the mean profiles.
        fx, fy = (
            (float(body_force), 0.0)
            if isinstance(body_force, int | float)
            else (float(body_force[0]), float(body_force[1]))
        )
        unit = P1.scalar_product(jnp.ones(P1.shape[0]))
        self.f0u = nnx.data(fx * unit if fx else None)
        self.f0v = nnx.data(fy * unit if fy else None)
        # Volume of the domain in scalar-product units, for exact averages.
        self.vol = nnx.static(
            float(Wo.scalar_product(jnp.ones(Wo.shape))[0, 0, 0].real)
        )

    # -- extension points --------------------------------------------------

    @property
    def scalar_integrators(self) -> tuple[IMEXRungeKutta, ...]:
        """Sub-integrators for transported scalars, appended to the state."""
        return ()

    def scalar_initial(self) -> tuple[Array, ...]:
        """Initial coefficients for each transported scalar."""
        return ()

    def scalar_terms(
        self,
        u_p: Array,
        v_p: Array,
        w_p: Array,
        scalars: tuple[Array, ...],
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
        return (self.gw, self.gg, self.g0u, self.g0v) + self.scalar_integrators

    def project_wz(self, w_hat: Array) -> Array:
        """Return w_z, in VD coefficients, from w in VB.

        Exact rather than a best fit: w in VB vanishes with its slope at both
        walls, so w_z satisfies VD's boundary conditions and is represented
        there without truncation.

        Which of the two forms runs was fixed at construction by the wall-normal
        basis: a single subdiagonal for Legendre, a banded solve for the rest.
        Both touch only the wall-normal axis, which is never the distributed
        one, so neither communicates.
        """
        if self.pz is not None:
            wide = jnp.pad(w_hat, ((0, 0), (0, 0), self.pz_pad))
            return jnp.asarray(self.pz) * wide
        solver = cast(TPMatricesWavenumberSolver, self.M_proj)
        return solver.solve(cast(TPMatrix, self.C_wz) @ w_hat)

    def velocity(
        self, w_hat: Array, g_hat: Array, u0: Array, v0: Array
    ) -> tuple[Array, Array]:
        """Return the horizontal velocities from w, g and the mean profiles.

        The 2x2 system of continuity and the definition of g. Projecting w_z
        into VD once leaves both rows elementwise. See "THE RECOVERY IS
        DIAGONAL" in the header for why the mass matrix cancels, and why doing
        the projection here rather than inside each row removes the second
        wall-normal solve.

        The mean flow is written straight into the (0,0) mode, which the
        horizontal operator says nothing about: its multiplier vanishes there,
        which is exactly why that mode needs its own equation.
        """
        f_hat = self.project_wz(w_hat)
        u_hat = (self.cx * f_hat + self.cy * g_hat) / self.weights
        v_hat = (self.cy * f_hat - self.cx * g_hat) / self.weights
        return (
            u_hat.at[0, 0].set(u0 + 0j),
            v_hat.at[0, 0].set(v0 + 0j),
        )

    @overload
    def velocity_from_state(
        self,
        state: tuple[Array, ...],
        pad: tuple[int, int, int] | None = None,
        kind: Literal[VelocityKind.SPECTRAL, VelocityKind.PHYSICAL] = ...,
    ) -> tuple[Array, Array, Array]: ...
    @overload
    def velocity_from_state(
        self,
        state: tuple[Array, ...],
        pad: tuple[int, int, int] | None = None,
        *,
        kind: Literal[VelocityKind.BOTH],
    ) -> tuple[Array, Array, Array, Array, Array, Array]: ...
    def velocity_from_state(
        self,
        state: tuple[Array, ...],
        pad: tuple[int, int, int] | None = None,
        kind: VelocityKind = VelocityKind.PHYSICAL,
    ) -> tuple[Array, ...]:
        """Return (u, v, w), spectral or physical or both, from a whole state."""
        w_hat, g_hat, u0, v0 = state[0], state[1], state[2], state[3]
        u_hat, v_hat = self.velocity(w_hat, g_hat, u0, v0)
        if kind == VelocityKind.SPECTRAL:
            return u_hat, v_hat, w_hat
        u_p = self.VD.backward(u_hat, N=pad)
        v_p = self.VD.backward(v_hat, N=pad)
        w_p = self.VB.backward(w_hat, N=pad)
        if kind == VelocityKind.PHYSICAL:
            return u_p, v_p, w_p
        return u_hat, v_hat, w_hat, u_p, v_p, w_p

    # -- the three halves of the transform ---------------------------------
    #
    # See "HOW THE NONLINEAR TERM IS EVALUATED" and "HOW THE THREE ARE
    # DISTRIBUTED" in the header for why the transform is split this way and how
    # the two batched shardings and the `all_to_all` axis pairs carry over
    # unchanged from the two-dimensional solver.

    def _wall_normal(self, *coeffs: Array, kz: int = 0) -> Array:
        """Evaluate the wall-normal direction, leaving x and y in coefficients.

        Batched over `coeffs`: every field goes through the same Vandermonde (for
        Legendre) -- `kz` picks which derivative of it -- so the matrix product
        runs once on the stacked fields rather than once each. Fields wanting a
        different `kz` need their own call; that is the only constraint on what
        can share a batch.

        Communicates nothing -- every wavenumber pair's wall-normal transform is
        independent -- but still goes through `shard_map` when distributed. The
        batching is why: folding the fields into the streamwise wavenumber axis
        is a concatenation *along the split axis*, which the compiler would have
        to resolve by redistributing. Inside the kernel the same concatenation is
        local, and the arithmetic is the one big matrix product it is on one
        device.
        """
        Nq = self.pad[2]
        zspace = self.Wo.basespaces[2]
        zback = jax.vmap(partial(zspace.backward_primitive, k=kz, N=Nq))
        nf = len(coeffs)

        def local(c: Array) -> Array:
            nk, nl = c.shape[1], c.shape[2]
            return zback(c.reshape(nf * nk * nl, -1)).reshape(nf, nk, nl, Nq)

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

    def _horizontal(self, *rows: Array) -> Array:
        """Evaluate both periodic directions: coefficients -> real padded field.

        Where the layout turns over when distributed. The spanwise transform runs
        first, while its axis is whole and the streamwise wavenumbers are still
        split; then the `all_to_all` trades this device's share of the streamwise
        wavenumbers for its share of the spanwise quadrature points, after which
        the streamwise axis is complete and the inverse transform along it is an
        ordinary local one. The result is split along the spanwise direction
        instead, which is what the pointwise products downstream want and what
        `_forward` expects back.

        Ordering the two this way is not only about the collective. The
        streamwise axis is the half spectrum, so its inverse transform is the
        r2c one and has to come last -- it is what produces the real array.
        """
        yback = along(partial(self.Fy.backward, N=self.pad[1]), 2)
        xback = along(partial(self.Fx.backward, N=self.pad[0]), 1)
        stacked = jnp.stack(rows)
        if not self.sharded:
            return xback(yback(stacked))

        def kernel(c: Array) -> Array:
            c = yback(c)  # (nf, kx_local, ky, Nqz) -> (nf, kx_local, Ny, Nqz)
            # (nf, kx_local, Ny, Nqz) -> (nf, kx, Ny_local, Nqz)
            c = jax.lax.all_to_all(
                c, axis_name="k", split_axis=2, concat_axis=1, tiled=True
            )
            return xback(c)

        return shard_map(
            kernel,
            mesh=batched_spectral_sharding.mesh,
            in_specs=(batched_spectral_sharding.spec,),
            out_specs=batched_physical_sharding.spec,
            check_vma=False,
        )(stacked)

    def _forward(self, *fields: Array) -> Array:
        """Transform padded real fields back to orthogonal coefficient arrays.

        The inverse of `_wall_normal` + `_horizontal`, batched the same way and
        for the same reason, and turning the layout back over the same way: the
        streamwise transform first, while its axis is still whole *and its input
        is still real*, then one `all_to_all` back to split wavenumbers, then the
        spanwise and wall-normal transforms, local again.
        """
        zspace = self.Wo.basespaces[2]
        xforward = along(self.Fx.forward, 1)
        yforward = along(self.Fy.forward, 2)
        zforward = jax.vmap(zspace.forward)
        nf = len(fields)

        def to_coefficients(half: Array) -> Array:
            nk, nl = half.shape[1], half.shape[2]
            return zforward(half.reshape(nf * nk * nl, -1)).reshape(nf, nk, nl, -1)

        stacked = jnp.stack(fields)
        if not self.sharded:
            return to_coefficients(yforward(xforward(stacked)))

        def kernel(c: Array) -> Array:
            half = xforward(c)  # (nf, Nx, Ny_local, Nqz) -> (nf, kx, Ny_local, Nqz)
            half = jax.lax.all_to_all(
                half, axis_name="k", split_axis=1, concat_axis=2, tiled=True
            )  # -> (nf, kx_local, Ny, Nqz)
            return to_coefficients(yforward(half))

        return shard_map(
            kernel,
            mesh=batched_physical_sharding.mesh,
            in_specs=(batched_physical_sharding.spec,),
            out_specs=batched_spectral_sharding.spec,
            check_vma=False,
        )(stacked)

    def explicit_terms(
        self,
        u_hat: Array,
        v_hat: Array,
        w_hat: Array,
        g_hat: Array,
        scalars: tuple[Array, ...],
    ) -> tuple[Array, Array, Array, Array, tuple[Array, ...]]:
        """Return the explicit right-hand side of every transient equation.

        Everything is evaluated on the padded mesh and truncated back by the
        forward transforms.

        The fields are mapped to the *orthogonal* basis first. A `Composite`
        transform is a banded stencil followed by the orthogonal Vandermonde, so
        doing the stencils here leaves all six fields sharing one wall-normal
        matrix product despite living in three different composite spaces --
        which is what lets `_wall_normal` batch them.

        omega_z is read off `g_hat` rather than recomputed as v_x - u_y: the
        recovery makes the two the same array to round-off (measured 6.7e-16),
        and one of them is already a state variable.
        """
        cw = self.VB.to_orthogonal(w_hat)
        (cu, cv, cg) = jax.vmap(self.VD.to_orthogonal)(jnp.stack((u_hat, v_hat, g_hat)))
        u_c, v_c, w_c, omz_c, wy_c, wx_c = self._wall_normal(
            cu,
            cv,
            cw,
            cg,
            self.iky[None, :, None] * cw,
            self.ikx[:, None, None] * cw,
        )
        uz_c, vz_c = self._wall_normal(cu, cv, kz=1)
        u_p, v_p, w_p, omx, omy, omz = self._horizontal(
            u_c, v_c, w_c, wy_c - vz_c, uz_c - wx_c, omz_c
        )
        Hx = omy * w_p - omz * v_p
        Hy = omz * u_p - omx * w_p
        Hz = omx * v_p - omy * u_p
        hx, hy, hz = self._forward(Hx, Hy, Hz)
        NL_w = self.C_hx @ hx + self.C_hy @ hy + self.C_hz @ hz
        buoyancy = self.buoyancy(scalars)
        if buoyancy is not None:
            NL_w = NL_w + buoyancy
        NL_g = self.C_gfx @ hx + self.C_gfy @ hy
        # Scalar product in z of (-w1, .) against the horizontal mean.
        NL_u0 = self.G_mean @ hx.real[0, 0]
        NL_v0 = self.G_mean @ hy.real[0, 0]
        if self.f0u is not None:
            NL_u0 = NL_u0 + jnp.asarray(self.f0u)
        if self.f0v is not None:
            NL_v0 = NL_v0 + jnp.asarray(self.f0v)
        return NL_w, NL_g, NL_u0, NL_v0, self.scalar_terms(u_p, v_p, w_p, scalars)

    # -- stepping ----------------------------------------------------------

    def _step_impl(
        self,
        state: tuple[Array, ...],
        dt: float,
        N: ScalarPadding = None,
        t: Array | float = 0.0,
        /,
    ) -> tuple[Array, ...]:
        """Advance one IMEX Runge-Kutta step.

        Each field's stage comes from its own `IMEXRungeKutta.stage`, which takes
        the nonlinear and linear caches as plain arrays -- so the explicit terms
        are computed here, in rotational form. The recovery is solved between the
        implicit solves and the explicit evaluation, so the horizontal velocities
        are never lagged: at every stage they are the exact solution of the 2x2
        system for that stage's w, g, u0 and v0.

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
            wi, gi, u0i, v0i, scalars = (
                stage[0],
                stage[1],
                stage[2],
                stage[3],
                stage[4:],
            )
            ui, vi = self.velocity(wi, gi, u0i, v0i)
            NL_w, NL_g, NL_u0, NL_v0, NL_s = self.explicit_terms(
                ui, vi, wi, gi, scalars
            )
            for k, (g, value) in enumerate(
                zip(integrators, (NL_w, NL_g, NL_u0, NL_v0) + NL_s, strict=True)
            ):
                nl[k].append(value)
                li[k].append(g.linear_operator @ stage[k])

        return stage

    def _zero_nyquist(self, state: tuple[Array, ...]) -> tuple[Array, ...]:
        """Return `state` with the Nyquist mode of both Fourier axes cleared.

        The operators use the raw wavenumbers while `backward_primitive` zeroes
        the Nyquist for odd derivatives, so leaving either populated would make
        the two disagree there. On the halved streamwise axis nothing is lost --
        a real field cannot carry a phase on that mode -- while on the full
        spanwise axis this is a deliberate one-mode truncation; see "THE NYQUIST
        MODE ON BOTH FOURIER AXES" in the header.

        Applied at every stage rather than once at the end of the step, so that
        no stage is ever *evaluated* with a mode the two conventions disagree
        about. u0 and v0 are 1-D and have no Fourier direction.
        """
        nx, ny = self.nyquist

        def clear(c: Array) -> Array:
            return c.at[nx].set(0.0).at[:, ny].set(0.0)

        return (clear(state[0]), clear(state[1]), state[2], state[3]) + tuple(
            clear(s) for s in state[4:]
        )

    def _setup_impl(self, dt: float) -> None:
        """Factorize every stage operator before time stepping starts.

        The recovery needs nothing here. Whichever form its projection takes,
        it was built -- and, on the solver path, factorised -- at construction,
        and what follows is elementwise.
        """
        for g in self.integrators:
            # `_step_impl` passes each sub-integrator its `linear_forcing`, not
            # `forcing_at(t)`, so a moving wall would not merely freeze -- the
            # boundary contribution is subtracted out of `linear_forcing` on the
            # transient path and would be dropped entirely. Refuse it here.
            if g._transient_boundary:  # noqa: SLF001
                raise NotImplementedError(
                    "KMM3D does not support time-dependent boundary data; the "
                    "channel walls must be steady."
                )
            g.setup(dt)

    def initial_coefficients(
        self, initial=None, t: float | None = None
    ) -> tuple[Array, ...]:
        """Return the state at rest, plus whatever the subclass contributes."""
        if initial is not None:
            return self._coerce_state(initial)
        return (
            jnp.zeros(self.VB.num_dofs, dtype=complex),
            jnp.zeros(self.VD.num_dofs, dtype=complex),
            jnp.zeros(self.D1.num_dofs),
            jnp.zeros(self.D1.num_dofs),
        ) + self.scalar_initial()

    def _coerce_state(
        self, state0: tuple[Array, ...], t: float | None = None
    ) -> tuple[Array, ...]:
        """Coerce a restart state into one coefficient array per field."""
        w_hat, g_hat, u0, v0, *scalars = state0
        return (
            jnp.asarray(w_hat).reshape(self.VB.num_dofs).astype(complex),
            jnp.asarray(g_hat).reshape(self.VD.num_dofs).astype(complex),
            jnp.asarray(u0).reshape(self.D1.num_dofs).real,
            jnp.asarray(v0).reshape(self.D1.num_dofs).real,
        ) + tuple(
            jnp.asarray(s).reshape(g.trialspace.num_dofs).astype(complex)
            for g, s in zip(self.scalar_integrators, scalars, strict=True)
        )

    # -- diagnostics -------------------------------------------------------

    def average(self, f: Array) -> Array:
        """Return the exact volume average of a padded physical field."""
        return self.Wo.scalar_product(f)[0, 0, 0].real / self.vol

    def courant(self, state: tuple[Array, ...], dt: float) -> float:
        """Return the advective Courant number on the padded mesh.

        Only the diffusive terms are implicit, so the step size is limited by
        advection alone. This is the finite-difference form,
        dt*(|u|/dx + |v|/dy + |w|/dz), and none of the three terms is the
        spectral criterion.

        Along either Fourier direction the operator is i*k*u, so what binds is
        dt*|u|*k_max, larger than dt*|u|/dx by k_max*dx = 2*pi/3 under the 3/2
        padding; ARS443's explicit half is stable to about 2 on the imaginary
        axis, so in two dimensions the two factors very nearly cancelled and the
        number read ~1 at the stability boundary (0.76 ran, 1.15 diverged).

        That calibration does NOT transfer here. It was measured on
        Orr-Sommerfeld in two dimensions, with one periodic direction rather than
        two, and the threshold moved with the padding even there (to ~0.6
        unpadded). Read this number as a relative indicator across runs of one
        configuration until it has been recalibrated, not as an absolute
        stability margin.

        Along the wall-normal direction 1/dz understates the stiffness rather
        than overstating it -- the derivative matrix has norm 3.1x (Legendre) to
        4.6x (Chebyshev) the reciprocal of the smallest spacing, the point
        clustering not being what sets it. What keeps that direction from binding
        is that |w| is small where the points are dense, which is a property of
        the flow and not of the discretization.
        """
        u_p, v_p, w_p = self.velocity_from_state(
            state, pad=self.pad, kind=VelocityKind.PHYSICAL
        )
        xm, ym, zm = self.VD.mesh(N=self.pad, broadcast=False)
        dx = float(self.Lx) / xm.shape[0]
        dy = float(self.Ly) / ym.shape[0]
        dz = jnp.abs(jnp.asarray(jnp.gradient(zm)))[None, None, :]
        return float(
            dt * (jnp.abs(u_p) / dx + jnp.abs(v_p) / dy + jnp.abs(w_p) / dz).max()
        )

    def diagnostics(self, state: tuple[Array, ...]) -> dict[str, float]:
        """Return the structural checks, plus any the subclass adds."""
        u0, v0 = state[2], state[3]
        pad = self.pad
        u_hat, v_hat, w_hat, u_p, v_p, w_p = self.velocity_from_state(
            state, pad=pad, kind=VelocityKind.BOTH
        )
        g_hat = state[1]
        div = (
            self.VD.backward_primitive(u_hat, k=(1, 0, 0), N=pad)
            + self.VD.backward_primitive(v_hat, k=(0, 1, 0), N=pad)
            + self.VB.backward_primitive(w_hat, k=(0, 0, 1), N=pad)
        )
        scale = max(
            float(jnp.abs(u_p).max()),
            float(jnp.abs(v_p).max()),
            float(jnp.abs(w_p).max()),
            1e-300,
        )
        return {
            "div": float(jnp.abs(div).max()) / scale,
            # Both must stay exactly zero; see the header.
            "w[k=0]": float(jnp.abs(w_hat[0, 0]).max()),
            "g[k=0]": float(jnp.abs(g_hat[0, 0]).max()),
            "u[k=0]-u0": float(jnp.abs(u_hat[0, 0].real - u0).max()),
            "v[k=0]-v0": float(jnp.abs(v_hat[0, 0].real - v0).max()),
            "max|u|": float(jnp.abs(u_p).max()),
            "max|v|": float(jnp.abs(v_p).max()),
            "max|w|": float(jnp.abs(w_p).max()),
            # Physical, not the coefficient max: for Poiseuille this must read 1.
            "max|u0|": float(jnp.abs(self.D1.backward(u0)).max()),
            "max|v0|": float(jnp.abs(self.D1.backward(v0)).max()),
        } | self.extra_diagnostics(state)
