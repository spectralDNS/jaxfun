"""Internal helpers shared across the `integrators` package.

Solver-option handling, field-coupling assembly, and shape/boundary lookups are
needed by more than one of `base.py`, `constraint.py`, `system.py`,
`backward_euler.py`, and `imex_rk.py`. They live here instead of in whichever
file happened to define them first. The module itself is the privacy boundary
(not part of `jaxfun.integrators`'s public API), so the names in it are not
individually underscore-prefixed -- except the two that would otherwise read as
same-named-but-different-signature next to an unrelated public API member
(`TimeStepper.solve`, `BaseMatrix.scale`).
"""

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax.numpy as jnp
import sympy as sp
from sympy.core.function import AppliedUndef

from jaxfun.coordinates import get_system
from jaxfun.galerkin import TensorProductSpace, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.la import BaseMatrix
from jaxfun.la.matrixprotocol import SolverNotApplicable
from jaxfun.typing import Array, IntegratorState, ScalarPadding, ScalarSpaceType
from jaxfun.utils import get_time_independent, split_time_derivative_terms
from jaxfun.utils.operator_tools import assemble_linear_term

type SolverOptions = tuple[tuple[str, Any], ...]


def accepted_solve_options(cls: type) -> frozenset[str]:
    """Return the keyword-only options ``cls.solve`` declares.

    Operators differ in what they can be tuned with -- `TPMatrices.solve` picks
    between factored and Kronecker paths, `DiaMatrix.solve` between banded and
    dense, and `Matrix.solve` has nothing to pick -- so each one is offered only
    the options it names.
    """
    solve = getattr(cls, "solve", None)
    if solve is None:
        return frozenset()
    try:
        params = inspect.signature(solve).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtin/C signatures
        return frozenset()
    return frozenset(
        name
        for name, param in params.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    )


@functools.cache
def known_solve_options() -> frozenset[str]:
    """Return every solve option any `BaseMatrix` in `jaxfun.la` accepts."""
    names: set[str] = set()
    pending: list[type] = [BaseMatrix]
    seen: set[type] = set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        pending.extend(cls.__subclasses__())
        names |= accepted_solve_options(cls)
    return frozenset(names)


def validate_solver_options(options: Mapping[str, Any] | None) -> SolverOptions:
    """Normalize user solver options into a hashable, order-independent tuple.

    A mapping cannot be stored as `nnx.static` state: static fields are compared
    (and hashed) on every jit cache lookup. Sorted pairs also make two integrators
    configured the same way compare equal regardless of how the mapping was
    written.

    Raises:
        ValueError: if an option is one no operator in `jaxfun.la` accepts. The
            options are filtered per operator when used, so an unrecognized name
            would otherwise be silently dropped everywhere.
    """
    if not options:
        return ()
    known = known_solve_options()
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(
            f"Unknown solver option(s): {', '.join(unknown)}. "
            f"Accepted options are: {', '.join(sorted(known))}."
        )
    return tuple(sorted(options.items()))


def solve_with_options(
    op: BaseMatrix, rhs: Array, options: SolverOptions = ()
) -> Array:
    """Solve ``op x = rhs``, passing on the options `op` knows what to do with."""
    if not options:
        return op.solve(rhs)
    accepted = accepted_solve_options(type(op))
    return op.solve(rhs, **{k: v for k, v in options if k in accepted})


def warm_operator_solve_cache(
    op: BaseMatrix,
    shape: tuple[int, ...] | None = None,
    options: SolverOptions = (),
) -> None:
    """Warm native solver caches for operators that support factorization.

    Everything a solver decides by inspecting matrix *values* -- which
    factorization applies, a sparsity reordering, the factors themselves -- has
    to happen here, while the matrices are still concrete. Inside the jitted
    step there is no second chance: the arrays are tracers by then.

    Args:
        op: Operator whose solve caches should be populated.
        shape: Coefficient shape of the right-hand sides `op` will be solving.
            When given, an operator with no applicable factored solver is warmed
            by running one throwaway solve, which fills the caches of whichever
            fallback path `solve` settles on.
        options: Solver options, passed on so that the warming solve takes the
            same path the stepping solves will.
    """
    lu_factor = getattr(op, "lu_factor", None)
    if lu_factor is not None:
        try:
            lu_factor()
            return
        except (SolverNotApplicable, ValueError, TypeError, RuntimeError):
            pass
    if shape is None:
        return
    try:
        solve_with_options(op, jnp.zeros(shape), options)
    except (SolverNotApplicable, ValueError, TypeError, RuntimeError):
        return


type FieldCoupling = tuple[int, BaseMatrix, Array | None]


def assemble_field_couplings(
    coupling_exprs: Mapping[Any, sp.Expr],
    node_for: Callable[[Any], AppliedUndef],
    field_order: tuple[AppliedUndef, ...] | None,
    *,
    sparse: bool,
    sparse_tol: int,
) -> tuple[FieldCoupling, ...]:
    """Assemble linear couplings into `(field slot, operator, forcing)` triples.

    Each expression is bilinear in the test function and one foreign field, so
    `inner` assembles it as an operator between the two spaces -- rectangular
    when they differ in size. Applying that is what replaces evaluating the term
    pointwise on the padded mesh.
    """
    if not coupling_exprs:
        return ()
    if field_order is None:
        raise ValueError(
            "Linear couplings to other fields need `field_order`, to know which "
            "entry of the state tuple each field is."
        )
    out: list[FieldCoupling] = []
    for field, expr in coupling_exprs.items():
        operator, forcing = assemble_linear_term(
            expr, sparse=sparse, sparse_tol=sparse_tol
        )
        if operator is None:  # pragma: no cover - a coupling always assembles one
            raise ValueError(f"Coupling term in {field} assembled no operator: {expr}")
        out.append((field_order.index(node_for(field)), operator, forcing))
    return tuple(out)


def apply_field_couplings(
    couplings: Sequence[FieldCoupling], uh: IntegratorState
) -> Array | None:
    """Sum every coupling operator applied to its field; None when there are none."""
    total: Array | None = None
    for slot, operator, forcing in couplings:
        term = operator @ cast(tuple[Array, ...], uh)[slot]
        if forcing is not None:
            term = term + jnp.asarray(forcing)
        total = term if total is None else total + term
    return total


def boundary_values(space) -> list[sp.Expr]:
    """Return the boundary values of every BC-carrying factor of ``space``.

    A `Composite`/`DirectSum` holds its own boundary conditions, while a tensor
    product space delegates to its 1D factors.
    """
    bcs = getattr(space, "bcs", None)
    if bcs is not None:
        return [sp.sympify(val) for val in bcs.orderedvals()]
    if isinstance(space, TensorProductSpace):
        return [val for factor in space for val in boundary_values(factor)]
    return []


def coefficient_shape(V: ScalarSpaceType) -> tuple[int, ...]:
    """Return the coefficient-array shape for the given space."""
    num_dofs = V.num_dofs
    return num_dofs if isinstance(num_dofs, tuple) else (num_dofs,)


def physical_shape(
    space: ScalarSpaceType, N: ScalarPadding
) -> int | tuple[int | None, ...]:
    """Resolve the physical-space shape to evaluate ``space`` at, given padding."""
    if N is None:
        N = space.shape
        if space.dims == 1:
            N = N[0]
    return N


def node_for(
    fields: Sequence[tuple[Any, AppliedUndef]], trial: TrialFunction
) -> AppliedUndef:
    """Return the shared JAXFunction node representing ``trial``.

    Falls back to a fresh node when `trial` isn't in `fields`, which is how a
    standalone (non-system) integrator represents its own field.
    """
    for field, node in fields:
        if field == trial:
            return node
    V = cast(ScalarSpaceType, trial.functionspace)
    return cast(
        AppliedUndef,
        JAXFunction(
            jnp.zeros(coefficient_shape(V)), V, name=f"{trial.name}_jax"
        ).doit(),
    )


def transient_trial(equation: sp.Expr) -> TrialFunction | None:
    """Return the transient field an equation is the evolution equation for.

    The time-derivative term identifies it unambiguously; every other field
    appearing in the equation belongs to a different equation of the system.
    Returns None for a *constraint* equation, which has no time derivative and
    is solved for its field rather than integrated.
    """
    t = get_system(equation).base_time()
    lhs, _ = split_time_derivative_terms(equation, t)
    if sp.sympify(lhs) == 0:
        return None
    _, trial = get_basisfunctions(lhs)
    if not isinstance(trial, TrialFunction):
        raise ValueError(
            "The time-derivative term must contain exactly one TrialFunction. "
            "Equations coupled through their time derivatives are not supported."
        )
    return trial


def own_trial(equation: sp.Expr) -> TrialFunction:
    """Return the transient field an equation evolves, requiring there to be one."""
    trial = transient_trial(equation)
    if trial is None:
        raise ValueError(
            "Time integrators require a first-order time derivative in the weak form"
        )
    return trial


def constrained_trial(
    equation: sp.Expr, transient: Sequence[TrialFunction]
) -> TrialFunction:
    """Return the field a constraint equation determines.

    A constraint carries no time derivative to name its field, so it is the one
    trial function in the equation that no evolution equation already owns.
    """
    _, trials = get_basisfunctions(equation)
    found = trials if isinstance(trials, set) else {trials}
    owned = {get_time_independent(u) for u in transient}
    candidates = {
        u
        for u in found
        if isinstance(u, TrialFunction) and get_time_independent(u) not in owned
    }
    if len(candidates) != 1:
        names = ", ".join(sorted(str(u) for u in candidates)) or "none"
        raise ValueError(
            "A constraint equation must be solvable for exactly one field of its "
            f"own -- a TrialFunction no other equation evolves -- but found {names}. "
            "Add an evolution equation for the extra fields, or drop them."
        )
    return get_time_independent(candidates.pop())


def scale_real(array: Array, coefficient: complex) -> Array:
    """Multiply by a constant, keeping a real coefficient real.

    A Python `complex` would otherwise promote a real coefficient array to
    complex and silently double its memory traffic.
    """
    if coefficient.imag == 0.0:
        return coefficient.real * array
    return coefficient * array


def mesh_axes(V: ScalarSpaceType, N: ScalarPadding) -> tuple[Array, ...]:
    """Return the per-axis quadrature mesh of ``V`` evaluated with padding ``N``."""
    if V.dims == 1:
        return (cast(Array, V.mesh(N=cast(int, N))),)  # ty: ignore[invalid-argument-type]
    mesh = V.mesh(N=N, broadcast=False)  # ty: ignore[invalid-argument-type,unknown-argument]
    return cast(tuple[Array, ...], mesh)
