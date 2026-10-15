"""Compilation helpers for nonlinear physical-space integrator terms."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.vector import VectorAdd

from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.arguments import (
    JAXFunction,
    is_derivative_of_jaxfunc_leaf,
    is_jaxfunc_leaf,
    is_jaxfunc_primitive,
)
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.typing import (
    Array,
    IntegratorState,
    RankTag,
    ScalarPadding,
    ScalarSpaceType,
)
from jaxfun.utils import JAX_FUNCTION_BY_NAME, lambdify, normalize_explicit

from ._utils import component_offset, mesh_axes

type NodeValueCache = dict[sp.Basic, Array]
type NodeEvaluator = Callable[[NodeValueCache, ScalarPadding], Array]


@dataclass(frozen=True)
class NonlinearCompileContext:
    """Static data required to compile nonlinear SymPy expressions.

    An expression may depend on several JAXFunctions -- one per field of a
    coupled system. Each field is evaluated through its own function space,
    resolved from the leaf node itself. `functionspace` is the space whose
    quadrature mesh the expression is evaluated on: static (purely spatial)
    subexpressions are evaluated there, and so is a field whose own space has
    other quadrature points along some axis, such as a Chebyshev field tested
    by a ChebyshevU space.
    """

    spatial_symbols: tuple[sp.Symbol, ...]
    functionspace: ScalarSpaceType
    jaxfunctions: tuple[AppliedUndef, ...]


class NonlinearCompiler:
    """Compile nonlinear SymPy expressions into cached evaluators."""

    def __init__(self, context: NonlinearCompileContext) -> None:
        """Store compile-time context and initialize evaluator caches."""
        self.context = context
        self._compiled: dict[sp.Basic, NodeEvaluator] = {}
        self._static: dict[tuple[sp.Basic, ScalarPadding], Array] = {}
        # SymPy rebuilds nodes during `doit`/`expand`, and a rebuilt leaf loses the
        # instance attribute the evaluator mutates (it would silently read the
        # stale class-level array instead). Rebuilt leaves still compare and hash
        # equal to the originals, so map them back to the objects we own.
        self._registry: dict[sp.Basic, JAXFunction[ScalarSpaceType]] = {
            node: cast(JAXFunction[ScalarSpaceType], node)
            for node in context.jaxfunctions
        }

    def compile(self, expr: sp.Expr) -> NodeEvaluator:
        """Compile a nonlinear SymPy expression into a cached evaluator."""
        # Resolve custom operator wrappers like Div(Grad(...)) before the
        # fallback lambdify path, since SymPy's JAX printer cannot emit them.
        evaluated = sp.sympify(expr).doit()
        return self.compile_node(sp.expand(evaluated))

    def compile_node(self, node: sp.Basic) -> NodeEvaluator:
        """Compile a single SymPy node, reusing cached evaluators when possible."""
        node = sp.sympify(node)
        cached = self._compiled.get(node)
        if cached is not None:
            return cached

        # Product/chain-rule derivatives must be expanded symbolically before
        # evaluation; only direct d^k(u)/dx^k terms are primitive lookups.
        if isinstance(node, sp.Derivative) and not is_derivative_of_jaxfunc_leaf(node):
            expanded = sp.expand(node.doit())
            evaluator = self.compile_node(expanded)
        elif is_jaxfunc_primitive(node):
            evaluator = self.compile_primitive(node)
        elif not contains_jaxfunction(node):
            evaluator = self.memoize(
                node,
                lambda _cache, N=None, node=node: self.get_static_value(node, N),
            )
        else:
            evaluator = self.compile_composite(node)

        self._compiled[node] = evaluator
        return evaluator

    def compile_composite(self, node: sp.Basic) -> NodeEvaluator:
        """Compile a non-primitive node from its recursively compiled children."""
        child_eval = tuple(self.compile_node(arg) for arg in node.args)

        if isinstance(node, sp.Add):
            return self.memoize(node, self.compile_add(child_eval))
        if isinstance(node, sp.Mul):
            return self.memoize(node, self.compile_mul(child_eval))
        if isinstance(node, sp.Pow):
            return self.memoize(node, self.compile_pow(node))
        if isinstance(node, sp.Function):
            return self.memoize(node, self.compile_function(node, child_eval))
        return self.memoize(node, self.compile_lambdified(node, child_eval))

    def compile_primitive(self, node: sp.Basic) -> NodeEvaluator:
        """Compile a primitive field or spatial derivative evaluation.

        The field is resolved from the node itself rather than from the compile
        context, so an expression coupling several fields evaluates each one
        through its own function space.
        """
        leaf = node if is_jaxfunc_leaf(node) else cast(sp.Derivative, node).expr
        jaxf = self._registry.get(leaf, cast(JAXFunction[ScalarSpaceType], leaf))
        space = jaxf.functionspace
        derivative_order: int | tuple[int, ...] = 0
        if not is_jaxfunc_leaf(node):
            assert is_derivative_of_jaxfunc_leaf(node)
            derivative = cast(sp.Derivative, node)
            known = set(self.context.spatial_symbols)
            unknown = tuple(var for var in derivative.variables if var not in known)
            if len(unknown) > 0:
                names = ", ".join(sorted(str(sym) for sym in set(unknown)))
                raise ValueError(
                    f"Only spatial derivatives are supported, got: {names}"
                )

            derivative_counts = tuple(
                derivative.variables.count(sym) for sym in self.context.spatial_symbols
            )
            if sum(derivative_counts) == 0:
                raise ValueError("Derivative order is zero in all spatial coordinates")
            if space.dims == 1:
                derivative_order = int(derivative_counts[0])
            else:
                derivative_order = derivative_counts

        target = self.context.functionspace
        if space.dims == target.dims and not _shares_quadrature(space, target):

            def evaluate_foreign(
                _cache: NodeValueCache,
                N: ScalarPadding = None,
                space: ScalarSpaceType = space,
                jaxf: JAXFunction[ScalarSpaceType] = jaxf,
                derivative_order: int | tuple[int, ...] = derivative_order,
            ) -> Array:
                # The transform of `space` gives its own quadrature points.
                return _evaluate_on_mesh(
                    space,
                    target,
                    cast(Array, jaxf.array),
                    derivative_order,
                    N,
                )

            return self.memoize(node, evaluate_foreign)

        if is_jaxfunc_leaf(node):

            def evaluate_leaf(
                _cache: NodeValueCache,
                N: ScalarPadding = None,
                space: ScalarSpaceType = space,
                jaxf: JAXFunction[ScalarSpaceType] = jaxf,
            ) -> Array:
                return space.backward(jaxf.array, N=N)  # ty: ignore[invalid-argument-type]

            return self.memoize(node, evaluate_leaf)

        def evaluate_derivative(
            _cache: NodeValueCache,
            N: ScalarPadding = None,
            space: ScalarSpaceType = space,
            jaxf: JAXFunction[ScalarSpaceType] = jaxf,
            derivative_order: int | tuple[int, ...] = derivative_order,
        ) -> Array:
            return space.backward_primitive(jaxf.array, k=derivative_order, N=N)  # ty: ignore[invalid-argument-type]

        return self.memoize(node, evaluate_derivative)

    def compile_add(self, child_eval: tuple[NodeEvaluator, ...]) -> NodeEvaluator:
        """Compile an additive node."""

        def evaluate(cache: NodeValueCache, N: ScalarPadding = None) -> Array:
            if len(child_eval) == 0:
                return jnp.asarray(0.0)
            total = jnp.asarray(child_eval[0](cache, N))
            for child in child_eval[1:]:
                total = total + jnp.asarray(child(cache, N))
            return total

        return evaluate

    def compile_mul(self, child_eval: tuple[NodeEvaluator, ...]) -> NodeEvaluator:
        """Compile a multiplicative node."""

        def evaluate(cache: NodeValueCache, N: ScalarPadding = None) -> Array:
            if len(child_eval) == 0:
                return jnp.asarray(1.0)
            product = jnp.asarray(child_eval[0](cache, N))
            for child in child_eval[1:]:
                product = product * jnp.asarray(child(cache, N))
            return product

        return evaluate

    def compile_pow(self, node: sp.Pow) -> NodeEvaluator:
        """Compile an exponentiation node."""
        base_eval = self.compile_node(node.base)
        exp_eval = self.compile_node(node.exp)
        exp = node.exp

        if isinstance(exp, sp.Number):
            exponent = int(exp) if bool(exp.is_integer) else float(exp)
            return lambda cache, N=None, exponent=exponent: (
                jnp.asarray(base_eval(cache, N)) ** exponent
            )

        return lambda cache, N=None: jnp.power(
            jnp.asarray(base_eval(cache, N)),
            jnp.asarray(exp_eval(cache, N)),
        )

    def compile_function(
        self, node: sp.Function, child_eval: tuple[NodeEvaluator, ...]
    ) -> NodeEvaluator:
        """Compile a SymPy function node using JAX primitives when available."""
        if node.func.__name__ == "Heaviside":
            return lambda cache, N=None: jnp.heaviside(
                jnp.asarray(child_eval[0](cache, N)), 0.5
            )

        jnp_function = JAX_FUNCTION_BY_NAME.get(node.func.__name__)
        if jnp_function is None:
            return self.compile_lambdified(node, child_eval)

        def evaluate(cache: NodeValueCache, N: ScalarPadding = None) -> Array:
            values = tuple(jnp.asarray(child(cache, N)) for child in child_eval)
            return jnp.asarray(jnp_function(*values))

        return evaluate

    def compile_lambdified(
        self, node: sp.Basic, child_eval: tuple[NodeEvaluator, ...]
    ) -> NodeEvaluator:
        """Compile a fallback node by lambdifying it against dummy symbols."""
        if len(node.args) == 0:
            raise ValueError(f"Unsupported nonlinear term node: {node}")

        n_args = len(node.args)
        dummies = sp.symbols(f"_z0:{n_args}", real=True)
        expr_with_dummies = node.func(*(dummies if n_args > 1 else dummies[:1]))
        composed = lambdify(
            dummies if n_args > 1 else dummies[0], expr_with_dummies, modules="jax"
        )

        def evaluate(cache: NodeValueCache, N: ScalarPadding = None) -> Array:
            values = tuple(jnp.asarray(child(cache, N)) for child in child_eval)
            if len(values) == 1:
                return jnp.asarray(composed(values[0]))
            return jnp.asarray(composed(*values))

        return evaluate

    def get_static_value(self, node: sp.Basic, N: ScalarPadding = None) -> Array:
        """Evaluate a static symbolic node once on the quadrature mesh."""
        cached = self._static.get((node, N))
        if cached is not None:
            return cached

        evaluated = sp.sympify(sp.sympify(node).doit())
        free_symbols = evaluated.free_symbols
        spatial_symbols = self.context.spatial_symbols
        V = self.context.functionspace
        # Evaluated eagerly even when reached from inside a trace: the value is
        # cached across calls, and a cached tracer is hoisted as a constant by
        # the next trace, which then fails in PJRT with an argument-count error.
        with jax.ensure_compile_time_eval():
            if len(free_symbols) == 0:
                value = float(evaluated) if evaluated.is_real else complex(evaluated)
                out = jnp.asarray(value)
            elif free_symbols.issubset(set(spatial_symbols)):
                # `N` has a lot of different shapes, trust the correct one is
                # passed. `mesh_axes` because a 1D space returns the mesh as a
                # bare array rather than a one-tuple, and splatting that spreads
                # the quadrature points across the arguments one point each.
                out = lambdify(spatial_symbols, evaluated, modules="jax")(
                    *mesh_axes(V, N)
                )
            else:
                names = ", ".join(sorted(str(sym) for sym in free_symbols))
                raise ValueError(
                    f"Unsupported nonlinear term with unresolved symbols: {names}"
                )

        self._static[(node, N)] = out
        return out

    def memoize(self, node: sp.Basic, compute: NodeEvaluator) -> NodeEvaluator:
        """Wrap an evaluator so repeated subexpressions are computed once."""

        def evaluate(cache: NodeValueCache, N: ScalarPadding = None) -> Array:
            if node in cache:
                return cache[node]
            value = jnp.asarray(compute(cache, N))
            cache[node] = value
            return value

        return evaluate


def _axes(space: ScalarSpaceType) -> tuple[OrthogonalSpace, ...]:
    """Return the 1D orthogonal space of each axis of `space`."""
    axes = (space,) if space.dims == 1 else tuple(space)  # ty: ignore[invalid-argument-type]
    return tuple(axis.orthogonal for axis in axes)  # ty: ignore[unresolved-attribute]


def _shares_quadrature(a: ScalarSpaceType, b: ScalarSpaceType) -> bool:
    """Return whether spaces `a` and `b` sample at the same physical points.

    The reference quadrature points of an axis are set by its family and the
    Jacobi parameters within it, and its domain maps them to physical points.
    A field transformed on `a` is sampled at the physical points of `a`, so it
    can only be used with the quadrature of `b` if both agree. A composite or
    direct sum has the points of its orthogonal space.
    """
    return all(
        type(ao) is type(bo)
        and ao.domain == bo.domain
        and getattr(ao, "alpha", None) == getattr(bo, "alpha", None)
        and getattr(ao, "beta", None) == getattr(bo, "beta", None)
        for ao, bo in zip(_axes(a), _axes(b), strict=True)
    )


def _evaluate_on_mesh(
    space: ScalarSpaceType,
    target: ScalarSpaceType,
    c: Array,
    k: int | tuple[int, ...],
    N: ScalarPadding,
) -> Array:
    """Return the field `c` of `space` on the quadrature mesh of `target`.

    Args:
        space: Space of the coefficients `c`.
        target: Space whose quadrature mesh to evaluate on, of the same
            dimension as `space`.
        c: Coefficients in `space`.
        k: Derivative order, one per axis, or one for all axes.
        N: Number of quadrature points of `target`, one per axis for more than
            one dimension, or None for its default.

    Returns:
        The k'th derivative of the field at the quadrature points of `target`.
    """
    axes, points = _axes(space), _axes(target)
    k = (k,) * len(axes) if isinstance(k, int) else k
    counts = cast(
        tuple[int | None, ...],
        (N,) if space.dims == 1 else ((None,) * space.dims if N is None else N),
    )
    c = space.to_orthogonal(c)
    for ax, (s, t) in enumerate(zip(axes, points, strict=True)):
        X = s.map_reference_domain(t.mesh(N=counts[ax]))
        M = s.eval_reconstruction(X)
        if k[ax] > 0:
            # Row m of D holds the coefficients of the k'th derivative of mode m.
            D = jax.vmap(lambda e, s=s, ax=ax: s.derivative_coeffs(e, k[ax]))(
                jnp.eye(s.N)
            )
            M = float(s.domain_factor ** k[ax]) * (M @ D.T)
        c = jnp.moveaxis(jnp.tensordot(M, c, axes=(1, ax)), 0, ax)
    return c.real if any(s.is_hermitian_half for s in axes) else c


def remove_test_function(expr: sp.Expr, test: TestFunction) -> sp.Expr:
    """Replace the test function with ``1`` in a weak-form expression."""
    return expr.subs(test, 1) if expr.has(test) else expr


def replace_trial_with_jaxfunction(
    expr: sp.Expr, trial: TrialFunction, jax_func: AppliedUndef
) -> sp.Expr:
    """Replace the trial function in ``expr`` with the supplied JAXFunction."""
    if not expr.has(trial):
        return expr
    return cast(sp.Expr, expr.replace(trial, jax_func))


def contains_jaxfunction(expr: sp.Basic) -> bool:
    """Return True when any subexpression depends on the JAX-backed field."""
    return any(
        is_jaxfunc_leaf(node) for node in sp.core.traversal.preorder_traversal(expr)
    )


def compile_nonlinear_evaluator(
    expr: sp.Expr,
    functionspace: ScalarSpaceType,
    jaxfunction: AppliedUndef | tuple[AppliedUndef, ...],
) -> Callable[[Array | tuple[Array, ...], ScalarPadding], Array]:
    """Compile a nonlinear physical-space evaluator for coefficient states.

    Args:
        expr: Nonlinear physical-space expression in one or more JAXFunctions.
        functionspace: Space used to evaluate static (purely spatial) factors.
        jaxfunction: The JAXFunction the expression depends on, or a tuple of
            them for a system of coupled equations. The returned evaluator takes
            a matching tuple of coefficient arrays in the same order.
    """
    multi = isinstance(jaxfunction, tuple)
    nodes = cast(
        tuple[JAXFunction[ScalarSpaceType], ...],
        jaxfunction if multi else (jaxfunction,),
    )
    context = NonlinearCompileContext(
        spatial_symbols=tuple(functionspace.system.base_scalars()),
        functionspace=functionspace,
        jaxfunctions=cast(tuple[AppliedUndef, ...], nodes),
    )
    compiled = NonlinearCompiler(context).compile(expr)

    def evaluate(uh: Array | tuple[Array, ...], N: ScalarPadding = None) -> Array:
        arrays = cast(tuple[Array, ...], uh) if multi else (cast(Array, uh),)
        for node, array in zip(nodes, arrays, strict=True):
            node.array = array
        return compiled({}, N)

    return evaluate


def compile_field_evaluator(
    nonlinear_expr: sp.Expr,
    own_trial: TrialFunction,
    explicit_trials: Sequence[TrialFunction],
    node_for: Callable[[TrialFunction], AppliedUndef],
    functionspace: ScalarSpaceType,
    field_order: tuple[AppliedUndef, ...] | None = None,
) -> tuple[sp.Expr, AppliedUndef, Callable[[IntegratorState, ScalarPadding], Array]]:
    """Substitute every field of a nonlinear term by its node, then compile it.

    Shared by the evolution equations and the constraint equations of a system:
    both need every TrialFunction -- their own and the foreign ones they are
    coupled to -- replaced by the system's shared JAXFunction nodes before the
    term can be evaluated in physical space.

    Args:
        nonlinear_expr: Nonlinear part of a weak form, test function removed.
        own_trial: The equation's own field, time-independent.
        explicit_trials: Fields of the other equations, lagged into this term.
        node_for: Resolves a field to the shared node representing it.
        functionspace: Space used to evaluate static (purely spatial) factors.
        field_order: Global field order the compiled evaluator takes its
            coefficient tuple in. `None` for a standalone scalar equation, whose
            evaluator then takes that one field's coefficients directly.

    Returns:
        The substituted expression, the node standing for `own_trial`, and the
        compiled evaluator.
    """
    jaxfunction = node_for(own_trial)
    nonlinear_expr = replace_trial_with_jaxfunction(
        nonlinear_expr, own_trial, jaxfunction
    )
    for item in normalize_explicit(explicit_trials):
        foreign = cast(TrialFunction, item)
        nonlinear_expr = replace_trial_with_jaxfunction(
            nonlinear_expr, foreign, node_for(foreign)
        )
    if nonlinear_expr.atoms(TrialFunction):
        raise ValueError(
            "Nonlinear term still contains TrialFunctions after substitution: "
            f"{nonlinear_expr}. This usually means a coupled field was not "
            "declared through `explicit_trials`."
        )
    # `compile_nonlinear_evaluator` lambdifies against the base scalars only, so
    # a stray time symbol would survive as a free name and surface as a
    # `NameError` from inside a traced evaluator, thousands of lines from the
    # cause. Refuse it where it can still be pointed at.
    time = functionspace.system.base_time()
    if nonlinear_expr.has(time):
        raise NotImplementedError(
            f"Nonlinear term depends on {time}: {nonlinear_expr}. Time-dependent "
            "data is supported in linear source terms and boundary values, not "
            "in terms evaluated pointwise in physical space."
        )
    evaluator = compile_nonlinear_evaluator(
        nonlinear_expr,
        functionspace,
        field_order if field_order is not None else jaxfunction,
    )
    return nonlinear_expr, jaxfunction, evaluator


def compile_coupled_nonlinear_evaluator(
    exprs: Sequence[sp.Expr],
    functionspace: ScalarSpaceType,
    jaxfunctions: tuple[AppliedUndef, ...],
) -> Callable[[tuple[Array, ...], ScalarPadding], tuple[Array, ...]]:
    """Compile several nonlinear expressions that share their field evaluations.

    Compiling the equations of a system together, through one `NonlinearCompiler`
    and one per-call value cache, means work common to several equations is done
    once per evaluation instead of once per equation. The backward transform of
    each field is the expensive part: the equations of a reaction-diffusion
    system all read the same fields, so sharing removes a transform per field per
    equation beyond the first.

    Args:
        exprs: One nonlinear physical-space expression per equation.
        functionspace: Space used to evaluate static (purely spatial) factors.
            All fields must share its quadrature mesh, since the expressions are
            evaluated pointwise.
        jaxfunctions: The system's shared JAXFunction nodes, in global field
            order. The returned evaluator takes a matching tuple of coefficient
            arrays and returns one physical-space array per expression.
    """
    context = NonlinearCompileContext(
        spatial_symbols=tuple(functionspace.system.base_scalars()),
        functionspace=functionspace,
        jaxfunctions=jaxfunctions,
    )
    # One compiler for all equations, so equal subexpressions compile to the
    # *same* memoizing evaluator object and therefore share a cache entry.
    compiler = NonlinearCompiler(context)
    compiled = tuple(compiler.compile(expr) for expr in exprs)
    nodes = cast(tuple[JAXFunction[ScalarSpaceType], ...], jaxfunctions)

    def evaluate(uh: tuple[Array, ...], N: ScalarPadding = None) -> tuple[Array, ...]:
        for node, array in zip(nodes, uh, strict=True):
            node.array = array
        # Fresh per call and shared across the equations within it: reusing a
        # cache across calls would serve values computed from an earlier stage.
        cache: NodeValueCache = {}
        return tuple(evaluator(cache, N) for evaluator in compiled)

    return evaluate


def compile_block_nonlinear_evaluator(
    expr: sp.Expr,
    fields: Sequence[TrialFunction],
    test_leaf: Any,
    functionspace: ScalarSpaceType,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    Callable[[tuple[Array, ...], ScalarPadding], tuple[Array, ...]],
]:
    """Compile the nonlinear terms of a coupled weak form over a mixed space.

    Every field is replaced by a JAXFunction node -- a vector field by one
    scalar node per component, times the base vectors -- and every test function
    by symbolic weights, which splits the form into one pointwise expression
    per flattened test component.

    Args:
        expr: Nonlinear part of the weak form, test functions still attached.
        fields: The top-level (time-independent) trial functions of the mixed
            trial space, in its order.
        test_leaf: The mixed test space.
        functionspace: Space used to evaluate static (purely spatial) factors.

    Returns:
        `(components, used, evaluator)`: the flattened test components with a
        nonlinear term, the flattened trial components the terms read, and an
        evaluator taking one coefficient array per flattened trial component and
        returning one physical-space array per entry of `components`.
    """
    from jaxfun.operators import Curl, Div, Grad

    system = functionspace.system
    base_vectors = system.base_vectors()

    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise ValueError(
            f"Coupled fields must have unique names, got {names}. Rename the "
            "TrialFunctions so each field of the system is distinguishable."
        )

    replacements: dict[sp.Basic, sp.Basic] = {}
    nodes: list[AppliedUndef] = []
    for field in fields:
        V = field.functionspace
        if V.rank == RankTag.VECTOR:
            dofs = cast(tuple[tuple[int, ...], ...], V.num_dofs)
            zeros = tuple(jnp.zeros(nd) for nd in dofs)
            vector = JAXFunction(zeros, V, name=f"{field.name}_jax").doit()
            components = cast(VectorAdd, vector).components
            nodes.extend(cast(AppliedUndef, components[b]) for b in base_vectors)
            replacements[field] = vector
        else:
            nd = V.num_dofs
            node = JAXFunction(
                jnp.zeros(nd if isinstance(nd, tuple) else (nd,)),
                V,
                name=f"{field.name}_jax",
            ).doit()
            nodes.append(cast(AppliedUndef, node))
            replacements[field] = node

    n_tests = len(test_leaf.flatten())
    weights = tuple(sp.Dummy(f"w{c}") for c in range(n_tests))
    for test in expr.atoms(TestFunction):
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Derivative | Grad | Div | Curl) and node.has(test):
                raise NotImplementedError(
                    f"Nonlinear term differentiates the test function {test}: "
                    f"{node}. Write explicit terms against the undifferentiated "
                    "test function."
                )
        k = component_offset(test.functionspace)
        if test.functionspace.rank == RankTag.VECTOR:
            replacements[test] = VectorAdd.fromiter(
                weights[k + i] * b for i, b in enumerate(base_vectors)
            )
        else:
            replacements[test] = weights[k]

    expanded = sp.expand(sp.sympify(expr).xreplace(replacements).doit())
    if expanded.atoms(TrialFunction):
        raise ValueError(
            "Nonlinear term still contains TrialFunctions after substitution: "
            f"{expanded}. Every field must belong to the same mixed trial space."
        )
    time = system.base_time()
    if expanded.has(time):
        raise NotImplementedError(
            f"Nonlinear term depends on {time}: {expanded}. Time-dependent data "
            "is not supported in terms evaluated pointwise in physical space."
        )
    components: list[int] = []
    exprs: list[sp.Expr] = []
    for c, w in enumerate(weights):
        term = expanded.xreplace({wk: (1 if wk is w else 0) for wk in weights})
        if sp.sympify(term) != 0:
            components.append(c)
            exprs.append(term)
    evaluator = compile_coupled_nonlinear_evaluator(exprs, functionspace, tuple(nodes))
    used = tuple(k for k, node in enumerate(nodes) if any(e.has(node) for e in exprs))
    return tuple(components), used, evaluator
