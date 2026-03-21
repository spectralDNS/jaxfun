"""Compilation helpers for nonlinear physical-space integrator terms."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

import jax.numpy as jnp
import sympy as sp
from sympy.core.function import AppliedUndef

from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.arguments import ArgumentTag, JAXFunction, get_arg
from jaxfun.typing import Array, FunctionSpaceType
from jaxfun.utils import lambdify

type NodeValueCache = dict[sp.Basic, Array]
type NodeEvaluator = Callable[[NodeValueCache], Array]


@dataclass(frozen=True)
class _NonlinearCompileContext:
    """Static data required to compile nonlinear SymPy expressions."""

    spatial_symbols: tuple[sp.Symbol, ...]
    functionspace: FunctionSpaceType
    mesh: tuple[Array, ...]
    expected_shape: tuple[int, ...]
    jaxfunction: AppliedUndef


_JAX_FUNCTION_BY_NAME: dict[str, Callable[..., Array]] = {
    "Abs": jnp.abs,
    "acos": jnp.arccos,
    "acosh": jnp.arccosh,
    "asin": jnp.arcsin,
    "asinh": jnp.arcsinh,
    "atan": jnp.arctan,
    "atan2": jnp.arctan2,
    "atanh": jnp.arctanh,
    "cos": jnp.cos,
    "cosh": jnp.cosh,
    "exp": jnp.exp,
    "log": jnp.log,
    "sign": jnp.sign,
    "sin": jnp.sin,
    "sinh": jnp.sinh,
    "sqrt": jnp.sqrt,
    "tan": jnp.tan,
    "tanh": jnp.tanh,
}


class _NonlinearCompiler:
    """Compile nonlinear SymPy expressions into cached evaluators."""

    def __init__(self, context: _NonlinearCompileContext) -> None:
        """Store compile-time context and initialize evaluator caches."""
        self.context = context
        self._compiled: dict[sp.Basic, NodeEvaluator] = {}
        self._static: dict[sp.Basic, Array] = {}

    def compile(self, expr: sp.Expr) -> NodeEvaluator:
        """Compile a nonlinear SymPy expression into a cached evaluator."""
        return self._compile_node(sp.expand(expr))

    def _compile_node(self, node: sp.Basic) -> NodeEvaluator:
        """Compile a single SymPy node, reusing cached evaluators when possible."""
        node = sp.sympify(node)
        cached = self._compiled.get(node)
        if cached is not None:
            return cached

        # Product/chain-rule derivatives must be expanded symbolically before
        # evaluation; only direct d^k(u)/dx^k terms are primitive lookups.
        if isinstance(node, sp.Derivative) and not _is_direct_jax_derivative(node):
            expanded = sp.expand(node.doit())
            evaluator = self._compile_node(expanded)
        elif _is_jaxfunction_primitive(node):
            evaluator = self._compile_primitive(node)
        elif not _contains_jaxfunction(node):
            value = self._get_static_value(node)
            evaluator = lambda _cache, value=value: value
        else:
            evaluator = self._compile_composite(node)

        self._compiled[node] = evaluator
        return evaluator

    def _compile_composite(self, node: sp.Basic) -> NodeEvaluator:
        """Compile a non-primitive node from its recursively compiled children."""
        child_eval = tuple(self._compile_node(arg) for arg in node.args)

        if isinstance(node, sp.Add):
            return self._memoize(node, self._compile_add(child_eval))
        if isinstance(node, sp.Mul):
            return self._memoize(node, self._compile_mul(child_eval))
        if isinstance(node, sp.Pow):
            return self._memoize(node, self._compile_pow(node))
        if isinstance(node, sp.Function):
            return self._memoize(node, self._compile_function(node, child_eval))
        return self._memoize(node, self._compile_lambdified(node, child_eval))

    def _compile_primitive(self, node: sp.Basic) -> NodeEvaluator:
        """Compile a primitive field or spatial derivative evaluation."""
        space = self.context.functionspace
        jaxf = cast(JAXFunction, self.context.jaxfunction)
        if _is_jaxfunction_leaf(node):

            def evaluate_leaf(
                _cache: NodeValueCache,
                space: FunctionSpaceType = space,
                jaxf: JAXFunction = jaxf,
            ) -> Array:
                return space.backward_primitive(jaxf.array, k=0)

            return self._memoize(node, evaluate_leaf)

        assert _is_direct_jax_derivative(node)
        derivative = cast(sp.Derivative, node)
        known = set(self.context.spatial_symbols)
        unknown = tuple(var for var in derivative.variables if var not in known)
        if len(unknown) > 0:
            names = ", ".join(sorted(str(sym) for sym in set(unknown)))
            raise ValueError(f"Only spatial derivatives are supported, got: {names}")

        derivative_counts = tuple(
            derivative.variables.count(sym) for sym in self.context.spatial_symbols
        )
        if sum(derivative_counts) == 0:
            raise ValueError("Derivative order is zero in all spatial coordinates")
        derivative_order: int | tuple[int, ...]
        if self.context.functionspace.dims == 1:
            derivative_order = int(derivative_counts[0])
        else:
            derivative_order = derivative_counts

        def evaluate_derivative(
            _cache: NodeValueCache,
            space: FunctionSpaceType = space,
            jaxf: JAXFunction = jaxf,
            derivative_order: int | tuple[int, ...] = derivative_order,
        ) -> Array:
            return space.backward_primitive(jaxf.array, k=derivative_order)

        return self._memoize(node, evaluate_derivative)

    def _compile_add(self, child_eval: tuple[NodeEvaluator, ...]) -> NodeEvaluator:
        """Compile an additive node."""

        def evaluate(cache: NodeValueCache) -> Array:
            if len(child_eval) == 0:
                return jnp.asarray(0.0)
            total = jnp.asarray(child_eval[0](cache))
            for child in child_eval[1:]:
                total = total + jnp.asarray(child(cache))
            return total

        return evaluate

    def _compile_mul(self, child_eval: tuple[NodeEvaluator, ...]) -> NodeEvaluator:
        """Compile a multiplicative node."""

        def evaluate(cache: NodeValueCache) -> Array:
            if len(child_eval) == 0:
                return jnp.asarray(1.0)
            product = jnp.asarray(child_eval[0](cache))
            for child in child_eval[1:]:
                product = product * jnp.asarray(child(cache))
            return product

        return evaluate

    def _compile_pow(self, node: sp.Pow) -> NodeEvaluator:
        """Compile an exponentiation node."""
        base_eval = self._compile_node(node.base)
        exp_eval = self._compile_node(node.exp)
        exp = node.exp

        if isinstance(exp, sp.Number):
            exponent = int(exp) if bool(exp.is_integer) else float(exp)
            return lambda cache, exponent=exponent: (
                jnp.asarray(base_eval(cache)) ** exponent
            )

        return lambda cache: jnp.power(
            jnp.asarray(base_eval(cache)),
            jnp.asarray(exp_eval(cache)),
        )

    def _compile_function(
        self, node: sp.Function, child_eval: tuple[NodeEvaluator, ...]
    ) -> NodeEvaluator:
        """Compile a SymPy function node using JAX primitives when available."""
        if node.func.__name__ == "Heaviside":
            return lambda cache: jnp.heaviside(jnp.asarray(child_eval[0](cache)), 0.5)

        jnp_function = _JAX_FUNCTION_BY_NAME.get(node.func.__name__)
        if jnp_function is None:
            return self._compile_lambdified(node, child_eval)

        def evaluate(cache: NodeValueCache) -> Array:
            values = tuple(jnp.asarray(child(cache)) for child in child_eval)
            return jnp.asarray(jnp_function(*values))

        return evaluate

    def _compile_lambdified(
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

        def evaluate(cache: NodeValueCache) -> Array:
            values = tuple(jnp.asarray(child(cache)) for child in child_eval)
            if len(values) == 1:
                return jnp.asarray(composed(values[0]))
            return jnp.asarray(composed(*values))

        return evaluate

    def _get_static_value(self, node: sp.Basic) -> Array:
        """Evaluate a static symbolic node once on the quadrature mesh."""
        cached = self._static.get(node)
        if cached is not None:
            return cached

        evaluated = sp.sympify(sp.sympify(node).doit())
        free_symbols = evaluated.free_symbols
        if len(free_symbols) == 0:
            value = float(evaluated) if evaluated.is_real else complex(evaluated)
            out = jnp.asarray(value)
        elif free_symbols.issubset(set(self.context.spatial_symbols)):
            symbols = self.context.spatial_symbols
            if len(symbols) == 1:
                sampled = lambdify(symbols[0], evaluated, modules="jax")(
                    self.context.mesh[0]
                )
            else:
                sampled = lambdify(symbols, evaluated, modules="jax")(
                    *self.context.mesh
                )
            out = jnp.asarray(sampled)
        else:
            names = ", ".join(sorted(str(sym) for sym in free_symbols))
            raise ValueError(
                f"Unsupported nonlinear term with unresolved symbols: {names}"
            )

        self._static[node] = out
        return out

    def _memoize(self, node: sp.Basic, compute: NodeEvaluator) -> NodeEvaluator:
        """Wrap an evaluator so repeated subexpressions are computed once."""

        def evaluate(cache: NodeValueCache) -> Array:
            if node in cache:
                return cache[node]
            value = jnp.asarray(compute(cache))
            cache[node] = value
            return value

        return evaluate


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


def _is_jaxfunction_leaf(expr: sp.Basic) -> bool:
    """Return True when ``expr`` is the base JAXFunction field."""
    return get_arg(expr) is ArgumentTag.JAXFUNC


def _contains_jaxfunction(expr: sp.Basic) -> bool:
    """Return True when any subexpression depends on the JAX-backed field."""
    return any(
        _is_jaxfunction_leaf(node)
        for node in sp.core.traversal.preorder_traversal(expr)
    )


def _is_direct_jax_derivative(expr: sp.Basic) -> bool:
    """Return True for direct spatial derivatives of the JAXFunction field."""
    return isinstance(expr, sp.Derivative) and _is_jaxfunction_leaf(expr.expr)


def _is_jaxfunction_primitive(expr: sp.Basic) -> bool:
    """Return True for primitive field/derivative nodes handled directly."""
    return _is_jaxfunction_leaf(expr) or _is_direct_jax_derivative(expr)


def _normalize_nonlinear_output(out: Array, expected_shape: tuple[int, ...]) -> Array:
    """Coerce nonlinear output to the mesh shape expected by the function space."""
    out = jnp.asarray(out)
    if out.ndim == 0:
        return jnp.broadcast_to(out, expected_shape)

    if (
        out.ndim == len(expected_shape) + 1
        and out.shape[-1] == 1
        and out.shape[:-1] == expected_shape
    ):
        out = jnp.squeeze(out, axis=-1)

    if out.shape == expected_shape:
        return out

    if out.size == prod(expected_shape):
        out = out.reshape(expected_shape)
    else:
        message = (
            f"Nonlinear evaluation returned shape {out.shape}; "
            f"expected {expected_shape}"
        )
        raise ValueError(message)

    return out


def _as_mesh_tuple(mesh: Array | Sequence[Array]) -> tuple[Array, ...]:
    """Normalize mesh inputs to a tuple of JAX arrays."""
    if isinstance(mesh, tuple | list):
        return tuple(jnp.asarray(m) for m in mesh)
    return (jnp.asarray(mesh),)


def compile_nonlinear_evaluator(
    expr: sp.Expr,
    functionspace: FunctionSpaceType,
    mesh: Array | tuple[Array, ...],
    jaxfunction: AppliedUndef,
) -> Callable[[Array], Array]:
    """Compile a nonlinear physical-space evaluator for coefficient states."""
    mesh_tuple = _as_mesh_tuple(mesh)
    expected_shape = jnp.broadcast_shapes(*(m.shape for m in mesh_tuple))
    context = _NonlinearCompileContext(
        spatial_symbols=tuple(functionspace.system.base_scalars()),
        functionspace=functionspace,
        mesh=mesh_tuple,
        expected_shape=expected_shape,
        jaxfunction=jaxfunction,
    )
    compiled = _NonlinearCompiler(context).compile(expr)

    def evaluate(uh: Array) -> Array:
        cast(JAXFunction, jaxfunction).array = uh
        return _normalize_nonlinear_output(compiled({}), expected_shape)

    return evaluate
