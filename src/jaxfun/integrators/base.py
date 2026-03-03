from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import sympy as sp
import tqdm
from flax import nnx
from jax.experimental.sparse import BCOO
from sympy.core.function import AppliedUndef

from jaxfun.galerkin import Composite, DirectSum, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import (
    ArgumentTag,
    JAXFunction,
    get_arg,
)
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import inner, project1D
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.galerkin.tensorproductspace import TensorMatrix, TPMatrices, TPMatrix
from jaxfun.typing import (
    Array,
    GalerkinAssembledForm,
    GalerkinOperator,
    GalerkinOperatorLike,
)
from jaxfun.utils import (
    lambdify,
    split_linear_nonlinear_terms,
    split_time_derivative_terms,
)

IntegratorSpace = OrthogonalSpace | DirectSum | Composite


@dataclass(frozen=True)
class _NonlinearCompileContext:
    x_symbol: sp.Symbol
    functionspace: IntegratorSpace
    orthogonal: OrthogonalSpace
    x_ref: Array
    jaxfunction: AppliedUndef


NodeValueCache = dict[sp.Basic, Array]
NodeEvaluator = Callable[[NodeValueCache], Array]

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
        self.context = context
        self._compiled: dict[sp.Basic, NodeEvaluator] = {}
        self._static: dict[sp.Basic, Array] = {}

    def compile(self, expr: sp.Expr) -> NodeEvaluator:
        return self._compile_node(sp.expand(expr))

    def _compile_node(self, node: sp.Basic) -> NodeEvaluator:
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
        space = self.context.functionspace
        jaxf = cast(JAXFunction, self.context.jaxfunction)
        if _is_jaxfunction_leaf(node):

            def evaluate_leaf(
                _cache: NodeValueCache,
                space: IntegratorSpace = space,
                jaxf: JAXFunction = jaxf,
            ) -> Array:
                return space.evaluate_nonlinear_primitive(jaxf.array)

            return self._memoize(node, evaluate_leaf)

        assert _is_direct_jax_derivative(node)
        derivative = cast(sp.Derivative, node)
        if any(var != self.context.x_symbol for var in derivative.variables):
            msg = "Only derivatives in the primary spatial coordinate are supported"
            raise ValueError(msg)

        k = int(derivative.derivative_count)

        def evaluate_derivative(
            _cache: NodeValueCache,
            space: IntegratorSpace = space,
            jaxf: JAXFunction = jaxf,
            k: int = k,
        ) -> Array:
            return space.evaluate_nonlinear_primitive(jaxf.array, derivative_order=k)

        return self._memoize(node, evaluate_derivative)

    def _compile_add(self, child_eval: tuple[NodeEvaluator, ...]) -> NodeEvaluator:
        def evaluate(cache: NodeValueCache) -> Array:
            if len(child_eval) == 0:
                return jnp.asarray(0.0)
            total = jnp.asarray(child_eval[0](cache))
            for child in child_eval[1:]:
                total = total + jnp.asarray(child(cache))
            return total

        return evaluate

    def _compile_mul(self, child_eval: tuple[NodeEvaluator, ...]) -> NodeEvaluator:
        def evaluate(cache: NodeValueCache) -> Array:
            if len(child_eval) == 0:
                return jnp.asarray(1.0)
            product = jnp.asarray(child_eval[0](cache))
            for child in child_eval[1:]:
                product = product * jnp.asarray(child(cache))
            return product

        return evaluate

    def _compile_pow(self, node: sp.Pow) -> NodeEvaluator:
        base_eval = self._compile_node(node.base)
        exp_eval = self._compile_node(node.exp)
        exp = node.exp

        if isinstance(exp, sp.Number):
            exponent = int(exp) if bool(exp.is_integer) else float(exp)
            return lambda cache, exponent=exponent: (
                jnp.asarray(base_eval(cache)) ** exponent
            )

        return lambda cache: jnp.power(
            jnp.asarray(base_eval(cache)), jnp.asarray(exp_eval(cache))
        )

    def _compile_function(
        self, node: sp.Function, child_eval: tuple[NodeEvaluator, ...]
    ) -> NodeEvaluator:
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
        cached = self._static.get(node)
        if cached is not None:
            return cached

        evaluated = sp.sympify(node).doit()
        free_symbols = evaluated.free_symbols
        if len(free_symbols) == 0:
            value = float(evaluated) if evaluated.is_real else complex(evaluated)
            out = jnp.asarray(value)
        elif free_symbols.issubset({self.context.x_symbol}):
            mapped = self.context.orthogonal.map_expr_true_domain(evaluated)
            sampled = lambdify(self.context.x_symbol, mapped, modules="jax")(
                self.context.x_ref
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
    return get_arg(expr) is ArgumentTag.JAXFUNC


def _contains_jaxfunction(expr: sp.Basic) -> bool:
    return any(
        _is_jaxfunction_leaf(node)
        for node in sp.core.traversal.preorder_traversal(expr)
    )


def _is_direct_jax_derivative(expr: sp.Basic) -> bool:
    return isinstance(expr, sp.Derivative) and _is_jaxfunction_leaf(expr.expr)


def _is_jaxfunction_primitive(expr: sp.Basic) -> bool:
    return _is_jaxfunction_leaf(expr) or _is_direct_jax_derivative(expr)


def _normalize_nonlinear_output(out: Array, expected_size: int) -> Array:
    out = jnp.asarray(out)
    if out.ndim == 0:
        return jnp.broadcast_to(out, (expected_size,))
    if out.ndim == 2 and out.shape[1] == 1:
        out = jnp.squeeze(out, axis=1)
    if out.ndim != 1:
        out = out.reshape((-1,))
    if out.shape[0] != expected_size:
        expected_shape = (expected_size,)
        message = (
            f"Nonlinear evaluation returned shape {out.shape}; "
            f"expected {expected_shape}"
        )
        raise ValueError(message)
    return out


def _compile_nonlinear_evaluator(
    expr: sp.Expr,
    functionspace: IntegratorSpace,
    x_ref: Array,
    jaxfunction: AppliedUndef,
) -> Callable[[Array], Array]:
    context = _NonlinearCompileContext(
        x_symbol=functionspace.system.base_scalars()[0],
        functionspace=functionspace,
        orthogonal=functionspace.orthogonal,
        x_ref=x_ref,
        jaxfunction=jaxfunction,
    )
    compiled = _NonlinearCompiler(context).compile(expr)
    expected_size = x_ref.shape[0]

    def evaluate(uh: Array) -> Array:
        cast(JAXFunction, jaxfunction).array = uh
        return _normalize_nonlinear_output(compiled({}), expected_size)

    return evaluate


def _bcoo_diagonal(mat: BCOO) -> Array | None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return None
    indices = mat.indices
    if indices.shape[1] != 2:
        return None
    if not bool(jnp.all(indices[:, 0] == indices[:, 1])):
        return None
    diag = jnp.zeros(mat.shape[0], dtype=mat.data.dtype)
    return diag.at[indices[:, 0]].add(mat.data)


def _sum_diagonals(operators: list[GalerkinOperator]) -> Array | None:
    diag_sum: Array | None = None
    for operator in operators:
        diag = _diag_from_matrix(operator)
        if diag is None:
            return None
        diag_sum = diag if diag_sum is None else diag_sum + diag
    return diag_sum


def _tpmatrix_diagonal(op: TPMatrix) -> Array | None:
    if len(op.mats) == 0:
        return None

    diagonals: list[Array] = []
    for diag in (_diag_from_matrix(item) for item in op.mats):
        if diag is None or diag.ndim != 1:
            return None
        diagonals.append(diag)

    diagonal = diagonals[0]
    for axis, diag in enumerate(diagonals[1:], start=1):
        shape = (1,) * axis + (diag.shape[0],)
        diagonal = diagonal[..., None] * diag.reshape(shape)
    return diagonal


def _diag_from_matrix(obj: GalerkinOperatorLike | None) -> Array | None:
    if obj is None:
        return None
    if isinstance(obj, list):
        return _sum_diagonals(cast(list[GalerkinOperator], obj))
    if isinstance(obj, BCOO):
        return _bcoo_diagonal(obj)
    if isinstance(obj, TPMatrices):
        return _sum_diagonals(cast(list[GalerkinOperator], list(obj.tpmats)))
    if isinstance(obj, TPMatrix):
        return _tpmatrix_diagonal(obj)
    if isinstance(obj, TensorMatrix):
        return _diag_from_matrix(obj.mat)
    arr = jnp.asarray(obj)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        diag = jnp.diag(arr)
        if bool(jnp.allclose(arr, jnp.diag(diag), atol=1e-12)):
            return diag
    return None


def _apply_operator(op: GalerkinOperatorLike | None, u: Array) -> Array:
    if op is None:
        return jnp.zeros_like(u)
    if isinstance(op, list):
        items = cast(list[GalerkinOperator], op)
        applied = [_apply_operator(item, u) for item in items]
        return jnp.sum(jnp.stack(applied), axis=0)
    if isinstance(op, TPMatrices | TPMatrix):
        return op(u)
    if isinstance(op, TensorMatrix):
        return _apply_operator(op.mat, u)
    if isinstance(op, BCOO):
        return op @ u
    arr = jnp.asarray(op)
    if arr.ndim == u.ndim:
        return arr * u
    if arr.ndim != 2:
        raise ValueError("Can only apply rank-1 or rank-2 operators")
    return arr @ u


def _sum_dense_operators(operators: list[GalerkinOperator]) -> Array:
    mats = [_operator_to_dense(op) for op in operators]
    return sum(mats[1:], mats[0]) if len(mats) > 0 else jnp.array([])


def _tpmatrix_to_dense(op: TPMatrix) -> Array:
    mats = [_operator_to_dense(item) for item in op.mats]
    if len(mats) == 0:
        return jnp.array([])

    dense: Array = mats[0]
    for mat in mats[1:]:
        dense = jnp.kron(dense, mat)
    return cast(Array, dense * jnp.asarray(op.scale))


def _operator_to_dense(op: GalerkinOperatorLike) -> Array:
    if isinstance(op, BCOO):
        return op.todense()
    if isinstance(op, list):
        return _sum_dense_operators(cast(list[GalerkinOperator], op))
    if isinstance(op, TPMatrices):
        return _sum_dense_operators(cast(list[GalerkinOperator], list(op.tpmats)))
    if isinstance(op, TPMatrix):
        return _tpmatrix_to_dense(op)
    if isinstance(op, TensorMatrix):
        return _operator_to_dense(op.mat)
    return jnp.asarray(op)


def _solve_operator(op: GalerkinOperatorLike, rhs: Array) -> Array:
    mat = _operator_to_dense(op)
    if mat.ndim != 2:
        raise ValueError("Can only solve systems with rank-2 operators")
    b = rhs.reshape((-1,))
    x = jnp.linalg.solve(mat, b)
    return x.reshape(rhs.shape)


def _split_operator_and_forcing(
    form: GalerkinAssembledForm,
) -> tuple[GalerkinOperatorLike | None, Array | None]:
    if form is None:
        return None, None
    if isinstance(form, tuple):
        operator, forcing = cast(tuple[GalerkinOperatorLike, Array | None], form)
        rhs = jnp.asarray(forcing) if forcing is not None else None
        return operator, rhs
    if isinstance(form, list | BCOO | TPMatrix | TensorMatrix | TPMatrices):
        return form, None
    arr = jnp.asarray(form)
    if arr.ndim <= 1:
        return None, arr
    return arr, None


def _assemble_linear_setup(
    expr: sp.Expr, sparse: bool, sparse_tol: int
) -> tuple[GalerkinOperatorLike | None, Array | None, Array | None]:
    if sp.sympify(expr) == 0:
        return None, None, None

    linear_form = cast(
        GalerkinAssembledForm, inner(expr, sparse=sparse, sparse_tol=sparse_tol)
    )
    linear_operator, linear_forcing = _split_operator_and_forcing(linear_form)
    linear_diag = _diag_from_matrix(linear_operator)
    return linear_operator, linear_forcing, linear_diag


def _assemble_mass_setup(
    V: IntegratorSpace, sparse: bool, sparse_tol: int
) -> tuple[GalerkinOperatorLike | None, Array | None]:
    v = TestFunction(V)
    u = TrialFunction(V)
    mass_form = cast(
        GalerkinAssembledForm, inner(v * u, sparse=sparse, sparse_tol=sparse_tol)
    )
    mass_operator, mass_forcing = _split_operator_and_forcing(mass_form)
    if mass_forcing is not None:
        raise NotImplementedError("Mass assembly returned forcing, unsupported for now")
    mass_diag = _diag_from_matrix(mass_operator)
    if mass_diag is not None:
        return None, mass_diag
    return mass_operator, mass_diag


class BaseIntegrator(ABC, nnx.Module):
    def __init__(
        self,
        V: IntegratorSpace,
        equation: sp.Expr,
        u0: sp.Expr | Array | None = None,
        *,
        time: tuple[float, float] | None = None,
        initial: sp.Expr | Array | None = None,
        sparse: bool = False,
        sparse_tol: int = 1000,
    ):
        if initial is None:
            initial = u0
        if initial is None:
            raise ValueError("Initial condition must be provided via `u0` or `initial`")

        self.sparse = sparse
        self.sparse_tol = sparse_tol
        self.time = time
        self.initial = initial
        self.functionspace = V
        t = V.system.base_time()
        _lhs, rhs = split_time_derivative_terms(equation, t)

        test, trial = get_basisfunctions(rhs)
        assert isinstance(test, TestFunction), (
            "Currently only supports TestFunction in weak form"
        )
        assert isinstance(trial, TrialFunction), (
            "Currently only supports TrialFunction in weak form"
        )

        linear, nonlinear = split_linear_nonlinear_terms(-rhs, trial)
        self.has_nonlinear = bool(sp.sympify(nonlinear) != 0)
        self.linear_expr = linear
        nonlinear = sp.expand(remove_test_function(nonlinear, test))
        self.nonlinear_expr = nonlinear

        mass_operator, mass_diag = _assemble_mass_setup(
            V, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        self.mass_operator: GalerkinOperatorLike | None = nnx.data(mass_operator)
        self.mass_diag: Array | None = nnx.data(mass_diag)

        linear_operator, linear_forcing, linear_diag = _assemble_linear_setup(
            self.linear_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )

        self.linear_operator: GalerkinOperatorLike | None = nnx.data(linear_operator)
        self.linear_forcing: Array | None = nnx.data(linear_forcing)
        self.linear_diag: Array | None = nnx.data(linear_diag)

        self._nonlinear_jaxfunction: AppliedUndef | None = None
        self._nonlinear_evaluator: Callable[[Array], Array] | None = None
        if self.has_nonlinear:
            base_jaxfunction = JAXFunction(
                jnp.zeros((V.num_dofs,)), V, name=f"{trial.name}_jax"
            )
            jaxfunction = cast(AppliedUndef, base_jaxfunction.doit())
            nonlinear_expr = replace_trial_with_jaxfunction(
                nonlinear, trial, jaxfunction
            )
            self.nonlinear_expr = nonlinear_expr
            x_ref = V.orthogonal.quad_points_and_weights()[0]
            self._nonlinear_jaxfunction = jaxfunction
            self._nonlinear_evaluator = _compile_nonlinear_evaluator(
                nonlinear_expr, V, x_ref, jaxfunction
            )

    def initial_coefficients(self, initial: sp.Expr | Array | None = None) -> Array:
        init = self.initial if initial is None else initial
        if isinstance(init, sp.Expr):
            return project1D(init, self.functionspace)
        return jnp.asarray(init).reshape(self.functionspace.num_dofs)

    def resolve_time(
        self,
        dt: float,
        steps: int | None = None,
        trange: tuple[float, float] | None = None,
    ) -> tuple[float, float, int]:
        interval = self.time if trange is None else trange
        if interval is None:
            if steps is None:
                raise ValueError("Either `steps` or `trange`/`time` must be provided")
            return 0.0, float(dt * steps), int(steps)

        t0, t1 = float(interval[0]), float(interval[1])
        if steps is None:
            span = t1 - t0
            steps = int(round(span / dt))
        return t0, t1, int(steps)

    def apply_mass(self, uh: Array) -> Array:
        if self.mass_diag is not None:
            return self.mass_diag * uh
        if self.mass_operator is None:
            return uh
        return _apply_operator(self.mass_operator, uh)

    def apply_mass_inverse(self, rhs: Array) -> Array:
        if self.mass_diag is not None:
            return rhs / self.mass_diag
        if self.mass_operator is None:
            return rhs
        return _solve_operator(self.mass_operator, rhs)

    def nonlinear_rhs(self, uh: Array) -> Array:
        if not self.has_nonlinear:
            return jnp.zeros_like(uh)
        assert self._nonlinear_evaluator is not None
        return self.functionspace.forward(self._nonlinear_evaluator(uh))

    def linear_rhs(self, uh: Array) -> Array:
        rhs = jnp.zeros_like(uh)
        if self.linear_operator is not None:
            if self.linear_diag is not None:
                rhs = rhs + self.linear_diag * uh
            else:
                rhs = rhs + _apply_operator(self.linear_operator, uh)
        if self.linear_forcing is not None:
            rhs = rhs + jnp.asarray(self.linear_forcing)
        return self.apply_mass_inverse(rhs)

    @jax.jit(static_argnums=0)
    def total_rhs(self, uh: Array) -> Array:
        return self.linear_rhs(uh) + self.nonlinear_rhs(uh)

    @abstractmethod
    def step(self, u_hat: Array, dt: float) -> Array: ...

    def setup(self, dt: float) -> None: ...

    def solve(
        self,
        dt: float,
        steps: int | None = None,
        u_hat: Array | None = None,
        trange: tuple[float, float] | None = None,
        progress: bool = True,
        n_batches: int = 100,
        return_each_step: bool = False,
    ) -> Array:
        if n_batches <= 0:
            raise ValueError("n_batches must be a positive integer")

        self.setup(dt)
        _, _, n_steps = self.resolve_time(dt, steps=steps, trange=trange)

        if u_hat is None:
            u_hat = self.initial_coefficients()
        if n_steps <= 0:
            if not return_each_step:
                return u_hat
            return jnp.empty((0,) + u_hat.shape, dtype=u_hat.dtype)

        def inner_n_steps(i: int, u_hat: Array) -> Array:
            u_hat = self.step(u_hat, dt)
            return u_hat

        batch_count = min(n_batches, n_steps)

        batch_len = n_steps // batch_count
        remainder = n_steps - batch_count * batch_len
        states: list[Array] = [u_hat]
        diverged = False

        r_batch = range(batch_count)
        iterator = (
            tqdm.tqdm(r_batch, desc="Integrating", unit="step", unit_scale=batch_len)
            if progress
            else r_batch
        )
        for _ in iterator:
            u_hat: Array = jax.lax.fori_loop(0, batch_len, inner_n_steps, u_hat)
            if return_each_step:
                states.append(u_hat)
            if jnp.isnan(u_hat).any() or jnp.isinf(u_hat).any():
                diverged = True
                break

        if remainder and not diverged:
            u_hat: Array = jax.lax.fori_loop(0, remainder, inner_n_steps, u_hat)
            if return_each_step:
                states.append(u_hat)

        if return_each_step:
            if len(states) == 0:
                return jnp.empty((0,) + u_hat.shape, dtype=u_hat.dtype)
            return jnp.stack(states)

        return u_hat
