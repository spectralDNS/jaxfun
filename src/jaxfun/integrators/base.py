from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import sympy as sp
import tqdm
from flax import nnx
from jax.experimental.sparse import BCOO
from sympy.core.function import AppliedUndef

from jaxfun.galerkin import Composite, DirectSum, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
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
    split_linear_nonlinear_terms,
    split_time_derivative_terms,
)

from .nonlinear import (
    _compile_nonlinear_evaluator,
    remove_test_function,
    replace_trial_with_jaxfunction,
)

IntegratorSpace = OrthogonalSpace | DirectSum | Composite


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
