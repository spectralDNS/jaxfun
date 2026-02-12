from abc import ABC, abstractmethod
from typing import Any, cast

import jax
import jax.numpy as jnp
import sympy as sp
import tqdm
from flax import nnx
from jax.experimental.sparse import BCOO

from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import inner, project1D
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.pinns import FlaxFunction, Residual
from jaxfun.pinns.module import SpectralModule
from jaxfun.typing import Array
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms


def remove_test_function(expr: sp.Expr, test: TestFunction) -> sp.Expr:
    "Replace test functions in expr with 1."
    if expr.has(test):
        return expr.subs(test, 1)
    return expr


def replace_trial_with_flaxfunction(
    expr: sp.Expr, trial: TrialFunction, flax_func: FlaxFunction
) -> sp.Expr:
    "Replace trial function in expr with a FlaxFunction module."
    if expr.has(trial):
        return expr.replace(trial, flax_func)  # ty:ignore[invalid-return-type]
    return expr


def _bcoo_diagonal(mat: BCOO) -> Array | None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return None
    indices = jnp.asarray(mat.indices)
    if indices.shape[1] != 2:
        return None
    if not bool(jnp.all(indices[:, 0] == indices[:, 1])):
        return None
    diag = jnp.zeros(mat.shape[0], dtype=mat.data.dtype)
    return diag.at[indices[:, 0]].add(mat.data)


def _diag_from_matrix(obj: Any) -> Array | None:
    if obj is None:
        return None
    if isinstance(obj, list | tuple):
        diag_sum = None
        for item in obj:
            diag = _diag_from_matrix(item)
            if diag is None:
                return None
            diag_sum = diag if diag_sum is None else diag_sum + diag
        return diag_sum
    if isinstance(obj, BCOO):
        return _bcoo_diagonal(obj)
    if hasattr(obj, "mats"):
        mats = list(obj.mats)
        if len(mats) == 0:
            return None
        diagonals = [_diag_from_matrix(mat) for mat in mats]
        if any(diag is None or diag.ndim != 1 for diag in diagonals):
            return None
        diagonal = cast(Array, diagonals[0])
        for axis, diag in enumerate(diagonals[1:], start=1):
            shape = (1,) * axis + (diag.shape[0],)
            diagonal = diagonal[..., None] * cast(Array, diag).reshape(shape)
        return diagonal
    if hasattr(obj, "mat"):
        return _diag_from_matrix(obj.mat)
    arr = jnp.asarray(obj)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        diag = jnp.diag(arr)
        if bool(jnp.allclose(arr, jnp.diag(diag), atol=1e-12)):
            return diag
    return None


def _apply_operator(op: Any, u: Array) -> Array:
    if op is None:
        return jnp.zeros_like(u)
    if isinstance(op, list):
        out = jnp.zeros_like(u)
        for item in op:
            out = out + _apply_operator(item, u)
        return out
    if hasattr(op, "tpmats"):
        return sum((mat(u) for mat in op.tpmats), jnp.zeros_like(u))
    if callable(op) and not isinstance(op, BCOO):
        return op(u)
    if hasattr(op, "mat"):
        return op.mat @ u
    return op @ u


def _operator_to_dense(op: Any) -> Array:
    if isinstance(op, BCOO):
        return op.todense()
    if isinstance(op, list):
        mats = [_operator_to_dense(item) for item in op]
        return sum(mats[1:], mats[0]) if len(mats) > 0 else jnp.array([])
    if hasattr(op, "tpmats"):
        mats = [_operator_to_dense(item) for item in op.tpmats]
        return sum(mats[1:], mats[0]) if len(mats) > 0 else jnp.array([])
    if hasattr(op, "mats"):
        mats = [_operator_to_dense(item) for item in op.mats]
        if len(mats) == 0:
            return jnp.array([])
        dense = mats[0]
        for mat in mats[1:]:
            dense = jnp.kron(dense, mat)
        return cast(Array, dense * getattr(op, "scale", 1))
    if hasattr(op, "mat"):
        mat = op.mat
        if isinstance(mat, BCOO):
            return mat.todense()
        return jnp.asarray(mat)
    return jnp.asarray(op)


def _solve_operator(op: Any, rhs: Array) -> Array:
    mat = _operator_to_dense(op)
    if mat.ndim != 2:
        raise ValueError("Can only solve systems with rank-2 operators")
    b = rhs.reshape((-1,))
    x = jnp.linalg.solve(mat, b)
    return x.reshape(rhs.shape)


class BaseIntegrator(ABC, nnx.Module):
    def __init__(
        self,
        V: OrthogonalSpace,
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

        self.sparse = bool(sparse)
        self.sparse_tol = int(sparse_tol)
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
        nonlinear = remove_test_function(nonlinear, test)
        flax_func = FlaxFunction(V, name=f"{trial.name}_flax")
        self.nonlinear_expr = replace_trial_with_flaxfunction(
            nonlinear, trial, flax_func
        )
        # uh = project(u0, V)
        self.module = cast(SpectralModule, flax_func.module)
        # self.module.kernel.set_value(uh.reshape(1, -1))

        df = float(V.domain_factor)
        norm_squared = V.norm_squared()
        mass_operator: Any | None = None
        mass_diag: Array | None = None
        if norm_squared is not None:
            mass_diag = jnp.asarray(norm_squared / df)
        else:
            mass_form = inner(
                test * trial,
                sparse=self.sparse,
                sparse_tol=self.sparse_tol,
            )
            if isinstance(mass_form, tuple) and len(mass_form) == 2:
                mass_operator = mass_form[0]
            else:
                mass_operator = mass_form
            mass_diag = _diag_from_matrix(mass_operator)
        self.mass_operator = nnx.data(mass_operator)
        self.mass_diag = nnx.data(mass_diag)

        linear_operator: Any | None = None
        linear_forcing: Any | None = None
        linear_diag: Array | None = None
        if sp.sympify(linear) != 0:
            linear_form = inner(
                self.linear_expr,
                sparse=self.sparse,
                sparse_tol=self.sparse_tol,
            )
            if isinstance(linear_form, tuple) and len(linear_form) == 2:
                linear_operator, linear_forcing = linear_form
            elif (
                isinstance(linear_form, list | BCOO)
                or hasattr(linear_form, "mat")
                or hasattr(linear_form, "mats")
            ):
                linear_operator = linear_form
            else:
                arr = jnp.asarray(linear_form)
                if arr.ndim == 1:
                    linear_operator = None
                    linear_forcing = linear_form
                else:
                    linear_operator = linear_form
            linear_diag = _diag_from_matrix(linear_operator)

        self.linear_operator = nnx.data(linear_operator)
        self.linear_forcing = nnx.data(linear_forcing)
        self.linear_diag = nnx.data(linear_diag)

        points: Array = V.mesh()[:, None]
        self.residual = Residual(self.nonlinear_expr, points)

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

    @nnx.jit
    def nonlinear_rhs(self, uh: Array) -> Array:
        self.module.set_kernel(uh.reshape(1, -1))
        # self.module.kernel = self.module.kernel.at[:].set(uh.reshape(1, -1))
        # self.module.kernel.set_value(uh.reshape(1, -1))
        return self.functionspace.forward(self.residual.evaluate(self.module))

    @nnx.jit
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

    def total_rhs(self, uh: Array) -> Array:
        return self.linear_rhs(uh) + self.nonlinear_rhs(uh)

    @abstractmethod
    def step(self, u_hat: Array, dt: float) -> Array: ...

    @abstractmethod
    def setup(self, dt: float) -> None: ...

    def solve(
        self,
        dt: float,
        steps: int | None = None,
        u_hat: Array | None = None,
        trange: tuple[float, float] | None = None,
        progress: bool = True,
    ) -> Array:
        # if self._dt is None or abs(self._dt - dt) > 1e-12:
        self.setup(dt)
        _, _, nsteps = self.resolve_time(dt, steps=steps, trange=trange)
        if u_hat is None:
            u_hat = self.initial_coefficients()
        if nsteps <= 0:
            return u_hat

        def inner_n_steps(i: int, u_hat: Array) -> Array:
            u_hat = self.step(u_hat, dt)
            return u_hat

        n_batches = min(100, nsteps)
        batch_len = nsteps // n_batches
        remainder = nsteps - n_batches * batch_len
        iterator = (
            tqdm.trange(
                n_batches, desc="Integrating", unit="step", unit_scale=batch_len
            )
            if progress
            else range(n_batches)
        )
        for _ in iterator:
            u_hat = jax.lax.fori_loop(0, batch_len, inner_n_steps, u_hat)
            if jnp.isnan(u_hat).any() or jnp.isinf(u_hat).any():
                break

        if remainder:
            u_hat = jax.lax.fori_loop(0, remainder, inner_n_steps, u_hat)

        return u_hat
