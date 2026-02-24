from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx
from jax.flatten_util import ravel_pytree
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxfun.coordinates import BaseScalar, get_system
from jaxfun.galerkin import TestFunction
from jaxfun.galerkin.arguments import ArgumentTag, get_arg
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.typing import Array, Loss_Tuple
from jaxfun.utils import lambdify

from .module import Comp

if TYPE_CHECKING:
    from jaxfun.pinns import FlaxFunction


# Differs from jaxfun.utils.common.jacn in the last if else
def jacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    """Return vectorized k-th order Jacobian of a function.

    Repeatedly applies jacfwd/jacrev k times (producing nested Jacobians) and
    then vmaps over the leading batch axis if k > 0.

    Args:
        fun: Function mapping a scalar/array to an Array.
        k: Number of derivatives (k >= 0).

    Returns:
        Callable producing the k-th order Jacobian for batched inputs.
    """
    for i in range(k):
        fun = jax.jacfwd(fun) if (i % 2 == 0) else jax.jacrev(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun  # type: ignore[return-value]


def get_flaxfunction_args(a: sp.Basic) -> tuple[sp.Symbol | BaseScalar, ...] | None:
    for p in sp.core.traversal.iterargs(a):
        if get_arg(p) is ArgumentTag.JAXFUNC:
            return p.args
    return None


def get_flaxfunctions(
    a: sp.Basic,
) -> set[FlaxFunction]:
    flax_found = set()
    for p in sp.core.traversal.iterargs(a):
        if get_arg(p) is ArgumentTag.JAXFUNC:
            flax_found.add(p)
    return flax_found


def get_testfunction(
    a: sp.Basic,
) -> TestFunction | None:
    for p in sp.core.traversal.iterargs(a):
        if get_arg(p) is ArgumentTag.TEST:
            return p
    return None


def evaluate(expr: sp.Expr, x: Array) -> Array:
    """Evaluate a sympy expression containing FlaxFunctions

    Args:
        expr: Sympy expression containing FlaxFunctions
        x: Points where to evaluate the expression (N, D)

    Returns:
        Array: The evaluated expression at points x
    """
    if len(get_flaxfunctions(expr)) == 0:
        raise ValueError("Expression does not contain any FlaxFunctions")
    if len(x.shape) == 1:
        x = x[:, None]

    assert get_testfunction(expr) is None, "Expression contains TestFunction"

    r = Residual(expr, x)
    return r.evaluate(get_flaxfunctions(expr).pop().module)


def expand(forms: sp.Expr) -> tuple[sp.Expr, list[sp.Expr]]:
    """Expand, and collect all terms without basis functions

    Args:
        forms: Sympy expression

    Returns:
        A tuple, where first item is a sympy expression that does not contain
        flax functions and second item is a list of sp.Exprs containing flax
        functions, used as arguments to Add
    """
    f = sp.Add.make_args(forms.doit().expand())
    consts = []
    flaxs = []
    for fi in f:
        v = get_flaxfunctions(fi)
        if len(v) == 0:
            consts.append(fi)
        else:
            flaxs.append(fi)

    return sp.Add(*consts), flaxs


def _process_input_arrays(
    arrays: tuple[Array | float | None, ...], x: Array
) -> tuple[Array, ...]:
    _newarrays = []
    for array in arrays:
        if isinstance(array, Array):
            if array.shape[0] != x.shape[0]:
                raise ValueError(
                    "Input array length does not match number of collocation points"
                )
            if len(array.shape) == 2 and array.shape[1] == 1:
                array = jnp.squeeze(array)
            if len(array.shape) > 1:
                raise ValueError("Input array must be one-dimensional")
        elif array is None:
            array = jnp.array(1 / x.shape[0], dtype=float)
        elif isinstance(array, Number):
            array = jnp.array(array, dtype=float)
        else:
            raise TypeError("Input target/weight must be an Array, a number, or None")
        _newarrays.append(array)
    return tuple(_newarrays)


class Residual:
    r"""Residual of a single equation

    Regular least squares residual defined at collocation points. The residual
    is computed as

    .. math::
        residual(x_i) = \mathcal{L}(u(x_i)) - b(x_i)

    where :math:`\mathcal{L}` is the differential operator, :math:`u` is the
    unknown solution, and :math:`b` is the target. The target is the part of
    the equation that does not contain the unknown solution. Hence, the target
    needs to be computed only once per collocation point, and is reused for all
    iterations.

    The loss is computed as
    .. math::
        L = \sum_{i=0}^{N-1} (w_i * residual(x_i)^2)

    Attributes:
        x: Collocation points where the residual is evaluated.
        target: Target values for the residual at the collocation points.
        weights: Weights for the residual at the collocation points.
        eqs: Tuple of subequations required to evaluate the residual.
        keys: Set of keys identifying required evaluations of FlaxFunctions.
    """

    x_id: int

    def __init__(
        self,
        f: sp.Expr,
        x: Array,
        target: float | Array = 0,
        weights: float | Array | None = None,
    ) -> None:
        r"""Residual of a single equation evaluated at collocation points

        Args:
            f: Sympy expression representing the equation residual. The
                expression may contain one or several FlaxFunctions.
            x: Collocation points where the residual is evaluated.
            target: Provided target value of the residual. If the expression f
                contains terms without FlaxFunctions, these are automatically
                placed in the target. The target needs to be a number or an array
                of the same length as the collocation points.
            weights: Weights for the residual. The weights need to be a number
                or an array of the same length as the collocation points.
        """
        t_expr, expr = expand(f)
        s = get_flaxfunction_args(f.doit())

        target, weights = _process_input_arrays((target, weights), x)

        # Place all terms without flaxfunctions in the target,
        # because these will not need to be computed more than once
        self.target_expr = t_expr
        self.target0 = target
        self.base_scalars = s
        self.target: Array = self._compute_target(x)

        if weights.sharding != x.sharding and jax.local_device_count() > 1:
            if len(weights.shape) > 0 and weights.shape[0] == x.shape[0]:
                weights = jax.device_put(weights, x.sharding)
            else:
                weights = jax.device_put(weights, NamedSharding(x.sharding.mesh, P()))  # ty:ignore[unresolved-attribute]

        self.x: Array = x
        self.weights: Array = weights

        # Build list of equations and all required evaluations of flaxfunctions
        eqs: list[ResidualFn] = []
        keys = set()
        for h in expr:
            eqs.append(get_fn(h, s))
            v = get_flaxfunctions(h)
            if len(v) == 0:
                continue
            mod_id = hash(v.pop().module)
            for p in sp.core.traversal.iterargs(h):
                if (
                    isinstance(p, sp.Derivative)
                    and get_arg(p.args[0]) is ArgumentTag.JAXFUNC
                ):
                    keys.add((id(x), mod_id, p.derivative_count))
                    continue
                if get_arg(p) is ArgumentTag.JAXFUNC:
                    keys.add((id(x), mod_id, 0))

        self.eqs: tuple[ResidualFn, ...] = tuple(eqs)
        self.keys: set[tuple[int, int, int]] = keys

    def __call__(
        self,
        x: Array,
        target: Array | float | int,
        module: nnx.Module | None,
        Js: dict[tuple[int, int, int], Array] | None = None,
        x_id: int | None = None,
    ) -> Array:
        applied_eqs: list[Array] = [eq(x, module, Js=Js, x_id=x_id) for eq in self.eqs]
        return sum(applied_eqs, start=jnp.array(0)) - target

    def update_arrays(self, x: Array) -> None:
        """Update the collocation points and target values

        Args:
            x: New collocation points where the residual is evaluated.
        """
        self.x = x
        self.target = self._compute_target(x)

    def _compute_target(
        self,
        x: Array,
        weights: Array | None = None,
    ) -> Array:
        t_expr = self.target_expr
        s = self.base_scalars
        if len(t_expr.free_symbols) > 0:
            assert s is not None, "Could not find base scalars in expression"
            t = lambdify(s, t_expr)(*x.T)
        else:
            assert isinstance(t_expr, Number)
            t = float(t_expr) if t_expr.is_real else complex(t_expr)
        t0 = jnp.atleast_1d(self.target0 - t)
        if t0.sharding != x.sharding and jax.local_device_count() > 1:
            if len(t0.shape) > 0 and t0.shape[0] == x.shape[0]:
                t0 = jax.device_put(t0, x.sharding)
            else:
                t0 = jax.device_put(t0, NamedSharding(x.sharding.mesh, P()))  # ty:ignore[unresolved-attribute]
        return t0

    def loss(
        self,
        x: Array,
        target: Array,
        module: nnx.Module,
        Js: dict[tuple[int, int, int], Array] | None = None,
        x_id: int | None = None,
    ) -> Array:
        r = self(x, target, module, Js=Js, x_id=x_id)
        return (self.weights * r**2).sum()

    def _compute_gradients(
        self, module: nnx.Module, x: Array
    ) -> dict[tuple[int, int, int], Array]:
        """Return (as a dictionary) all the required evaluations of module
        for all the residuals' equations"""
        Js = {}
        for key in self.keys:
            x_id, mod_id, k = key
            mod = (
                module.data[module.mod_index[str(mod_id)]]
                if isinstance(module, Comp)
                else module
            )
            Js[key] = jacn(mod, key[2])(x)  # ty:ignore[invalid-argument-type]
        return Js

    def eval_compute_grad(
        self, x: Array, target: Array, module: nnx.Module, x_id: int
    ) -> Array:
        Js = self._compute_gradients(module, x)
        return self(x, target, module, Js=Js, x_id=x_id)

    def loss_compute_grad(
        self, x: Array, target: Array, module: nnx.Module, x_id: int
    ) -> Array:
        Js = self._compute_gradients(module, x)
        return self.loss(x, target, module, Js=Js, x_id=x_id)

    def evaluate(self, module: nnx.Module) -> Array:
        """Evaluate the residual at internal points

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The residuals at points self.x, self.target
        """
        Js = self._compute_gradients(module, self.x)
        return self(self.x, self.target, module, Js=Js)


class ResidualVPINN(Residual):
    r"""Residual of a single equation for VPINNs.

    The residual is computed as an inner product between the equation
    and test functions v_k (`TestFunction`). The equation residual is
    identified as

    .. ::math
        R(x; u) = \mathcal{L}(u(x)) - b(x),

    for operator :mathcal:`L`, target :math:`b(x)`, and solution :math:`u(x)`.
    The loss is then computed as

    .. math::
        L = \sum_{k=0}^{K-1} \left( \int R(x; u) v_k(x) dx \right)^2 / K

    The initialization of this class requires the integrand, here
    :math:`f = R(x; u) v_k(x)`, of the inner product as its first
    argument. This may be the integrand above, or the one obtained after using,
    e.g., integration by parts.

    The inner product is approximated using quadrature at the collocation points

    .. math::
        \phi_k &= \int R(x; u) v_k(x) dx \\
               &\approx \sum_{i=0}^{N-1} w_i R(x_i; u(x_i)) v_k(x_i)

    where :math:`w_i` are the (provided) weights (e.g., quadrature weights).

    The loss is then computed as

    .. math::
        L = \sum_{k=0}^{K-1} \phi_k^2 / K

    Note that the integral may be manipulated using integration by parts. For
    example

    .. math::
        \int u''(x) v_k(x) dx = -\int u'(x) v'_k(x) dx

    if :math:`v_k` vanishes on the boundary. In this case, :math:`f` would
    contain the term :math:`-u'(x) v'_k(x)`.

    Attributes:
        x: Collocation points where the residual is evaluated.
        target: Target values for the residual at the collocation points.
        weights: Weights for the residual at the collocation points.
        eqs: Tuple of subequations required to evaluate the residual.
        keys: Set of keys identifying required evaluations of FlaxFunctions.
        TD: Dictionary mapping derivative counts to evaluations of test functions

    """  # noqa: E501

    def __init__(
        self,
        f: sp.Expr,
        x: Array,
        target: float | Array = 0,
        weights: float | Array | None = None,
    ) -> None:
        r"""Residual of a single equation using VPINN.

        Args:
            f: Sympy expression representing the equation residual. The
                expression may contain one or several FlaxFunctions and a TestFunction.
            x: Collocation points where the residual is evaluated.
            target: Provided target value of the residual. If the expression f
                contains terms without FlaxFunctions, these are automatically
                placed in the target. The target needs to be a number or an array
                of the same length as the collocation points.
            weights: Weights for the residual. The weights need to be a number
                or an array of the same length as the collocation points.
        """  # noqa: E501
        t_expr, expr = expand(f)
        s = get_flaxfunction_args(f.doit())
        test = get_testfunction(f)
        if test is None:
            raise ValueError("No TestFunction found in VPINN residual expression")
        self.V = test.functionspace

        target, weights = _process_input_arrays((target, weights), x)

        # Place all terms without flaxfunctions in the target, since these will only be
        # computed once
        # Place all terms without flaxfunctions in the target,
        # because these will not need to be computed more than once
        self.target_expr = t_expr
        self.target0 = target  # Target provided as array
        self.base_scalars = s

        # t_expr is now a sum of targets, but contains test functions
        # that may or may not have to be differentiated. Process these
        # to separate out the derivative counts.
        t_args = sp.Add.make_args(t_expr)

        def _pop_test_and_get_derivative_count(iterable) -> tuple[sp.Expr, int]:
            test = None
            remains = []
            for z in iterable:
                t_ = get_testfunction(z)
                if t_ is not None:
                    test = z
                else:
                    remains.append(z)
            derivative_count: int = getattr(test, "derivative_count", 0)
            return sp.Mul(*remains), derivative_count

        t_split: list[tuple[sp.Expr | int, int]] = []
        for ta in t_args:
            if isinstance(ta, sp.Mul):
                t_split.append(_pop_test_and_get_derivative_count(ta.args))
            elif get_testfunction(ta) is not None:
                derivative_count: int = getattr(ta, "derivative_count", 0)
                t_split.append((1, derivative_count))
            elif isinstance(ta, Number):
                t_split.append((ta, 0))
            else:
                raise ValueError("Could not parse target expression")
        self.t_split = t_split

        # Add all terms with the same test function derivative count
        t_v = {}
        for tn, tv in t_split:
            if tv not in t_v:
                t_v[tv] = []
            t_v[tv].append(tn)
        for tv, tn in t_v.items():
            t_v[tv] = sp.Add(*tn)
        self.target_dict: dict[int, sp.Expr] = t_v

        # Build list of equations and all required evaluations of flaxfunctions
        eqs: list[tuple[ResidualFn, int]] = []
        keys: set[tuple[int, int, int]] = set()
        for h in expr:
            assert isinstance(h, sp.Mul)
            hn, hi = _pop_test_and_get_derivative_count(h.args)
            eqs.append((get_fn(hn, s), hi))
            v = get_flaxfunctions(hn)
            assert len(v) > 0, "No FlaxFunctions found in equation term"
            mod_id = hash(v.pop().module)
            for p in sp.core.traversal.iterargs(hn):
                if (
                    isinstance(p, sp.Derivative)
                    and get_arg(p.args[0]) is ArgumentTag.JAXFUNC
                ):
                    keys.add((id(x), mod_id, p.derivative_count))
                    continue
                if get_arg(p) is ArgumentTag.JAXFUNC:
                    keys.add((id(x), mod_id, 0))
        self.eqs: tuple[tuple[ResidualFn, int], ...] = tuple(eqs)
        self.keys: set[tuple[int, int, int]] = keys

        if weights.sharding != x.sharding and jax.local_device_count() > 1:
            if len(weights.shape) > 0 and weights.shape[0] == x.shape[0]:
                weights = jax.device_put(weights, x.sharding)
            else:
                weights = jax.device_put(weights, NamedSharding(x.sharding.mesh, P()))  # ty:ignore[unresolved-attribute]

        # Compute both the test functions (all derivatives needed)
        # and the target stored in target_dict.
        self.TD = self._compute_test_function(x)
        self.target = self._compute_target(x, weights)
        self.x, self.weights = x, weights

    def _compute_test_function(self, x: Array) -> dict[int, Array]:
        # The test functions should be evaluated once per derivative count
        # FIXME: only implemented for 1D currently
        assert isinstance(self.V, OrthogonalSpace)
        TD = {
            k: self.V.evaluate_basis_derivative(self.V.map_reference_domain(x[:, 0]), k)
            for k in self.target_dict
        }
        return TD

    def _compute_target(self, x: Array, weights: Array | None = None) -> Array:
        s = self.base_scalars
        assert weights is not None, "Weights must be provided"
        t0 = jnp.expand_dims(jnp.atleast_1d(self.target0), axis=-1)
        tns = []
        for tv, tn in self.target_dict.items():
            if len(tn.free_symbols) > 0:
                assert s is not None, "Could not find base scalars in expression"
                tx = lambdify(s, tn)(*x.T)
                tns.append(self.TD[tv] * tx[:, None])
            else:
                assert isinstance(tn, Number)
                tn = float(tn) if tn.is_real else complex(tn)
                tns.append(self.TD[tv] * tn)

        t0 = t0 - jnp.array(tns, dtype=float).sum(axis=0)
        if t0.sharding != x.sharding and jax.local_device_count() > 1:
            if len(t0.shape) > 0 and t0.shape[0] == x.shape[0]:
                t0 = jax.device_put(t0, x.sharding)
            else:
                t0 = jax.device_put(t0, NamedSharding(x.sharding.mesh, P()))  # ty:ignore[unresolved-attribute]
        return t0 * weights[:, None]

    def loss(
        self,
        x: Array,
        target: Array | float | int,
        module: nnx.Module | None,
        Js: dict[tuple[int, int, int], Array] | None = None,
        x_id: int | None = None,
    ) -> Array:
        target = jnp.atleast_1d(jnp.array(target))
        # rk = self(x, target, module, Js=Js, x_id=x_id)
        # return (rk.sum(axis=0)**2).mean() # slower
        return (
            jnp.array(
                sum(
                    [
                        (
                            self.TD[k].T
                            @ (self.weights * eq(x, module, Js=Js, x_id=x_id))
                        )
                        for eq, k in self.eqs
                    ]
                )
                - target.sum(axis=0)
            )
            ** 2
        ).mean()

    def __call__(
        self,
        x: Array,
        target: Array | float | int,
        module: nnx.Module | None,
        Js: dict[tuple[int, int, int], Array] | None = None,
        x_id: int | None = None,
    ) -> Array:
        target = jnp.atleast_1d(jnp.array(target))
        # Return N x K array of residuals for all points N and test functions K
        return jnp.array(
            sum(
                [
                    (
                        self.TD[k]
                        * (self.weights * eq(x, module, Js=Js, x_id=x_id))[:, None]
                    )
                    for eq, k in self.eqs
                ]
            )
            - target
        )


def process_input(*fs, residuals=None) -> tuple[Residual | ResidualVPINN, ...]:
    from jaxfun.operators import Dot

    if residuals is None:
        residuals = []
    for f in fs:
        f0 = f[0].doit()
        f1 = f[1]
        res = ResidualVPINN if get_testfunction(f0) is not None else Residual
        if f0.is_Vector:  # Vector equation
            sys = get_system(f0)
            for i in range(sys.dims):
                bt = sys.get_contravariant_basis_vector(i)
                g = (Dot(f0, bt),) + (f1,)
                if len(f) > 2:
                    if isinstance(f[2], Number):
                        g += (f[2],)
                    else:
                        g += (f[2][..., i],)
                if len(f) > 3:
                    g += (f[3],)

                residuals.append(res(*g))

        else:
            residuals.append(res(*((f[0], f1) + f[2:])))
    return tuple(residuals)


class Loss:
    r"""Loss function

    Computes the total loss over several equations, all defined at their
    own collocation points. The collocation points need to be arrays of
    shape (N, D), where N is the number of points and D is the number of
    dimensions. The collocation points need to be fully addressable.

    The loss is defined as the sum of the losses for all provided subproblems.
    The subproblems are defined by tuples containing

    1. The equation residual (sympy expression)
    2. The collocation points (Array of shape (N, D))
    3. (optional) The target (Number or Array of shape (N,))
    4. (optional) The weights (Number or Array of shape (N,))

    The residual of each subproblem is understood as:

    .. math::
        R(x; u) = \mathcal{L}(u(x)) - b(x),

    where \mathcal{L} is the differential operator, u is the unknown solution,
    and b is the target. The target is the part of the subproblem that does not
    contain the unknown solution. This target may be part of the provided equation,
    or provided as an explicit array under 3.

    The weights are used to weight the contribution at each collocation point.
    For a regular least squares method the losses are computed for each
    subproblem j as:

    .. math::
        L_j = sum_{i=0}^{N-1} (w_i * R(x_i; u(x_i))^2)

    where :math:`w_i` are the weights for each collocation point :math:`x_i`.

    If the subproblem contains a test function :math:`v_k`, then the loss is
    interpreted as an inner product, computed as

    .. math::
        \phi_k &= \int R(x; u) v_k(x) dx \\
        \phi_k &\approx sum_{i=0}^{N-1} (w_i * R(x_i, u(x_i)) v_k(x_i))

    and subsequently the loss for each subproblem j is

    .. math::
        L_j = sum_{k=0}^{K-1} \phi_k^2 / K

    All subproblems are then finally combined into a total loss as

    .. math::
        L = sum_j gw_j * L_j,

    where :math:`gw_j` are global weights that can be set in the Trainer, or
    computed automatically to achieve a balanced contribution from each
    subproblem.

    For all losses, sums (not means) are used in the inner computations. Hence
    an appropriate weight for a least squares loss is 1 / N, where N
    is the number of collocation points. If weights is None, which is default,
    then the weights are automatically set to 1 / N.

    For VPINN, the weights represent quadrature weights for inner products, and
    the weights should be chosen accordingly.

    """

    def __init__(self, *fs: Loss_Tuple) -> None:
        r"""Computes the total loss over all input equations at all collocation
        points.

        Args:
            fs:
                One or several tuples. The tuples contain the subproblems that
                are to be solved. The subproblem tuples are defined by items:

                    1. The equation residual (sympy expression)
                    2. The collocation points (Array of shape (N, D))
                    3. (optional) The target (Number or Array of shape (N,))
                    4. (optional) The weights (Number or Array of shape (N,))

                For VPINN the equation residual needs to contain a `TestFunction`,
                and the equation residual is the integrand of the inner product,
                except the weight.

        Examples:

            >>> import jax.numpy as jnp
            >>> from jaxfun.operators import Div, Grad
            >>> from jaxfun.pinns.loss import Loss
            >>> from jaxfun.pinns.module import MLPSpace, FlaxFunction
            >>> V = MLPSpace([8, 8], dims=1, rank=0, name="V")
            >>> u = FlaxFunction(V, name="u")
            >>> eq = Div(Grad(u)) + 2
            >>> xj = jnp.linspace(-1, 1, 10)[:, None]
            >>> xb = jnp.array([[-1.0], [1.0]])
            >>> loss_fn = Loss((eq, xj), (u, xb, 0, 10))
        """

        self.residuals: tuple[Residual, ...] = process_input(*fs, residuals=[])

        # Store the unique collocation points and their order for later use
        self.xs: dict[int, Array] = {id(eq.x): eq.x for eq in self.residuals}
        x_keys = list(self.xs.keys())
        self.x_ids = tuple(x_keys.index(id(eq.x)) for eq in self.residuals)
        # use indices into xs (0, 1, ...) as keys instead of id(x)
        for i, eq in enumerate(self.residuals):
            eq.keys = set([(self.x_ids[i], mod_id, k) for (_, mod_id, k) in eq.keys])
            eq.x_id = self.x_ids[i]

        # Store all keys needed for gradient computations
        self.keys = set(key for i, eq in enumerate(self.residuals) for key in eq.keys)

    @property
    def args(self) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        targets = tuple(eq.target for eq in self.residuals)
        return tuple(self.xs.values()), targets

    @abstractmethod
    def update_time(self, module: nnx.Module, march: Array) -> None: ...

    @property
    def local_mesh(self) -> jax.sharding.Mesh | None:
        if jax.local_device_count() == 1:
            return None

        if self.residuals[0].x.sharding.num_devices != jax.local_device_count():
            raise ValueError(
                "Cannot determine local mesh, collocation points are not sharded "
                "over all local devices"
            )

        # TODO: Does `x.sharding` actually have the attribute `mesh`?
        return self.residuals[0].x.sharding.mesh  # ty:ignore[unresolved-attribute]

    def _compute_residual_arrays(
        self, module: nnx.Module, xs: tuple[Array, ...], targets: tuple[Array, ...]
    ) -> list[Array]:
        """Return the residuals for all collocation points

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The residuals for all collocation points
        """
        Js = self._compute_gradients(module, xs)
        return [
            eq(xs[x_id], target, module, Js=Js, x_id=x_id)
            for i, (eq, target, x_id) in enumerate(
                zip(self.residuals, targets, self.x_ids, strict=True)
            )
        ]

    def JTJ(
        self,
        module: nnx.Module,
        gw: Array,
        xs: tuple[Array, ...],
        targets: tuple[Array, ...],
    ) -> Array:
        """Return the Gauss-Newton approximation to the Hessian

        For the loss L = sum_i gw[i] * mean( res_i^2 ), the Gauss-Newton
        approximation to the Hessian is given by

            H = 2 * sum_i gw[i] * (J_i*w)^T @ J_i / N_i

        where J_i is the Jacobian (J_i)_kj = ∂res_i(x^i_k) / ∂p_j (weights p_j) of
        the residuals for equation i evaluated at all collocation points
        (x^i = (x^i_k)_{k=0}^{N_i-1}) for equation i, and N_i = |x^i| is the number
        of collocation points for equation i. The weights w are optional weights for
        the residuals.

        Args:
            module: The module (nnx.Module)
            gw: Global weights
            xs: All the collocation arrays used for all equations
            targets: Targets for all residuals"""
        res = jax.jacfwd(self._compute_residual_arrays, argnums=0)(module, xs, targets)
        JTJ = []
        for i, r in enumerate(res):
            jf = ravel_pytree(r)[0]
            N = xs[self.x_ids[i]].shape[0]
            J = jf.reshape((N, -1))
            w = gw[i] * self.residuals[i].weights
            if len(w.shape) == 0:
                w = jnp.array([w])
            JTJ.append(((J * w[:, None]).T @ J) / N)
        return 2 * sum(JTJ)

    @nnx.jit(static_argnums=0)
    def value_and_grad_and_JTJ(
        self,
        module: nnx.Module,
        gw: Array,
        xs: tuple[Array, ...],
        targets: tuple[Array, ...],
    ) -> tuple[Array, Array, Array]:
        """Return the loss value, gradient and JTJ

        Args:
            module: The module (nnx.Module)
            gw: Global weights
            xs: All the collocation arrays used for all equations
            targets: Targets for all residuals

        Returns:
            A tuple where the first item is the loss value and the second item
            is the gradient and the third is the Gauss-Newton approximation to the
            Hessian
        """
        res = self._compute_residual_arrays(module, xs, targets)
        dres = jax.jacfwd(self._compute_residual_arrays, argnums=0)(module, xs, targets)
        unravel = ravel_pytree(nnx.state(module))[1]
        JTJ = []
        grads = []
        loss = 0
        for i, (r, dr) in enumerate(zip(res, dres, strict=True)):
            jf = ravel_pytree(dr)[0]
            N = xs[self.x_ids[i]].shape[0]
            J = jf.reshape((N, -1))
            w = gw[i] * self.residuals[i].weights
            if len(w.shape) == 0:
                w = jnp.array([w])
            JTJ.append(((J * w[:, None]).T @ J) / N)
            rw = r * w
            grads.append(
                ravel_pytree(
                    jax.tree.map(
                        lambda g, rw=rw: jax.lax.dot_general(
                            rw, g, (((0,), (0,)), ((), ()))
                        ),
                        dr,
                    )
                )[0]
                / N
            )
            loss = loss + rw @ r / N

        return loss, unravel(2 * sum(grads)), 2 * sum(JTJ)  # type: ignore[return-value]

    def compute_residual_i(self, module: nnx.Module, i: int) -> Array:
        """Return the residuals for equation i

        Args:
            module: The module (nnx.Module)
            i: The equation number

        Returns:
            Array: The residuals for equation i
        """
        xs, targets = self.args
        return self.residuals[i].eval_compute_grad(
            xs[self.x_ids[i]], targets[i], module, x_id=self.x_ids[i]
        )

    def compute_residuals(self, module: nnx.Module) -> Array:
        """Return the residuals for all equations

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The residuals for all equations (not including global weights)
        """
        xs, targets = self.args
        L2 = []
        Js = self._compute_gradients(module, xs)
        for eq, target, x_id in zip(self.residuals, targets, self.x_ids, strict=True):
            L2.append(eq.loss(xs[x_id], target, module, Js=Js, x_id=x_id))
        return jnp.array(L2)

    def loss_i(
        self,
        module: nnx.Module,
        xs: tuple[Array, ...],
        targets: tuple[Array, ...],
        i: int,
    ) -> Array:
        """Return the loss for equation i

        Does not include the global weight.

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)
            i: The equation number

        Returns:
            Array: The loss for equation i
        """
        return self.residuals[i].loss_compute_grad(
            xs[self.x_ids[i]], targets[i], module, x_id=self.x_ids[i]
        )

    def norm_grad_loss_i(
        self,
        module: nnx.Module,
        xs: tuple[Array, ...],
        targets: tuple[Array, ...],
        i: int,
    ) -> Array:
        """Return the norm of the gradient of the loss for equation i

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)
            i: The equation number

        Returns:
            Array: The norm of the gradient of the loss for equation i
        """
        return jnp.linalg.norm(
            ravel_pytree(nnx.grad(self.loss_i)(module, xs, targets, i))[0]
        )

    def norm_grad_loss(
        self, module: nnx.Module, xs: tuple[Array, ...], targets: tuple[Array, ...]
    ) -> Array:
        """Return the norms of the gradients of the losses for all equations

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The norms of the gradients of the losses for all equations
        """
        norms = []
        for i in range(len(self.residuals)):
            norms.append(self.norm_grad_loss_i(module, xs, targets, i))
        return jnp.array(norms)

    def compute_global_weights(
        self, module: nnx.Module, xs: tuple[Array, ...], targets: tuple[Array, ...]
    ) -> Array:
        """Return global weights based on the norms of the gradients

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The global weights
        """
        norms = self.norm_grad_loss(module, xs, targets)
        return jnp.sum(norms) / jnp.where(norms < 1e-16, 1e-16, norms)

    @nnx.jit(static_argnums=0)
    def update_global_weights(
        self,
        module: nnx.Module,
        gw: Array,
        alpha: float,
        xs: tuple[Array, ...],
        targets: tuple[Array, ...],
    ) -> Array:
        """Return updated global weights

        Args:
            module: The module (nnx.Module)
            gw: Current global weights
            alpha: Smoothing parameter (0 < alpha < 1)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The updated global weights
        """
        from jax.experimental import multihost_utils as mh

        new = self.compute_global_weights(module, xs, targets)
        if jax.process_count() > 1:
            new = mh.process_allgather(new).mean(0)
        return new * (1 - alpha) + gw * alpha

    def _compute_gradients(
        self, module: nnx.Module, xs: tuple[Array, ...]
    ) -> dict[tuple[int, int, int], Array]:
        """Return (as a dictionary) all the required evaluations of module
        for the loss function"""
        Js = {}
        for key in self.keys:
            x_id, mod_id, k = key
            mod = (
                module.data[module.mod_index[str(mod_id)]]
                if isinstance(module, Comp)
                else module
            )
            Js[key] = jacn(mod, k)(xs[x_id])  # ty:ignore[invalid-argument-type]
        return Js

    def loss_with_gw(
        self,
        module: nnx.Module,
        gw: Array,
        xs: tuple[Array, ...],
        targets: tuple[Array, ...],
    ) -> Array:
        """Return the weighted loss with given global weights

        Args:
            module: The module (nnx.Module)
            gw: Global weights
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The weighted loss
        """
        Js = self._compute_gradients(module, xs)
        return (
            jnp.array(
                [
                    eq.loss(xs[x_id], target, module, Js=Js, x_id=x_id)
                    for i, (eq, target, x_id) in enumerate(
                        zip(self.residuals, targets, self.x_ids, strict=True)
                    )
                ]
            )
            @ gw
        )

    def __call__(self, module: nnx.Module) -> Array:
        """Return the total least squares loss

        Does not include global weights.

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The total (unweighted) least squares loss
        """
        xs, targets = self.args
        Js = self._compute_gradients(module, xs)
        return sum(
            [
                eq.loss(xs[x_id], target, module, Js=Js, x_id=x_id)
                for i, (eq, target, x_id) in enumerate(
                    zip(self.residuals, targets, self.x_ids, strict=True)
                )
            ]
        )


class ResidualFn(Protocol):
    def __call__(
        self,
        x: Array,
        mod: nnx.Module | None = None,
        Js: dict[tuple[int, int, int], Array] | None = None,
        x_id: int | None = None,
    ) -> Array: ...


def get_fn(f: sp.Expr, s: tuple[BaseScalar | sp.Symbol, ...] | None) -> ResidualFn:
    """Return Sympy Expr as function evaluated by points and gradients

    Args:
        f: Sympy expression that may contain FlaxFunction
        s: Tuple of base scalars used as variables in f

    Returns:
        Callable[[Array, nnx.Module, dict[tuple[int, int, int], Array], int], Array]
    """

    v = get_flaxfunctions(f)

    if len(v) == 0:
        # Coefficient independent of basis function
        if len(f.free_symbols) > 0:
            fun = lambdify(s, f)
            return lambda x, mod=None, Js=None, x_id=None, fun=fun: fun(*x.T)
        else:
            arr = jnp.array(float(f) if f.is_real else complex(f))
            return lambda x, mod=None, Js=None, x_id=None, arr=arr: arr

    if isinstance(f, sp.Mul):
        # Multiplication of terms that either contain the basis function or not
        f_flax: list[sp.Expr] = []
        f_const: list[sp.Expr] = []
        for fx in f.args:
            v0 = get_flaxfunctions(fx)
            # Collect terms that do not (f_const) and contain (f_flax) the flaxfunction
            if len(v0) == 0:
                f_const.append(fx)
                continue
            f_flax.append(fx)

        f_const: sp.Expr = sp.Mul(*f_const)
        fun = get_fn(f_const, s)

        def res_fun(
            x: Array,
            mod: nnx.Module | None = None,
            Js: dict[tuple[int, int, int], Array] | None = None,
            x_id: int | None = None,
        ) -> Array:
            return fun(x, mod, Js, x_id) * jnp.prod(
                jnp.array([get_fn(fi, s)(x, mod, Js, x_id) for fi in f_flax]), axis=0
            )

        return res_fun

    elif isinstance(f, sp.Pow):
        f0: sp.Expr = f.args[0]
        p0: int = int(f.args[1])

        def res_fun(
            x: Array,
            mod: nnx.Module | None = None,
            Js: dict[tuple[int, int, int], Array] | None = None,
            x_id: int | None = None,
        ) -> Array:
            return get_fn(f0, s)(x, mod, Js, x_id) ** p0

        return res_fun

    assert len(v) == 1
    v = v.pop()
    return partial(
        _lookup_or_eval,
        mod_id=hash(v.module),
        global_index=v.global_index,  # ty:ignore[unresolved-attribute]
        k=int(getattr(f, "derivative_count", "0")),
        variables=getattr(f, "variables", ()),
    )


def _lookup_or_eval(
    x: Array,
    mod: nnx.Module,
    Js: dict[tuple[int, int, int], Array] | None = None,
    x_id: int = 0,
    mod_id: int = 0,
    global_index: int = 0,
    k: int = 1,
    variables: tuple[BaseScalar, ...] = (),
) -> Array:
    """Compute or look up k-th order Jacobian of module at points x"""
    if Js is None:
        Js = {}
    module = mod.data[mod.mod_index[str(mod_id)]] if isinstance(mod, Comp) else mod
    assert mod_id == hash(module), (mod_id, hash(module), module)
    var = tuple((slice(None), global_index)) + tuple(int(s._id[0]) for s in variables)
    key: tuple[int, int, int] = (x_id, mod_id, k)
    if key not in Js:
        # Compute gradient
        z = jacn(module, k)(x)  # ty:ignore[invalid-argument-type]
        return z[var]
    # look up gradient
    return Js[key][var]


class TimeMarchingLoss(Loss):
    r"""Loss function for time-marching problems

    Extends the Loss class to handle time-marching problems. The initial
    condition is treated specially, and only evaluated at the initial
    time step.

    Attributes:
        initial_conditions: Index of the subproblem that represents the initial
            condition. This subproblem will be treated specially in case of a
            marching problem.
    """

    def __init__(self, *fs: Loss_Tuple, initial_conditions: int = 1) -> None:
        r"""Computes the total loss over all input equations at all collocation
        points, with special treatment of the initial condition.

        Args:
            fs:
                One or several tuples. The tuples contain the subproblems that
                are to be solved. The subproblem tuples are defined by items:

                    1. The equation residual (sympy expression)
                    2. The collocation points (Array of shape (N, D))
                    3. (optional) The target (Number or Array of shape (N,))
                    4. (optional) The weights (Number or Array of shape (N,))

                For VPINN the equation residual needs to contain a `TestFunction`,
                and the equation residual is the integrand of the inner product,
                except the weight.
            initial_conditions: The number of initial conditions for the problem.
                For example, the wave equation requires two, whereas the heat equation
                requires one. The initial conditions must be the last subproblems in
                the input fs.

        """
        self.initial_conditions = initial_conditions
        super().__init__(*fs)

    def update_time(self, module: nnx.Module, march: Array) -> None:
        """Update the collocation points and targets used in the residuals

        Args:
            module: The module (nnx.Module)
            march: Array to be added to all collocation points
        """
        xs = tuple(self.xs.values())
        if xs[0].shape[-1] != march.shape[-1]:
            raise ValueError("Cannot update collocation points, dimension mismatch")

        for ic in range(
            len(self.residuals) - self.initial_conditions, len(self.residuals)
        ):
            init_res = self.residuals[ic]
            # Compute the initial value at the new time.
            init_res.target0 = init_res(init_res.x + march, 0, module)
            # The only term in the target independent of the solution is the
            # initial condition. Set to zero.
            init_res.target_expr = sp.S.Zero
        for eq, x_id in zip(self.residuals, self.x_ids, strict=True):
            eq.update_arrays(xs[x_id] + march)
        self.xs = {id(eq.x): eq.x for eq in self.residuals}
