"""Symbolic utilities for splitting transient weak forms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sympy as sp

if TYPE_CHECKING:
    from jaxfun.galerkin import TrialFunction

# Operators treated as linear in the dependent field.
_LINEAR_UNARY = {"Grad", "Div", "Curl", "Derivative"}
_LINEAR_BINARY = {"Dot", "Cross", "Outer"}
_TIME_INDEPENDENT_TRIALS: dict[TrialFunction, TrialFunction] = {}


def _as_expr(node: sp.Basic | sp.Expr) -> sp.Expr:
    """Coerce a SymPy node to ``Expr`` for typed expression helpers."""
    return cast(sp.Expr, sp.sympify(node))


def get_time_independent(u: TrialFunction) -> TrialFunction:
    """Return the cached time-independent counterpart of a transient trial field."""
    if not u.transient:
        return u
    cached = _TIME_INDEPENDENT_TRIALS.get(u)
    if cached is not None:
        return cached
    V = u.functionspace
    name = u.name
    out = u.func(V, name, transient=False)
    _TIME_INDEPENDENT_TRIALS[u] = out
    return out


def drop_time_argument(expr: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """Replace transient TrialFunctions in ``expr`` by time-independent ones."""
    from jaxfun.galerkin.arguments import TrialFunction

    expr = sp.sympify(expr)
    repl = {
        fapp: get_time_independent(fapp)
        for fapp in expr.atoms(TrialFunction)
        if t in fapp.args
    }
    return expr.xreplace(repl)


def time_derivative_as_operator(
    expr: sp.Expr, dependent: TrialFunction, time_symbol: sp.Symbol
) -> sp.Expr:
    """Convert first-order time derivatives into a linear operator on ``dependent``.

    For example ``a(x) * v * d/dt(u(x, t))`` becomes ``a(x) * v * u(x)`` and
    ``v * d/dt(du/dx)`` becomes ``v * du/dx``.
    """
    dependent_ind = get_time_independent(dependent)

    def replace_time_derivative(node: sp.Basic) -> sp.Basic:
        assert isinstance(node, sp.Derivative)

        time_count = node.variables.count(time_symbol)
        if time_count != 1:
            raise ValueError(
                "Integrators only support first-order time derivatives in the weak form"
            )
        if not node.expr.has(dependent):
            raise ValueError(
                "Time-derivative terms must act on the transient trial function"
            )

        expr_without_time = drop_time_argument(node.expr, time_symbol)
        remaining_variables = tuple(var for var in node.variables if var != time_symbol)
        if len(remaining_variables) == 0:
            return expr_without_time
        return sp.Derivative(expr_without_time, *remaining_variables)

    transformed = sp.sympify(expr).replace(
        lambda node: isinstance(node, sp.Derivative) and time_symbol in node.variables,
        replace_time_derivative,
    )
    transformed_expr = cast(sp.Expr, transformed)
    transformed = sp.expand(drop_time_argument(transformed_expr, time_symbol))
    linear, nonlinear = split_linear_nonlinear_terms(transformed, dependent_ind)
    if sp.sympify(nonlinear) != 0:
        raise ValueError(
            "Time-derivative terms must be linear in the transient trial function"
        )
    return linear


def split_time_derivative_terms(
    expr: sp.Expr, time_symbol: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """Split an expression into time-derivative terms and the remainder.

    Returns:
        A tuple ``(lhs, rhs)`` where ``lhs`` contains all additive terms with a
        derivative in ``time_symbol`` and ``rhs`` contains the remaining terms
        with transient TrialFunctions replaced by their time-independent forms.
    """
    expr = sp.expand(expr)

    time_terms: list[sp.Expr] = []
    other_terms: list[sp.Expr] = []

    for term in sp.Add.make_args(expr):
        has_time_derivative = any(
            time_symbol in d.variables for d in term.atoms(sp.Derivative)
        )
        (time_terms if has_time_derivative else other_terms).append(term)

    lhs = sp.Add(*time_terms)
    rhs = drop_time_argument(sp.Add(*other_terms), time_symbol)

    return lhs, rhs


def split_linear_nonlinear_terms(
    expr: sp.Expr, dependent: TrialFunction | sp.Function
) -> tuple[sp.Expr, sp.Expr]:
    """Split an expression into linear and nonlinear parts in ``dependent``.

    A term is classified as linear if it contains at most one dependent factor
    and the dependent does not appear inside nonlinear functions. Common linear
    operators (Grad/Div/Curl/Derivative and Dot/Cross/Outer with a single
    dependent operand) are treated as linear.
    """
    from jaxfun.galerkin.arguments import TrialFunction

    if isinstance(dependent, TrialFunction):
        dependent = get_time_independent(dependent)
    expr = sp.expand(expr)

    linear_terms: list[sp.Expr] = []
    nonlinear_terms: list[sp.Expr] = []

    for term in sp.Add.make_args(expr):
        linear, nonlinear = split_linear_nonlinear_node(term, dependent)
        if sp.sympify(linear) != 0:
            linear_terms.append(linear)
        if sp.sympify(nonlinear) != 0:
            nonlinear_terms.append(nonlinear)

    return sp.expand(sp.Add(*linear_terms)), sp.expand(sp.Add(*nonlinear_terms))


def split_linear_nonlinear_node(
    node: sp.Basic, dependent: TrialFunction | sp.Function
) -> tuple[sp.Expr, sp.Expr]:
    """Split a single node into parts linear and nonlinear in ``dependent``."""
    node = sp.sympify(node)
    node_expr = _as_expr(node)

    if not node.has(dependent):
        return node_expr, sp.Integer(0)
    if is_linear_in_dependent(node, dependent):
        return node_expr, sp.Integer(0)

    if isinstance(node, sp.Add):
        linear_terms: list[sp.Expr] = []
        nonlinear_terms: list[sp.Expr] = []
        for arg in node.args:
            linear, nonlinear = split_linear_nonlinear_node(arg, dependent)
            if sp.sympify(linear) != 0:
                linear_terms.append(linear)
            if sp.sympify(nonlinear) != 0:
                nonlinear_terms.append(nonlinear)
        return sp.Add(*linear_terms), sp.Add(*nonlinear_terms)

    if isinstance(node, sp.Derivative):
        linear, nonlinear = split_linear_nonlinear_node(sp.expand(node.expr), dependent)
        return _apply_linear_unary_operator(node, linear), _apply_linear_unary_operator(
            node, nonlinear
        )

    if isinstance(node, sp.Mul):
        dependent_factors = [arg for arg in node.args if arg.has(dependent)]
        if len(dependent_factors) == 1:
            dependent_factor = _as_expr(dependent_factors[0])
            static_factor = _as_expr(
                sp.Mul(*(arg for arg in node.args if arg is not dependent_factor))
            )
            linear, nonlinear = split_linear_nonlinear_node(
                sp.expand(dependent_factor), dependent
            )
            return (
                sp.expand(static_factor * linear),
                sp.expand(static_factor * nonlinear),
            )
        return sp.Integer(0), sp.sympify(node)

    if len(node.args) == 1 and node.func.__name__ in _LINEAR_UNARY:
        linear, nonlinear = split_linear_nonlinear_node(
            sp.expand(node.args[0]), dependent
        )
        return _apply_linear_unary_operator(node, linear), _apply_linear_unary_operator(
            node, nonlinear
        )

    if len(node.args) == 2 and node.func.__name__ in _LINEAR_BINARY:
        a0, a1 = (_as_expr(arg) for arg in node.args)
        has0 = a0.has(dependent)
        has1 = a1.has(dependent)
        if has0 and has1:
            return sp.Integer(0), node_expr
        if has0:
            linear, nonlinear = split_linear_nonlinear_node(sp.expand(a0), dependent)
            return _apply_linear_binary_operator(
                node, linear, a1
            ), _apply_linear_binary_operator(node, nonlinear, a1)
        if has1:
            linear, nonlinear = split_linear_nonlinear_node(sp.expand(a1), dependent)
            return _apply_linear_binary_operator(
                node, a0, linear
            ), _apply_linear_binary_operator(node, a0, nonlinear)

    return sp.Integer(0), node_expr


def _apply_linear_unary_operator(node: sp.Basic, arg: sp.Expr) -> sp.Expr:
    """Apply a linear unary operator to a split subexpression."""
    arg = sp.sympify(arg)
    if arg == 0:
        return sp.Integer(0)
    if isinstance(arg, sp.Add):
        return sp.expand(
            sp.Add(*(_apply_linear_unary_operator(node, term) for term in arg.args))
        )
    if isinstance(node, sp.Derivative):
        return sp.expand(node.func(arg, *node.variables))
    if node.func.__name__ == "Grad":
        return sp.expand(node.func(arg, transpose=getattr(node, "_transpose", False)))
    return sp.expand(node.func(arg))


def _apply_linear_binary_operator(
    node: sp.Basic, left: sp.Expr, right: sp.Expr
) -> sp.Expr:
    """Apply a linear binary operator to a split subexpression."""
    left = sp.sympify(left)
    right = sp.sympify(right)
    if left == 0 or right == 0:
        return sp.Integer(0)
    if isinstance(left, sp.Add):
        return sp.expand(
            sp.Add(
                *(
                    _apply_linear_binary_operator(node, term, right)
                    for term in left.args
                )
            )
        )
    if isinstance(right, sp.Add):
        return sp.expand(
            sp.Add(
                *(
                    _apply_linear_binary_operator(node, left, term)
                    for term in right.args
                )
            )
        )
    return sp.expand(node.func(left, right))


def is_linear_add(node: sp.Add, dependent: TrialFunction | sp.Function) -> bool:
    """Return True when every additive term is linear in ``dependent``."""
    return all(is_linear_in_dependent(arg, dependent) for arg in node.args)


def is_linear_mul(node: sp.Mul, dependent: TrialFunction | sp.Function) -> bool:
    """Return True when a product contains at most one dependent factor."""
    dependent_factors = [a for a in node.args if a.has(dependent)]
    if len(dependent_factors) == 1:
        return is_linear_in_dependent(dependent_factors[0], dependent)
    return len(dependent_factors) == 0


def is_linear_pow(node: sp.Pow, dependent: TrialFunction | sp.Function) -> bool:
    """Return True when a power node preserves linearity in ``dependent``."""
    base, exp = node.as_base_exp()
    if not base.has(dependent):
        return True
    if exp.is_number:
        if exp == 1:
            return is_linear_in_dependent(base, dependent)
        if exp == 0:
            return True
    return False


def is_linear_in_dependent(
    node: sp.Basic, dependent: TrialFunction | sp.Function
) -> bool:
    """Return True when ``node`` is linear in the supplied dependent field."""
    node = sp.sympify(node)

    if not node.has(dependent) or node == dependent:
        return True

    match node:
        case sp.Add():
            return is_linear_add(node, dependent)
        case sp.Mul():
            return is_linear_mul(node, dependent)
        case sp.Pow():
            return is_linear_pow(node, dependent)
        case sp.Derivative():
            return is_linear_in_dependent(node.expr, dependent)
        case _:
            pass

    if len(node.args) == 1 and node.func.__name__ in _LINEAR_UNARY:
        return is_linear_in_dependent(node.args[0], dependent)
    if len(node.args) == 2 and node.func.__name__ in _LINEAR_BINARY:
        a0, a1 = node.args
        has0 = a0.has(dependent)
        has1 = a1.has(dependent)
        if has0 and has1:
            return False
        if has0:
            return is_linear_in_dependent(a0, dependent)
        if has1:
            return is_linear_in_dependent(a1, dependent)
        return True

    if isinstance(node, sp.Function):  # Preserve check for later impl.
        return False
    return False
