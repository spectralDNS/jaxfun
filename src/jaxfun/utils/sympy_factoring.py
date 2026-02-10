from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from jaxfun.galerkin import TrialFunction

# Operators treated as linear in the dependent field.
_LINEAR_UNARY = {"Grad", "Div", "Curl", "Derivative"}
_LINEAR_BINARY = {"Dot", "Cross", "Outer"}


def get_time_independent(u: TrialFunction) -> TrialFunction:
    if not u.transient:
        return u
    V = u.functionspace
    name = u.name
    return u.func(V, name, transient=False)


def drop_time_argument(expr: sp.Expr, t: sp.Symbol) -> sp.Expr:
    from jaxfun.galerkin.arguments import TrialFunction

    expr = sp.sympify(expr)
    repl = {
        fapp: get_time_independent(fapp)
        for fapp in expr.atoms(TrialFunction)
        if t in fapp.args
    }
    return expr.xreplace(repl)


def split_time_derivative_terms(
    expr: sp.Expr, time_symbol: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """Split an expression into time-derivative terms and the remainder."""
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

    def is_linear_in_dependent(node: sp.Basic) -> bool:
        node = sp.sympify(node)

        if not node.has(dependent):
            return True
        if node == dependent:
            return True

        if isinstance(node, sp.Add):
            return all(is_linear_in_dependent(arg) for arg in node.args)

        if isinstance(node, sp.Mul):
            dependent_args = [arg for arg in node.args if arg.has(dependent)]
            if not dependent_args:
                return True
            if len(dependent_args) > 1:
                return False
            return is_linear_in_dependent(dependent_args[0])

        if isinstance(node, sp.Pow):
            base, exp = node.as_base_exp()
            if not base.has(dependent):
                return True
            if exp.is_number:
                if exp == 1:
                    return is_linear_in_dependent(base)
                if exp == 0:
                    return True
            return False

        if isinstance(node, sp.Derivative):
            return is_linear_in_dependent(node.expr)

        if len(node.args) == 1 and node.func.__name__ in _LINEAR_UNARY:
            return is_linear_in_dependent(node.args[0])

        if len(node.args) == 2 and node.func.__name__ in _LINEAR_BINARY:
            a0, a1 = node.args
            has0 = a0.has(dependent)
            has1 = a1.has(dependent)
            if has0 and has1:
                return False
            if has0:
                return is_linear_in_dependent(a0)
            if has1:
                return is_linear_in_dependent(a1)
            return True

        if isinstance(node, sp.Function):
            return False

        return False

    linear_terms: list[sp.Expr] = []
    nonlinear_terms: list[sp.Expr] = []

    for term in sp.Add.make_args(expr):
        if is_linear_in_dependent(term):
            linear_terms.append(term)
        else:
            nonlinear_terms.append(term)

    return sp.Add(*linear_terms), sp.Add(*nonlinear_terms)
