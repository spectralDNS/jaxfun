from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from jaxfun.galerkin import TrialFunction


def get_time_independent(u: TrialFunction) -> TrialFunction:
    if not u.transient:
        return u
    V = u.functionspace
    name = u.name
    return u.func(V, name, transient=False)


def drop_time_argument(expr: sp.Expr, t: sp.Symbol) -> sp.Expr:
    from jaxfun.galerkin.arguments import TrialFunction

    expr = sp.sympify(expr)

    repl = {}
    for fapp in expr.atoms(TrialFunction):
        if t in fapp.args:
            repl[fapp] = get_time_independent(fapp)

    return expr.xreplace(repl)


def split_time_derivative_terms(
    expr: sp.Expr, time_symbol: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """Split an expression into time-derivative terms and the remaining terms.

    The input is treated as an additive expression. A term is classified as a
    time-derivative term if it contains a SymPy ``Derivative`` whose differentiation
    variables include ``time_symbol``.

    Args:
        expr: SymPy expression (typically an expanded sum of terms).
        time_symbol: Time variable symbol (e.g. ``t``).

    Returns:
        A tuple ``(time_terms, rest)`` where:

        - ``time_terms`` is the sum of terms that contain a derivative with respect
          to ``time_symbol``.
        - ``rest`` is the sum of all remaining terms.

    Examples:
        >>> import sympy as sp
        >>> x, t = sp.symbols("x t")
        >>> u = sp.Function("u")(x, t)
        >>> expr = u.diff(t) - u.diff(x, 2)
        >>> split_time_derivative_terms(expr, t)
        (Derivative(u(x, t), t), -Derivative(u(x), (x, 2)))

    Notes:
        If you represent a PDE residual as ``expr = LHS - RHS``, you can build
        explicit sides via ``LHS = time_terms`` and ``RHS = -rest``.
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

    The expression is treated as additive. We consider ``dependent`` (e.g.
    ``u(x, y, t)``) and all of its derivatives appearing in ``expr`` as algebraic
    generators, then classify each term by its total polynomial degree in those
    generators.

    Classification:
        - Degree 0 or 1: linear
        - Degree >= 2, or non-polynomial dependence: nonlinear

    This convention places forcing/constant terms (degree 0) in the linear piece.

    Args:
        expr: SymPy expression to split.
        dependent: Dependent function expression (e.g. ``u(x, y, t)``).

    Returns:
        A tuple ``(linear, nonlinear)``.

    Examples:
        >>> import sympy as sp
        >>> x, t = sp.symbols("x t")
        >>> u = sp.Function("u")(x, t)
        >>> expr = 2 * u.diff(x) + u * u.diff(x)
        >>> split_linear_nonlinear_terms(expr, u)
        (2*Derivative(u(x), x), u(x, t)*Derivative(u(x), x))
    """
    from jaxfun.galerkin.arguments import TrialFunction

    if isinstance(dependent, TrialFunction):
        dependent = get_time_independent(dependent)
    expr = sp.expand(expr)

    generators = sorted(
        ({dependent} | {d for d in expr.atoms(sp.Derivative) if d.expr == dependent}),
        key=sp.default_sort_key,
    )

    linear_terms: list[sp.Expr] = []
    nonlinear_terms: list[sp.Expr] = []

    for term in sp.Add.make_args(expr):
        if not term.has(*generators):
            linear_terms.append(term)
            continue

        try:
            poly = sp.Poly(term, *generators, domain="EX")
        except sp.PolynomialError:
            nonlinear_terms.append(term)
            continue

        if poly.total_degree() <= 1:
            linear_terms.append(term)
        else:
            nonlinear_terms.append(term)

    return sp.Add(*linear_terms), sp.Add(*nonlinear_terms)
