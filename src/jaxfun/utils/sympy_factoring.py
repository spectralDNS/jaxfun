from __future__ import annotations

import sympy as sp


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
        (Derivative(u(x, t), t), -Derivative(u(x, t), (x, 2)))

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

    return sp.Add(*time_terms), sp.Add(*other_terms)


def split_linear_nonlinear_terms(
    expr: sp.Expr, dependent: sp.Expr
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
        (2*Derivative(u(x, t), x), u(x, t)*Derivative(u(x, t), x))
    """
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
