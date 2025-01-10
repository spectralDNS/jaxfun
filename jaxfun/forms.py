from typing import Union
import sympy as sp
from jaxfun.arguments import TestFunction
from jaxfun.arguments import TrialFunction


def get_basisfunctions(
    a: sp.Expr,
) -> list[Union[TestFunction, None], Union[TrialFunction, None]]:
    test, trial = None, None
    for p in sp.core.traversal.iterargs(a):
        if isinstance(p, TrialFunction):
            trial = p
        if isinstance(p, TestFunction):
            test = p
    return test, trial


def inspect_form(a: sp.Expr) -> list[list[sp.Expr], list[sp.Expr]]:
    num_test = a.count(TestFunction)
    num_trial = a.count(TrialFunction)
    a = a.doit()
    aforms = []
    bforms = []
    if num_test > 0 and num_trial > 0:
        # bilinear form
        assert num_test == num_trial
        aforms = a.args if isinstance(a, sp.core.Add) else [a]
    elif num_test > 0:
        # linear form
        assert num_trial == 0
        bforms = a.args if isinstance(a, sp.core.Add) else [a]

    return aforms, bforms
