from typing import Union
from collections import defaultdict
import sympy as sp
import numpy as np
from jaxfun.arguments import TestFunction, TrialFunction, test, trial


def get_basisfunctions(
    a: sp.Expr,
) -> list[Union[TestFunction, None], Union[TrialFunction, None]]:
    test_found, trial_found = None, None
    for p in sp.core.traversal.iterargs(a):
        if isinstance(p, (TrialFunction, trial)):
            trial_found = p
        if isinstance(p, (TestFunction, test)):
            test_found = p
        if test_found is not None and trial_found is not None:
            break
    return test_found, trial_found


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

def split(forms):
    v, u = get_basisfunctions(forms)
    assert v is not None, 'A test function is required'
    V = v.functionspace
    def _split(ms):
        d = sp.separatevars(ms, dict=True, symbols=V.system._base_scalars)
        if d is None:
            raise RuntimeError('Could not split form')
        return d

    forms = forms.doit()
    #original_args = list(forms.args)
    #scale = sp.S.One
    #if isinstance(forms, sp.Mul):
    #    nobasis = None
    #    for i, oa in enumerate(original_args):
    #        bf = get_basisfunctions(oa)
    #        if bf == (None, None):
    #            nobasis = i        
    #    if nobasis is not None:
    #        scale = V.system.simplify(original_args.pop(i))
    #        forms = sp.Mul(*original_args)
        
    forms = forms.expand()
    result = {'linear': [], 'bilinear': []}
    if isinstance(forms, sp.Add):
        for arg in forms.args:
            basisfunctions = get_basisfunctions(arg)
            d = _split(arg)
            if basisfunctions[1] is None:
                result['linear'].append(d)
            else:
                result['bilinear'].append(d)

    else:
        basisfunctions = get_basisfunctions(forms)
        d = _split(forms)
        if basisfunctions[1] is None:
            result['linear'].append(d)
        else:
            result['bilinear'].append(d)

    return result