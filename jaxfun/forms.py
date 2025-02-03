
import sympy as sp

from jaxfun.arguments import TestFunction, TrialFunction, test, trial


def get_basisfunctions(
    a: sp.Expr,
) -> list[TestFunction | None, TrialFunction | None]:
    test_found, trial_found = set(), set()
    for p in sp.core.traversal.iterargs(a):
        if isinstance(p, (TrialFunction, trial)):
            trial_found.add(p)
        if isinstance(p, (TestFunction, test)):
            test_found.add(p)
    if len(test_found) == 1 and len(trial_found) == 1:
        return test_found.pop(), trial_found.pop()
    elif len(test_found) == 1 and len(trial_found) == 0:
        return test_found.pop(), None
    elif len(test_found) == 0 and len(trial_found) == 1:
        return None, trial_found.pop()
    elif len(test_found) > 0 or len(trial_found) > 0:
        return test_found, trial_found
    return None, None


def split(forms) -> dict:
    v, u = get_basisfunctions(forms)
    assert v is not None, "A test function is required"
    V = v.functionspace

    def _split(ms):
        d = sp.separatevars(ms, dict=True, symbols=V.system._base_scalars)
        if d is None:
            raise RuntimeError("Could not split form")
        return d

    forms = forms.doit().expand()
    result = {"linear": [], "bilinear": []}
    if isinstance(forms, sp.Add):
        for arg in forms.args:
            basisfunctions = get_basisfunctions(arg)
            d = _split(arg)
            if basisfunctions[1] in (None, set()):
                result["linear"].append(d)
            else:
                result["bilinear"].append(d)

    else:
        basisfunctions = get_basisfunctions(forms)
        d = _split(forms)
        if basisfunctions[1] in (None, set()):
            result["linear"].append(d)
        else:
            result["bilinear"].append(d)

    return result
