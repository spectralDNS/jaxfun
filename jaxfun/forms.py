import sympy as sp

from jaxfun.arguments import TestFunction, TrialFunction, test, trial


def get_basisfunctions(
    a: sp.Expr,
) -> list[TestFunction | None, TrialFunction | None]:
    test_found, trial_found = set(), set()
    for p in sp.core.traversal.iterargs(a):
        if isinstance(p, TrialFunction | trial):
            trial_found.add(p)
        if isinstance(p, TestFunction | test):
            test_found.add(p)

    match (len(test_found), len(trial_found)):
        case (1, 1):
            return test_found.pop(), trial_found.pop()
        case (1, 0):
            return test_found.pop(), None
        case (0, 1):
            return None, trial_found.pop()
        case _ if len(test_found) > 0 or len(trial_found) > 0:
            return test_found, trial_found
        case _:
            return None, None


def split(forms) -> dict:
    v, _ = get_basisfunctions(forms)
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
