import sympy as sp

from jaxfun.arguments import JAXArray, TestFunction, TrialFunction, test, trial


def get_basisfunctions(
    a: sp.Expr,
) -> tuple[TestFunction | test | None, TrialFunction | trial | None]:
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


def get_jaxarray(
    forms: sp.Expr,
) -> tuple[None | JAXArray, sp.Expr]:
    jaxarray = None
    for p in sp.core.traversal.iterargs(forms):
        if isinstance(p, JAXArray):
            jaxarray = True
            break
    if jaxarray:
        assert isinstance(forms, sp.Mul)
        formargs = []
        for arg in forms.args:
            if isinstance(arg, JAXArray):
                jaxarray = arg
            else:
                formargs.append(arg)
        forms = sp.Mul(*formargs)
    return jaxarray, forms


def split(forms) -> dict:
    v, _ = get_basisfunctions(forms)
    assert v is not None, "A test function is required"
    V = v.functionspace
    
    def _split(ms):
        d = sp.separatevars(ms, dict=True, symbols=V.system._base_scalars)
        if d is None and isinstance(ms, sp.Mul):
            scale = []
            rest = []
            for arg in ms.args:
                if isinstance(arg, sp.Derivative | test | trial):
                    rest.append(arg)
                else:
                    scale.append(arg)
            if len(rest) > 0:
                d = sp.separatevars(
                    sp.Mul(*rest), dict=True, symbols=V.system._base_scalars
                )
            if len(scale) > 0:
                d["multivar"] = sp.Mul(*scale)
        if d is None:
            raise RuntimeError("Could not split form")
        return d

    forms = forms.doit().expand()
    result = {"linear": [], "bilinear": []}
    if isinstance(forms, sp.Add):
        for arg in forms.args:
            jaxarray, remargs = get_jaxarray(arg)
            basisfunctions = get_basisfunctions(remargs)
            d = _split(remargs)
            if jaxarray is not None:
                d["jaxarray"] = jaxarray
            if basisfunctions[1] in (None, set()):
                result["linear"].append(d)
            else:
                result["bilinear"].append(d)

    else:
        jaxarray, remargs = get_jaxarray(forms)
        basisfunctions = get_basisfunctions(remargs)
        d = _split(remargs)
        if jaxarray is not None:
            d["jaxarray"] = jaxarray
        if basisfunctions[1] in (None, set()):
            result["linear"].append(d)
        else:
            result["bilinear"].append(d)

    return result
