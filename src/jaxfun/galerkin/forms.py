import jax.numpy as jnp
import sympy as sp

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.arguments import Jaxf, JAXFunction


def get_basisfunctions(
    a: sp.Expr,
) -> tuple[
    set[sp.Function] | sp.Function | None,
    set[sp.Function] | sp.Function | None,
]:
    test_found, trial_found = set(), set()
    for p in sp.core.traversal.iterargs(a):
        if getattr(p, "argument", -1) == 1:
            trial_found.add(p)
        if getattr(p, "argument", -1) == 0:
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


def get_system(a: sp.Expr) -> CoordSys:
    for p in sp.core.traversal.iterargs(a):
        if isinstance(p, CoordSys):
            return p
    raise RuntimeError("CoordSys not found")


def split_coeff(c0: sp.Expr) -> dict:
    coeffs = {}
    c0 = sp.sympify(c0)

    if c0.is_number:
        coeffs["bilinear"] = float(c0) if c0.is_real else complex(c0)

    elif isinstance(c0, Jaxf):
        coeffs["linear"] = {"scale": 1, "jaxfunction": c0}

    elif isinstance(c0, sp.Mul):
        coeffs["linear"] = {"scale": 1, "jaxfunction": None}
        for ci in c0.args:
            if isinstance(ci, Jaxf):
                coeffs["linear"]["jaxfunction"] = ci
            else:
                coeffs["linear"]["scale"] *= float(ci) if ci.is_real else complex(ci)

    elif isinstance(c0, sp.Add):
        coeffs.update({"linear": {"scale": 1, "jaxfunction": None}, "bilinear": 0})
        for arg in c0.args:
            if arg.is_number:
                coeffs["bilinear"] = float(arg) if arg.is_real else complex(arg)
            elif isinstance(arg, Jaxf):
                coeffs["linear"]["jaxfunction"] = arg
            elif isinstance(arg, sp.Mul):
                for ci in arg.args:
                    if isinstance(ci, Jaxf):
                        coeffs["linear"]["jaxfunction"] = ci
                    else:
                        print(ci, c0, type(c0.free_symbols.pop()))
                        coeffs["linear"]["scale"] *= (
                            float(ci) if ci.is_real else complex(ci)
                        )
    return coeffs


def split(forms: sp.Expr) -> dict:
    v, _ = get_basisfunctions(forms)
    assert v is not None, "A test function is required"
    V = v.functionspace

    def _split(ms):
        d = sp.separatevars(ms, dict=True, symbols=V.system._base_scalars)
        if d is None and isinstance(ms, sp.Mul):
            scale = []
            rest = []
            jfun = []
            for arg in ms.args:
                if isinstance(arg, sp.Derivative) or hasattr(arg, "argument"):
                    rest.append(arg)
                elif isinstance(arg, JAXFunction | Jaxf):
                    jfun.append(arg)
                else:
                    scale.append(arg)
            if len(rest) > 0:
                d = sp.separatevars(
                    sp.Mul(*rest), dict=True, symbols=V.system._base_scalars
                )
            if len(scale) > 0:
                d["multivar"] = sp.Mul(*scale)
            if len(jfun) > 0:
                d["coeff"] = jfun[0]
        if d is None:
            raise RuntimeError("Could not split form")
        return d

    result = {"linear": [], "bilinear": []}
    for arg in sp.Add.make_args(forms.doit().factor().expand()):
        basisfunctions = get_basisfunctions(arg)
        d = _split(arg)
        if basisfunctions[1] in (None, set()):
            result["linear"] = add_result(result["linear"], d, V.system)
        else:
            result["bilinear"] = add_result(result["bilinear"], d, V.system)

    return result


def add_result(res: list[dict], d: dict, system: CoordSys) -> list[dict]:
    """Add result dictionary `d` to list of results `res`.

    Collect all results with identical basis functions in one dict.

    Args:
        res (list[dict]): list of results
        d (dict): new result
        system (CoordSys): Coordinate system

    Returns:
        list[dict]: appended list of results
    """
    found_d: bool = False
    for g in res:
        if jnp.all(jnp.array([g[s] == d[s] for s in system.base_scalars()])):
            if "multivar" not in d:
                g["coeff"] += d["coeff"]
            else:
                g["multivar"] = g["coeff"] * g["multivar"] + d["coeff"] * d["multivar"]
                g["coeff"] = 1
                g["multivar"] = g["multivar"].factor()
            found_d = True
            break

    if not found_d:
        res.append(d)
    return res
