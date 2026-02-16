"""Utilities for splitting Galerkin weak forms into linear / bilinear parts.

The functions traverse SymPy expressions composed of:
  * Test / trial basis Functions (argument attribute 0 / 1)
  * JAX-backed symbolic wrappers (JAXFunction, Jaxf, JAXArray)
  * Derivatives, sums, products, numeric coefficients

They identify:
  - Basis function structure (test / trial presence)
  - Coefficient factors (scalars, arrays, JAXFunction references)
  - Separable variable factors (multivariate vs separated)

Main entry points:
  get_basisfunctions(expr) -> locate test / trial functions
  get_jaxarrays(expr)      -> locate JAXArray symbols
  split_coeff(expr)        -> split bilinear/linear coefficient (for bilinear forms)
  split_linear_coeff(expr) -> assemble scalar/array coefficient (for linear forms)
  split(form_expr)         -> decompose full weak form into grouped terms
"""

from typing import Protocol, TypeGuard

import jax.numpy as jnp
import sympy as sp
from jax import Array
from sympy.core.function import AppliedUndef

from jaxfun.coordinates import CoordSys, get_system as get_system
from jaxfun.typing import (
    CoeffDict,
    FunctionSpaceType,
    InnerResultDict,
    LinearCoeffDict,
    ResultDict,
    TestSpaceType,
)

from .arguments import JAXArray, Jaxc, Jaxf, JAXFunction, TestFunction, TrialFunction


class _HasTestSpace(Protocol):
    functionspace: TestSpaceType


def _has_testspace(obj: object) -> TypeGuard[_HasTestSpace]:
    return hasattr(obj, "functionspace")


class _HasFunctionSpace(Protocol):
    functionspace: FunctionSpaceType


def _has_functionspace(obj: object) -> TypeGuard[_HasFunctionSpace]:
    return hasattr(obj, "functionspace")


class _HasGlobalIndex(Protocol):
    global_index: int


def _has_globalindex(obj: object) -> TypeGuard[_HasGlobalIndex]:
    return hasattr(obj, "global_index")


def get_basisfunctions(
    a: sp.Expr,
) -> tuple[
    set[TestFunction | AppliedUndef] | TestFunction | AppliedUndef | None,
    set[TrialFunction | AppliedUndef] | TrialFunction | AppliedUndef | None,
]:
    """Return test / trial basis Function objects present in expression.

    A basis Function is recognized by a custom attribute 'argument':
      argument == 0 : test function
      argument == 1 : trial function

    Depending on multiplicity the return values are either a single
    Function, a set of Functions, or None if absent.

    Args:
        a: SymPy expression to inspect.

    Returns:
        (test, trial) where each element is:
          - A single Function if exactly one found
          - A set of Functions if multiple
          - None if none found
    """
    test_found: set[TestFunction | AppliedUndef] = set()
    trial_found: set[TrialFunction | AppliedUndef] = set()
    for p in sp.core.traversal.iterargs(sp.sympify(a)):
        if isinstance(p, TestFunction | TrialFunction | AppliedUndef):
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


def get_jaxarrays(
    a: sp.Expr | float,
) -> set[JAXArray]:
    """Return set of JAXArray symbolic wrappers inside expression.

    JAXArray nodes are identified through attribute 'argument' == 3.

    Args:
        a: SymPy expression.

    Returns:
        Set with zero or more JAXArray objects.
    """
    return sp.sympify(a).atoms(JAXArray)


def get_jaxfunctions(
    a: sp.Expr | float,
) -> set[JAXFunction]:
    """Return set of JAXFunction symbolic wrappers inside expression.

    JAXFunction nodes are identified through attribute 'argument' == 2.

    Args:
        a: SymPy expression.

    Returns:
        Set with zero or more JAXArray objects.
    """
    jaxfunctions: set[JAXFunction] = set()
    for p in sp.core.traversal.iterargs(sp.sympify(a)):
        if getattr(p, "argument", -1) == 2:
            jaxfunctions.add(p)
    return jaxfunctions


def get_jaxf(a: sp.Expr | float) -> set[Jaxf]:
    """Return set of Jaxf symbolic wrappers inside expression.

    Args:
        a: SymPy expression.

    Returns:
        Set with zero or more Jaxf objects.
    """
    return sp.sympify(a).atoms(Jaxf)


def check_if_nonlinear_in_jaxfunction(a: sp.Expr) -> bool:
    """Check if expression is nonlinear in any JAXFunction.

    Args:
        a: SymPy expression.

    Returns:
        True if expression is linear in any JAXFunction, False otherwise.
    """
    jaxfunctions = get_jaxfunctions(a)
    have_jaxfunctions = len(jaxfunctions) > 0 or len(a.atoms(Jaxc)) > 0
    if not have_jaxfunctions:
        return False
    assert len(jaxfunctions) <= 1, "Multiple JAXFunctions found"
    ad = a.doit(linear=True)  # assume linear
    jf = ad.atoms(Jaxc).pop()
    return sp.diff(ad, jf, 2) != 0


def split_coeff(c0: sp.Expr | float) -> CoeffDict:
    """Split coefficient for bilinear form into linear / bilinear pieces.

    Patterns handled:
      * Pure number -> {'bilinear': scalar}
      * Single Jaxc -> {'linear': {'scale': 1, 'jaxfunction': Jaxc}}
      * Product including Jaxc -> scale isolated
      * Sum of numbers / (scaled) Jaxc terms -> combined

    Args:
        c0: SymPy expression with optional Jaxc factor(s).
    Returns:
        Dictionary with possible keys:
          'bilinear': numeric scalar (if present)
          'linear': {'scale': number, NotRequired('jaxfunction': Jaxc)}

    Raises:
        AssertionError: If basis functions are present in the coefficient.
    """
    coeffs = CoeffDict()
    c0 = sp.sympify(c0)
    assert get_basisfunctions(c0) == (None, None), (
        "Basis functions found in coefficient"
    )

    if c0.is_number:
        coeffs["bilinear"] = float(c0) if c0.is_real else complex(c0)

    elif isinstance(c0, Jaxc):
        coeffs["linear"] = LinearCoeffDict(scale=1, jaxcoeff=c0)

    elif isinstance(c0, sp.Mul):
        coeffs["linear"] = LinearCoeffDict(scale=1)
        for ci in c0.args:
            if isinstance(ci, Jaxc):
                coeffs["linear"]["jaxcoeff"] = ci
            else:
                coeffs["linear"]["scale"] *= float(ci) if ci.is_real else complex(ci)

    elif isinstance(c0, sp.Add):
        linear_coeffs = LinearCoeffDict(scale=1)
        coeffs.update(CoeffDict(linear=linear_coeffs, bilinear=0))
        for arg in c0.args:
            if arg.is_number:
                coeffs["bilinear"] = float(arg) if arg.is_real else complex(arg)
            elif isinstance(arg, Jaxc):
                coeffs["linear"]["jaxcoeff"] = arg
            elif isinstance(arg, sp.Mul):
                for ci in arg.args:
                    if isinstance(ci, Jaxc):
                        coeffs["linear"]["jaxcoeff"] = ci
                    else:
                        coeffs["linear"]["scale"] *= (
                            float(ci) if ci.is_real else complex(ci)
                        )
    return coeffs


def split_linear_coeff(c0: sp.Expr | float) -> Array:
    """Assemble (possibly array-valued) coefficient for a linear form term.

    Accepts at most one JAXArray node (optionally raised to an integer
    power). All numeric / complex factors are multiplied into a single
    scalar and broadcast with the array.

    Args:
        c0: SymPy expression (number, JAXArray, product, or power).

    Returns:
        jnp.ndarray: Final numeric / array scale.

    Raises:
        AssertionError: If more than one JAXArray is present.
        NotImplementedError: If unsupported power pattern encountered.
    """
    c0 = sp.sympify(c0)
    jaxarrays = get_jaxarrays(c0)
    assert len(jaxarrays) <= 1, "Multiple JAXArrays found in coefficient"

    scale = jnp.array(1.0)
    args = c0.args if isinstance(c0, sp.Mul) else (c0,)

    for ci in args:
        if isinstance(ci, JAXArray):
            scale *= ci.array
        elif isinstance(ci, sp.Pow):
            base, exp = ci.args
            if isinstance(base, JAXArray) and exp.is_number:
                scale = base.array ** int(exp)
            else:
                raise NotImplementedError(
                    "Only power of JAXArray with integer exponent allowed in linear "
                    "form at present"
                )
        else:
            scale *= float(ci) if ci.is_real else complex(ci)

    return scale


def split(forms: sp.Expr) -> ResultDict:
    """Split a full weak form expression into linear and bilinear parts.

    For each additive term:
      * Identify presence of trial basis -> classify as bilinear
      * Otherwise classify as linear
      * Separate variable factors and coefficient (JAXFunction / Jaxc)
      * Combine like terms via add_result

    Args:
        forms: SymPy expression representing the (expanded) weak form.

    Returns:
        {
          'linear':  [dict, ...],
          'bilinear':[dict, ...]
        }
        Each dict contains separated variables keyed by base scalars plus
        optional 'coeff' and 'multivar' entries.

    Raises:
        AssertionError: If no test function is found.
        RuntimeError: If term cannot be separated.
    """
    v, _ = get_basisfunctions(forms)
    assert v is not None, "A test function is required"
    assert not isinstance(v, set), "Multiple test functions not supported"
    assert _has_functionspace(v)
    V = v.functionspace

    def _split(ms: sp.Expr) -> InnerResultDict:
        d: InnerResultDict | None = sp.separatevars(
            ms, dict=True, symbols=V.system._base_scalars
        )
        if d is None and isinstance(ms, sp.Mul):
            multivar = []
            rest = []
            jfun = []
            jaxc = []
            for arg in ms.args:
                test, trial = get_basisfunctions(arg)
                if test is not None or trial is not None:
                    rest.append(arg)
                    continue
                jaxfuns = get_jaxfunctions(arg)
                if len(jaxfuns) > 0:
                    jfun.append(arg)
                    continue
                if arg.atoms(Jaxc):
                    jaxc.append(arg)
                    continue
                if len(arg.free_symbols) == 1:
                    rest.append(arg)
                else:
                    multivar.append(arg)

                # if isinstance(arg, sp.Derivative) or hasattr(arg, "argument"):
                #    rest.append(arg)
                # elif isinstance(arg, JAXFunction | Jaxf):
                #    jfun.append(arg)
                # else:
                #    scale.append(arg)
            if len(rest) > 0:
                d: InnerResultDict = sp.separatevars(
                    sp.Mul(*rest), dict=True, symbols=V.system._base_scalars
                )
            if isinstance(d, dict):
                if len(multivar) > 0:
                    d["multivar"] = sp.Mul(*multivar)
                if len(jfun) > 0:
                    d["jaxfunction"] = jfun[0]
                if len(jaxc) > 0:
                    d["coeff"] = jaxc[0]
        if d is None:
            raise RuntimeError("Could not split form")
        return d

    result = ResultDict(linear=[], bilinear=[])
    for arg in sp.Add.make_args(forms):
        basisfunctions = get_basisfunctions(arg)
        bilinear = basisfunctions[1] not in (None, set())
        arg = arg.doit(linear=not (bilinear or check_if_nonlinear_in_jaxfunction(arg)))
        for argi in sp.Add.make_args(arg.factor().expand()):
            basisfunctions = get_basisfunctions(argi)
            d = _split(argi)
            if basisfunctions[1] in (None, set()):
                result["linear"] = add_result(result["linear"], d, V.system)
            else:
                result["bilinear"] = add_result(result["bilinear"], d, V.system)

    return result


def add_result(
    res: list[InnerResultDict], d: InnerResultDict, system: CoordSys
) -> list[InnerResultDict]:
    """Accumulate result dictionary into list merging like basis factors.

    Two dictionaries are considered identical if they match on every
    coordinate scalar key. Coefficients are combined:
      * Without 'multivar': add coefficient
      * With 'multivar': distribute & factor

    Args:
        res: Existing list of grouped term dictionaries.
        d: New term dictionary.
        system: Coordinate system providing base scalar ordering.

    Returns:
        Updated list with merged or appended dictionary.
    """
    found_d: bool = False
    for g in res:
        if jnp.all(jnp.array([g[s] == d[s] for s in system.base_scalars()])):
            if "multivar" not in d:
                g["coeff"] += d["coeff"]
            elif "multivar" not in g:
                g["multivar"] = d["coeff"] * d["multivar"]
                g["coeff"] = 1
            else:
                g["multivar"] = (
                    g["coeff"] * g["multivar"] + d["coeff"] * d["multivar"]
                ).factor()
                g["coeff"] = 1
            found_d = True
            break

    if not found_d:
        res.append(d)
    return res
