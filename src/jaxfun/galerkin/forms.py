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

from typing import NotRequired, Protocol, TypeGuard

import jax.numpy as jnp
import sympy as sp
from jax import Array
from typing_extensions import TypedDict

from jaxfun.coordinates import CoordSys, get_system as get_system
from jaxfun.typing import FunctionSpaceType

from .arguments import JAXArray, Jaxf, JAXFunction


class _HasFunctionSpace(Protocol):
    functionspace: FunctionSpaceType


def _has_functionspace(obj: object) -> TypeGuard[_HasFunctionSpace]:
    return hasattr(obj, "functionspace")


def get_basisfunctions(
    a: sp.Expr,
) -> tuple[
    set[sp.Function] | sp.Function | None,
    set[sp.Function] | sp.Function | None,
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
    test_found, trial_found = set(), set()
    for p in sp.core.traversal.iterargs(sp.sympify(a)):
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
    array_found = set()
    for p in sp.core.traversal.iterargs(sp.sympify(a)):
        if getattr(p, "argument", -1) == 3:
            array_found.add(p)
    return array_found


class LinearCoeffDict(TypedDict, total=False):
    scale: float
    jaxfunction: NotRequired[Jaxf]


class CoeffDict(TypedDict, total=False):
    bilinear: complex
    linear: LinearCoeffDict


def split_coeff(c0: sp.Expr | float) -> CoeffDict:
    """Split coefficient for bilinear form into linear / bilinear pieces.

    Patterns handled:
      * Pure number -> {'bilinear': scalar}
      * Single Jaxf -> {'linear': {'scale': 1, 'jaxfunction': Jaxf}}
      * Product including Jaxf -> scale isolated
      * Sum of numbers / (scaled) Jaxf terms -> combined

    Args:
        c0: SymPy expression with optional Jaxf factor(s).

    Returns:
        Dictionary with possible keys:
          'bilinear': numeric scalar (if present)
          'linear': {'scale': number, NotRequired('jaxfunction': Jaxf)}

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

    elif isinstance(c0, Jaxf):
        coeffs["linear"] = LinearCoeffDict(scale=1, jaxfunction=c0)

    elif isinstance(c0, sp.Mul):
        coeffs["linear"] = LinearCoeffDict(scale=1)
        for ci in c0.args:
            if isinstance(ci, Jaxf):
                coeffs["linear"]["jaxfunction"] = ci
            else:
                coeffs["linear"]["scale"] *= float(ci) if ci.is_real else complex(ci)

    elif isinstance(c0, sp.Add):
        linear_coeffs = LinearCoeffDict(scale=1)
        coeffs.update(CoeffDict(linear=linear_coeffs, bilinear=0))
        # coeffs.update({"linear": {"scale": 1, "jaxfunction": None}, "bilinear": 0})
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


class InnerResultDict(TypedDict, extra_items=sp.Expr):
    coeff: sp.Expr | float
    multivar: NotRequired[sp.Expr]


class ResultDict(TypedDict):
    linear: list[InnerResultDict]
    bilinear: list[InnerResultDict]


def split(forms: sp.Expr) -> ResultDict:
    """Split a full weak form expression into linear and bilinear parts.

    For each additive term:
      * Identify presence of trial basis -> classify as bilinear
      * Otherwise classify as linear
      * Separate variable factors and coefficient (JAXFunction / Jaxf)
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
                d: InnerResultDict = sp.separatevars(
                    sp.Mul(*rest), dict=True, symbols=V.system._base_scalars
                )
            if isinstance(d, dict):
                if len(scale) > 0:
                    d["multivar"] = sp.Mul(*scale)
                if len(jfun) > 0:
                    d["coeff"] = jfun[0]
        if d is None:
            raise RuntimeError("Could not split form")
        return d

    result = ResultDict(linear=[], bilinear=[])
    for arg in sp.Add.make_args(forms.doit().factor().expand()):
        basisfunctions = get_basisfunctions(arg)
        d = _split(arg)
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
