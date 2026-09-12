from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeGuard, cast, overload

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx
from jax import Array

from jaxfun.coordinates import BaseTime
from jaxfun.la import (
    BaseMatrix,
    BlockArray,
    BlockMatrix,
    DiaMatrix,
    GlobalArray,
    GlobalMatrix,
    Matrix,
    TensorMatrix,
    TPMatrices,
    TPMatrix,
)
from jaxfun.sharding import place, replicate
from jaxfun.typing import (
    CoeffDict,
    ComputationalSpaceType,
    FunctionSpaceType,
    GalerkinAssembledForm,
    InnerItems,
    InnerKind,
    InnerKindLike,
    InnerResultDict,
    RankTag,
    ScalarSpaceType,
    TrialSpaceType,
)
from jaxfun.utils.common import ArrayFn, lambdify, matmat

from .arguments import (
    ArgumentTag,
    TestFunction,
    TrialFunction,
    evaluate_jaxfunction_expr_quad,
    get_arg,
)
from .cartesianproductspace import (
    CartesianProductSpace,
    CartesianTensorProductSpace,
    VectorTensorProductSpace,
)
from .composite import BCGeneric, Composite, DirectSum
from .forms import (
    _has_functionspace,
    _has_globalindex,
    _has_testspace,
    get_basisfunctions,
    get_jaxfunctions,
    split,
    split_coeff,
    unwrap_single_testfunction,
)
from .orthogonal import OrthogonalSpace
from .tensorproductspace import (
    DirectSumTPS,
    TensorProductSpace,
)

type _NumQuadPoints = int | tuple[int | None, ...]
type _BilinearFactor = tuple[Array, Array] | Matrix | DiaMatrix
type _BilinearMat = tuple[_BilinearFactor, tuple[int, int]]
type _InnerTerm = tuple[BaseMatrix | GlobalMatrix | None, GlobalArray | None]
type _LinearFactor = Array | tuple[Array, ...]


@dataclass(frozen=True)
class _BoundarySpec:
    """Everything about a boundary block except the operator itself.

    Kept apart because this half is static -- read as Python values, hashed and
    compared when a jit cache is consulted -- while the operator is an array and
    must travel as data. A frozen dataclass holding an array compares elementwise
    and raises when JAX checks two treedefs for equality.
    """

    kind: Literal["1d", "separable", "multivar"]
    space: DirectSumTPS | BCGeneric
    key: tuple[OrthogonalSpace, ...] | None
    sign: int
    global_index: int

    def contract(self, op: object, lifting: Array) -> Array:
        if self.kind == "1d":
            return _contract_1d(cast(Matrix | DiaMatrix, op), self.sign, lifting)
        if self.kind == "multivar":
            return _contract_multivar(cast(Array, op), self.sign, lifting)
        return _contract_separable(list(cast(tuple, op)), self.sign, lifting)


@dataclass(frozen=True)
class _BoundaryBlock:
    """One assembled boundary block, kept apart from its lifting.

    `op` maps lifting coefficients to a test-space load vector, and is fixed:
    the boundary basis functions do not move, so only the coefficients they are
    contracted with can vary. `space`/`key` say where those coefficients live --
    a `DirectSumTPS` and one of its `bndvals` keys, or, in one dimension, the
    `BCGeneric` whose ordered values *are* the coefficients.
    """

    kind: Literal["1d", "separable", "multivar"]
    op: object
    space: DirectSumTPS | BCGeneric
    key: tuple[OrthogonalSpace, ...] | None
    sign: int
    global_index: int

    @property
    def spec(self) -> _BoundarySpec:
        """The block without its operator, safe to keep as pytree metadata."""
        return _BoundarySpec(
            self.kind, self.space, self.key, self.sign, self.global_index
        )


@dataclass(frozen=True)
class _InnerContext:
    test_space: ComputationalSpaceType
    trial_space: TrialSpaceType | None
    a_forms: list[InnerResultDict]
    b_forms: list[InnerResultDict]
    num_quad_points: _NumQuadPoints
    all_linear: bool
    # When set, the boundary blocks of every bilinear form are recorded here
    # instead of being contracted into a load vector. See `inner_boundary`.
    boundary_blocks: list[_BoundaryBlock] | None = None


@overload
def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
    *,
    kind: Literal[InnerKind.BILINEAR, "bilinear"],
) -> BaseMatrix: ...
@overload
def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
    *,
    kind: Literal[InnerKind.LINEAR, "linear"],
) -> Array | BlockArray: ...
@overload
def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
    *,
    kind: Literal[InnerKind.SYSTEM, "system"],
) -> tuple[BaseMatrix, Array | BlockArray]: ...
@overload
def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
    *,
    kind: None = None,
) -> GalerkinAssembledForm: ...
@overload
def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
    *,
    kind: InnerKindLike,
) -> GalerkinAssembledForm: ...
def inner(
    expr: sp.Expr,
    sparse: bool = False,
    sparse_tol: int = 1000,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
    *,
    kind: InnerKind | str | None = None,
) -> GalerkinAssembledForm:
    r"""Assemble Galerkin inner products (bilinear / linear forms).

    Supports expressions of the forms:
        a(u, v) - L(v)
        a(u, v)
        L(v)

    Finds test / trial functions, splits expression into coefficients and
    separated coordinate factors, constructs (tensor) matrices and load
    vectors. Handles:
      * Composite / BCGeneric spaces (boundary constraints / lifting)
      * Direct sums of tensor product spaces
      * Multivariable non-separable coefficients (symbolic factor kept)
      * JAXFunction (produces linear contributions)

    Args:
        expr: SymPy expression containing TestFunction (mandatory) and
            optionally TrialFunction, JAXFunction, scalar
            coordinate-dependent factors.
        sparse: If True, sparsify (1D) matrix/tensor factors (DiaMatrix).
        sparse_tol: Zero tolerance (integer multiple of ulp) for sparsify.
        num_quad_points: Number of quadrature points to use for evaluating
            the inner products. Can be an integer (1D) or a tuple of integers
            for each dimension. If None, the default number of quadrature
            points is used. This is sufficient for bilinear forms with
            constant coefficients, but for nonlinear forms or forms with
            non-constant coefficients, the number of quadrature points may need
            to be increased for exact integration.
            Note that this offers the possibility to use de-aliasing for
            nonlinear terms, but the user is responsible for ensuring that
            the number of quadrature points is sufficient. For the 3/2 rule,
            use tuple(int(1.5 * n) for n in test_space.num_quad_points).
        use_precomputed_matrices: If True, use precomputed sparse matrices if
            available for the given test/trial derivative orders. If False,
            always compute matrices via quadrature. Setting to False can be
            useful for testing.
        kind: Optional expected result kind. If omitted, the assembled result
            is returned unchanged. If provided, accepts ``InnerKind`` or its
            string values (``"bilinear"``, ``"linear"``, ``"system"``) and
            validates the assembled result before returning it.


    Returns:
        Depending on content:
          * Matrix (bilinear only, 1D)
          * Vector (linear only)
          * (Matrix, Vector) tuple
          * TensorMatrix / TPMatrix objects for >1D
    """  # noqa: E501
    if kind is not None:
        kind = _coerce_inner_kind(kind)
    context = _prepare_inner_context(expr, num_quad_points)
    aresults, bresults = _assemble_inner_items(context, use_precomputed_matrices)
    result = _finalize_inner_result(
        aresults, bresults, context.test_space, context.trial_space, sparse, sparse_tol
    )
    if kind is None:
        return result
    return _validate_inner_kind(result, kind)


def _coerce_inner_kind(kind: InnerKind | str) -> InnerKind:
    try:
        return InnerKind(kind)
    except ValueError as e:
        valid = ", ".join(repr(member.value) for member in InnerKind)
        e.add_note(f"Expected one of: {valid}")
        raise


def _validate_inner_kind(
    result: GalerkinAssembledForm, kind: InnerKind
) -> BaseMatrix | Array | BlockArray | tuple[BaseMatrix, Array | BlockArray]:
    if isinstance(result, BaseMatrix):
        actual = InnerKind.BILINEAR
    elif isinstance(result, Array | BlockArray):
        actual = InnerKind.LINEAR
    else:
        actual = InnerKind.SYSTEM

    if actual != kind:
        raise ValueError(
            f"inner(..., kind={kind!r}) assembled {actual.value}; expected {kind}"
        )
    return result


class BoundaryForcing(nnx.Module):
    """The boundary contribution to a weak form, as a function of time.

    Calling it returns the load vector at a given time; `rate` returns its time
    derivative. Both are traceable in `t`.
    """

    def __init__(
        self,
        blocks: Sequence[_BoundaryBlock],
        rank: RankTag,
        outer_sign: int = 1,
    ) -> None:
        # Operators are data -- they cross `_advance`'s jit boundary as traced
        # arguments, like the stage operators do. Everything read as a Python
        # value stays static.
        self._ops = nnx.data(tuple(b.op for b in blocks))
        self._specs = nnx.static(tuple(b.spec for b in blocks))
        self._rank = nnx.static(rank)
        # `split_operator_and_forcing` flips the sign of everything `inner`
        # moved to the other side of the equation, so that an assembled term
        # always reads `operator @ u + forcing`. A caller working in that
        # convention asks for the same flip here; the default reproduces what
        # `inner` itself returned.
        self._outer_sign = nnx.static(outer_sign)

    @property
    def is_transient(self) -> bool:
        """Return True if any contributing boundary value varies in time."""
        return any(_block_is_transient(b) for b in self._specs)

    def __call__(self, t: float | Array | None = None) -> Array:
        """Return the boundary load vector at time `t`."""
        return self._evaluate(t, rate=False)

    def rate(self, t: float | Array | None = None) -> Array:
        """Return d/dt of the boundary load vector at time `t`."""
        return self._evaluate(t, rate=True)

    def _evaluate(self, t: float | Array | None, rate: bool) -> Array:
        if self._rank != RankTag.SCALAR:
            raise NotImplementedError(
                "Deferred boundary forcing is implemented for scalar equations; "
                "assemble each component of a block system separately."
            )
        # One lifting evaluation per space, however many blocks read it.
        liftings: dict[int, Array | dict] = {}
        total: Array | None = None
        for spec, op in zip(self._specs, self._ops, strict=True):
            holder = id(spec.space)
            if holder not in liftings:
                liftings[holder] = _lifting_of(spec.space, t, rate)
            values = liftings[holder]
            # A key of `None` is the one-dimensional case, where the lifting is
            # one flat vector rather than one block per tensor subspace.
            fun = (
                cast(Array, values)
                if spec.key is None
                else cast(dict, values)[spec.key]
            )
            term = spec.contract(op, fun)
            total = term if total is None else total + term
        assert total is not None
        return self._outer_sign * total


def _lifting_of(
    space: DirectSumTPS | BCGeneric, t: float | Array | None, rate: bool
) -> Array | dict:
    """Return a space's lifting coefficients, or their time derivative.

    One dimension keeps its coefficients as a flat ordered vector on the
    `BCGeneric`; higher dimensions keep one block per tensor subspace.

    `t=None` means *the lifting `inner` collapsed against* -- the space's cached
    `bndvals`, or its one-dimensional equivalent. That is what makes
    `BoundaryForcing()` reproduce the vector `inner` folded into the forcing,
    which is in turn what lets a caller subtract the frozen copy back out and
    put a live one in its place.
    """
    if isinstance(space, BCGeneric):
        if rate:
            return space.bnd_vals_rate(t)
        return _frozen_bnd_vals(space) if t is None else space.bnd_vals(t)
    if rate:
        return space.lifting.rate(t)
    return space.bndvals if t is None else space.lifting(t)


def _block_is_transient(block: _BoundarySpec) -> bool:
    space = block.space
    if isinstance(space, BCGeneric):
        return space.bcs.has_time(space.system.base_time())
    return space.lifting.is_transient


def inner_boundary(
    expr: sp.Expr,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
    outer_sign: int = 1,
) -> BoundaryForcing | None:
    """Assemble only the boundary blocks of `expr`, without contracting them.

    The counterpart to `inner` for a lifting that moves: `inner` returns the
    same terms already collapsed against the space's current `bndvals`.

    Returns:
        A `BoundaryForcing`, or None when the form has no inhomogeneous
        boundary contribution.
    """
    # Runs the same assembly `inner` does, through the same context, so the
    # operators are the ones `inner` would have used -- there is one definition
    # of each contraction and both paths call it.
    context = _prepare_inner_context(expr, num_quad_points)
    blocks: list[_BoundaryBlock] = []
    context = replace(context, boundary_blocks=blocks)
    _assemble_inner_items(context, use_precomputed_matrices)
    if not blocks:
        return None
    test_leaf = context.test_space.leaf
    rank = RankTag.SCALAR if test_leaf is None else test_leaf.rank
    return BoundaryForcing(blocks, rank, outer_sign)


class SourceForcing(nnx.Module):
    """A source term's load vector as a function of time.

    The time dependence of a supported source factors out of the quadrature:
    each term is `g_k(t) * h_k(x)`, so `h_k` assembles once into a fixed vector
    and a call costs one scalar evaluation and one axpy per term. Traceable in
    `t`.
    """

    def __init__(
        self, vectors: Sequence[Array], funs: Sequence[ArrayFn], rank: RankTag
    ) -> None:
        # The assembled vectors are data -- they cross `_advance`'s jit boundary
        # as traced arguments, like the boundary operators do. The lambdified
        # time factors are ordinary Python callables and stay static.
        #
        # Replicated, not sharded, for the reason `BaseIntegrator.linear_forcing`
        # is: assembly places a load vector on the global spectral sharding, and
        # these are stored on the integrator, so they are reached through
        # `_advance`'s closure. JAX refuses to close over an array spanning
        # devices the process cannot address.
        self._vectors = nnx.data(tuple(replicate(v) for v in vectors))
        self._funs = nnx.static(tuple(funs))
        self._rank = nnx.static(rank)

    def __call__(self, t: float | Array) -> Array:
        """Return the source load vector at time `t`."""
        if self._rank != RankTag.SCALAR:
            raise NotImplementedError(
                "Deferred source forcing is implemented for scalar equations; "
                "assemble each component of a block system separately."
            )
        total: Array | None = None
        for vector, fun in zip(self._vectors, self._funs, strict=True):
            term = jnp.asarray(fun(t)) * jnp.asarray(vector)
            total = term if total is None else total + term
        assert total is not None
        return total


def _time_factor_of(b0: InnerResultDict, time: BaseTime) -> sp.Expr | None:
    """Return the term's time factor, or None when it has none.

    `separatevars` puts everything free of the coordinates into `coeff`, so a
    source that is separable in time arrives with `t` there and nowhere else.
    Anything else -- `t` riding inside a coordinate factor or a `multivar` --
    would need the quadrature itself re-evaluated at every stage, and is
    refused rather than silently mishandled.
    """
    coeff = sp.sympify(b0.get("coeff", 1))
    elsewhere = [
        (key, value)
        for key, value in b0.items()
        if key != "coeff" and sp.sympify(value).has(time)
    ]
    if elsewhere:
        where = ", ".join(f"{key}: {value}" for key, value in elsewhere)
        raise NotImplementedError(
            f"Source term with `{time}` inside a coordinate factor ({where}). "
            "Only sources separable in time are supported -- write the term as "
            f"a sum of `g({time}) * h(x)` products. `sp.expand_trig` turns a "
            "travelling wave into one; an expression like sqrt(x + t) has no "
            "such form."
        )
    if not coeff.has(time):
        return None
    # The factor is lambdified against `time` alone, so anything else left free
    # in it becomes an undefined name inside the generated function and surfaces
    # as a `TypeError` from within a trace. A frequency written as a bare
    # `sp.Symbol` rather than a number is the easy way to get here.
    stray = coeff.free_symbols - {time}
    if stray:
        names = ", ".join(sorted(str(sym) for sym in stray))
        raise ValueError(
            f"Source term's time factor `{coeff}` has free symbol(s) {names} "
            f"besides `{time}`. A source's time dependence is evaluated from "
            f"`{time}` alone, so every parameter in it must be a number."
        )
    return coeff


def inner_source(
    expr: sp.Expr,
    num_quad_points: int | tuple[int | None, ...] | None = None,
) -> SourceForcing | None:
    """Assemble the time-dependent linear forms of `expr` as a function of time.

    The counterpart to `inner` for a source that moves. `expr` must hold linear
    forms only -- the caller splits the time-dependent terms out of the weak
    form first, which is what keeps `inner`'s own path untouched.

    Returns:
        A `SourceForcing`, or None when no term depends on time.
    """
    context = _prepare_inner_context(expr, num_quad_points)
    if context.a_forms:
        raise ValueError(
            "inner_source expects an expression with no bilinear forms; "
            "operators are assembled once and cannot depend on time."
        )
    time = context.test_space.system.base_time()
    test_leaf = context.test_space.leaf
    rank: RankTag = RankTag.SCALAR if test_leaf is None else test_leaf.rank

    vectors: list[Array] = []
    funs: list[ArrayFn] = []
    for b0 in context.b_forms:
        factor = _time_factor_of(b0, time)
        if factor is None:
            continue
        # Assembled with the time factor replaced by one, through the same
        # `_assemble_linear_form` `inner` uses -- so vector spaces, global
        # indices, `multivar` and `jaxfunction` all keep working unchanged, and
        # a constant `g` reproduces `inner`'s own vector.
        steady = cast(InnerResultDict, {**b0, "coeff": sp.Integer(1)})
        item = _assemble_linear_form(steady, context)
        if item is None:  # pragma: no cover - a linear form always assembles one
            raise ValueError(f"Source term assembled no load vector: {b0}")
        vector = _finalize_inner_result(
            [], [item], context.test_space, context.trial_space, False, 1000
        )
        vectors.append(cast(Array, vector))
        funs.append(lambdify((time,), factor))

    if not vectors:
        return None
    return SourceForcing(vectors, funs, rank)


def inner_items(
    expr: sp.Expr,
    num_quad_points: int | tuple[int | None, ...] | None = None,
    use_precomputed_matrices: bool = True,
) -> InnerItems:
    r"""Assemble Galerkin inner products and return unsummed raw term lists.

    This is the raw-items counterpart to :func:`inner`: bilinear matrix terms
    and linear vector terms are returned before final summation or sparse
    conversion.
    """
    context = _prepare_inner_context(expr, num_quad_points)
    return _assemble_inner_items(context, use_precomputed_matrices)


def _prepare_inner_context(
    expr: sp.Expr,
    num_quad_points: int | tuple[int | None, ...] | None,
) -> _InnerContext:
    V, U = get_basisfunctions(expr)
    assert V is not None, "No TestFunction found in expression"
    V = unwrap_single_testfunction(V)
    assert _has_testspace(V), "TestFunction has no associated function space"
    test_space = cast(ComputationalSpaceType, V.functionspace)
    if isinstance(U, set):
        leaf = set(
            cast(
                CartesianTensorProductSpace | CartesianProductSpace,
                cast(TrialFunction, u).functionspace,
            ).leaf
            for u in U
        )
        assert len(leaf) == 1
        trial_space = leaf.pop()
    else:
        trial_space: TrialSpaceType | None = getattr(U, "functionspace", None)
    measure = test_space.system.sg
    allforms = split(expr * measure)
    a_forms = allforms["bilinear"]
    b_forms = allforms["linear"]
    all_linear = not (
        len(a_forms) > 0
        and jnp.any(
            jnp.array(["bilinear" in split_coeff(c0["coeff"]) for c0 in a_forms])
        )
    )
    num_quad_points = (
        num_quad_points if num_quad_points is not None else test_space.num_quad_points
    )

    return _InnerContext(
        test_space=test_space,
        trial_space=trial_space,
        a_forms=a_forms,
        b_forms=b_forms,
        num_quad_points=num_quad_points,
        all_linear=all_linear,
    )


def _assemble_inner_items(
    context: _InnerContext,
    use_precomputed_matrices: bool,
) -> InnerItems:
    aresults: list[BaseMatrix | GlobalMatrix] = []
    bresults: list[GlobalArray] = []

    for a0 in context.a_forms:
        aitem, bitem = _assemble_bilinear_form(a0, context, use_precomputed_matrices)
        if aitem is not None:
            aresults.append(aitem)
        if bitem is not None:
            bresults.append(bitem)

    for b0 in context.b_forms:
        bitem = _assemble_linear_form(b0, context)
        if bitem is not None:
            bresults.append(bitem)

    return aresults, bresults


def _form_factors(form: InnerResultDict) -> Iterator[sp.Expr]:
    for key, value in form.items():
        if key in ("coeff", "multivar", "jaxfunction"):
            continue
        assert isinstance(value, sp.Expr)
        yield value


def _linear_sign(all_linear: bool) -> int:
    return 1 if all_linear else -1


# The boundary block of a bilinear form contracts an assembled operator with the
# lifting coefficients. Shared by the collapsed path below, which does it at
# assembly time, and by `inner_boundary`, which keeps the operator so that a
# lifting that moves in time can be contracted again at every step. One
# definition, so the two cannot produce different numbers.


def _frozen_bnd_vals(uf: BCGeneric) -> Array:
    """Return the one-dimensional lifting as `inner` collapses it.

    The counterpart of reading `DirectSumTPS.bndvals`, which likewise holds the
    lifting as it stood when the space was built. Boundary data that varies in
    time is therefore frozen at zero here, exactly as it is there -- a caller
    who needs it to move wants `inner_boundary`, which keeps the block
    uncontracted.
    """
    time = uf.system.base_time()
    return uf.bnd_vals(0.0 if uf.bcs.has_time(time) else None)


def _contract_1d(z: Matrix | DiaMatrix, sign: int, fun: Array) -> Array:
    return sign * (z @ jnp.asarray(fun, dtype=z.dtype))


def _contract_multivar(Am: Array, sign: int, fun: Array) -> Array:
    return sign * jnp.einsum("ikjl,kl->ij", Am, fun)


def _contract_separable(
    mats_: list[Matrix | DiaMatrix], sign: int, fun: Array
) -> Array:
    return TPMatrix(mats_, sign) @ fun


def _quad_points_for_space(
    num_quad_points: _NumQuadPoints,
    space: OrthogonalSpace,
) -> int:
    if isinstance(num_quad_points, int):
        return num_quad_points
    return num_quad_points[space.system.base_scalars()[0]._id[0]]


def _assemble_bilinear_factor(
    ai: sp.Expr,
    scale: float | complex,
    is_multivar: bool,
    num_quad_points: _NumQuadPoints,
    use_precomputed_matrices: bool,
) -> tuple[_BilinearFactor, tuple[int, int], OrthogonalSpace]:
    v, u = get_basisfunctions(ai)
    assert v is not None and u is not None, (
        "Both test and trial functions required in bilinear form"
    )
    if isinstance(v, set) and isinstance(u, set):
        raise ValueError(
            f"Too many test/trial functions {(v, u)} found in bilinear form"
        )

    assert _has_testspace(v), "TestFunction has no associated function space"
    assert _has_functionspace(u), "TrialFunction has no associated function space"
    vf = v.functionspace
    uf = u.functionspace

    assert isinstance(vf, OrthogonalSpace)
    assert isinstance(uf, OrthogonalSpace)
    assert _has_globalindex(v)
    assert _has_globalindex(u)

    N = _quad_points_for_space(num_quad_points, vf)
    z = inner_bilinear(ai, vf, uf, scale, is_multivar, N, use_precomputed_matrices)
    return z, (v.global_index, u.global_index), uf


def _assemble_bilinear_form(
    a0: InnerResultDict, context: _InnerContext, use_precomputed_matrices: bool
) -> _InnerTerm:
    mats: list[_BilinearMat] = []
    coeffs = split_coeff(a0["coeff"])
    sc = coeffs.get("bilinear", 1)

    aresult: BaseMatrix | GlobalMatrix | None = None
    bresult: GlobalArray | None = None
    trial: list[OrthogonalSpace] = []
    has_bcs = False

    for ai in _form_factors(a0):
        is_multivar = "multivar" in a0 or "jaxfunction" in a0
        z, global_indices, uf = _assemble_bilinear_factor(
            ai, sc, is_multivar, context.num_quad_points, use_precomputed_matrices
        )
        trial.append(uf)
        if isinstance(uf, BCGeneric):
            has_bcs = True

        if isinstance(z, tuple):  # multivar
            mats.append((z, global_indices))
            sc = 1
            continue

        if z.size == 0:
            continue
        if isinstance(uf, BCGeneric) and context.test_space.dims == 1:
            sign = _linear_sign(context.all_linear)
            if context.boundary_blocks is not None:
                context.boundary_blocks.append(
                    _BoundaryBlock("1d", z, uf, None, sign, global_indices[0])
                )
                continue
            bresult = GlobalArray(
                global_indices[0], _contract_1d(z, sign, _frozen_bnd_vals(uf))
            )
            continue
        if "linear" in coeffs and context.test_space.dims == 1:
            sign = _linear_sign(context.all_linear)
            scale = coeffs["linear"].get("scale", 1) * sign
            bresult = GlobalArray(
                global_indices[0], scale * (z @ coeffs["linear"]["jaxcoeff"].array)
            )

        sc = 1
        mats.append((z, global_indices))

    if len(mats) == 0:
        return aresult, bresult

    if context.test_space.dims == 1 and "bilinear" in coeffs:
        assert isinstance(mats[0][0], Matrix | DiaMatrix)
        test_leaf = context.test_space.leaf
        if test_leaf is not None:
            # Inside a CartesianProductSpace: tag with block indices for BlockMatrix
            aresult = GlobalMatrix(mats[0][1], mats[0][0])
        else:
            aresult = mats[0][0]
        return aresult, bresult

    if isinstance(mats[0][0], tuple):
        return _assemble_multivar_bilinear_form(
            a0, coeffs, mats, trial, has_bcs, context
        )

    if len(mats) > 1:
        return _assemble_separable_bilinear_form(coeffs, mats, trial, has_bcs, context)

    return aresult, bresult


def _multivar_boundary_source(
    trial_space: TrialSpaceType | None,
    trial: list[OrthogonalSpace],
    gi: list[tuple[int, int]],
) -> tuple[DirectSumTPS, tuple[OrthogonalSpace, ...]]:
    """Return the space holding this boundary block's lifting, and its key."""
    assert isinstance(trial_space, DirectSumTPS | CartesianTensorProductSpace)
    if isinstance(trial_space, DirectSumTPS):
        return trial_space, tuple(trial)
    assert isinstance(trial_space, CartesianTensorProductSpace)
    dsspace = trial_space.flatten()[gi[1][1]]
    assert isinstance(dsspace, DirectSumTPS)
    return dsspace, tuple(trial)


def _multivar_boundary_values(
    trial_space: TrialSpaceType | None,
    trial: list[OrthogonalSpace],
    gi: list[tuple[int, int]],
) -> Array:
    space, key = _multivar_boundary_source(trial_space, trial, gi)
    return space.bndvals[key]


def _assemble_multivar_bilinear_form(
    a0: InnerResultDict,
    coeffs: CoeffDict,
    mats: list[_BilinearMat],
    trial: list[OrthogonalSpace],
    has_bcs: bool,
    context: _InnerContext,
) -> _InnerTerm:
    aresult: BaseMatrix | None = None
    bresult: GlobalArray | None = None
    assert len(mats) == 2
    assert isinstance(context.test_space, TensorProductSpace)
    mats_ = [
        cast(tuple[Array, Array], mats[0][0]),
        cast(tuple[Array, Array], mats[1][0]),
    ]
    gi = [m[1] for m in mats]

    scales = []

    if "multivar" in a0:
        scales.append(a0["multivar"])
    if "jaxfunction" in a0:
        scales.append(
            evaluate_jaxfunction_expr_quad(a0["jaxfunction"], N=context.num_quad_points)
        )
    Am = assemble_multivar(mats_, scales, context.test_space)
    if has_bcs:
        sign = _linear_sign(context.all_linear)
        space, key = _multivar_boundary_source(context.trial_space, trial, gi)
        if context.boundary_blocks is not None:
            context.boundary_blocks.append(
                _BoundaryBlock("multivar", Am, space, key, sign, gi[0][0])
            )
            return aresult, bresult
        res = _contract_multivar(Am, sign, space.bndvals[key])
        bresult = GlobalArray(gi[0][0], res)

    else:
        if "linear" in coeffs:
            sign = _linear_sign(context.all_linear)
            res = sign * jnp.einsum(
                "ikjl,kl->ij", Am, coeffs["linear"]["jaxcoeff"].array
            )
            bresult = GlobalArray(gi[0][0], res)

        if "bilinear" in coeffs:
            assert isinstance(context.trial_space, TensorProductSpace)
            aresult = TensorMatrix(Am)

    return aresult, bresult


def _separable_boundary_source(
    trial_space: TrialSpaceType | None,
    coeffs: CoeffDict,
    trial: list[OrthogonalSpace],
    gi: list[tuple[int, int]],
) -> tuple[DirectSumTPS, tuple[OrthogonalSpace, ...]]:
    """Return the space holding this boundary block's lifting, and its key."""
    if trial_space is not None:
        if isinstance(trial_space, DirectSumTPS):
            return trial_space, tuple(trial)
        if isinstance(trial_space, CartesianTensorProductSpace):
            dsspace = trial_space.flatten()[gi[1][1]]
            assert isinstance(dsspace, DirectSumTPS)
            return dsspace, tuple(trial)
        raise NotImplementedError(
            "BCs only implemented for TensorProductSpace"
            " and CartesianTensorProductSpace"
        )

    jfs = coeffs["linear"]["jaxcoeff"].functionspace
    assert isinstance(jfs, DirectSumTPS | CartesianTensorProductSpace)
    if isinstance(jfs, DirectSumTPS):
        return jfs, tuple(trial)
    dsspace = jfs.flatten()[gi[1][1]]
    assert isinstance(dsspace, DirectSumTPS)
    return dsspace, tuple(trial)


def _separable_boundary_values(
    trial_space: TrialSpaceType | None,
    coeffs: CoeffDict,
    trial: list[OrthogonalSpace],
    gi: list[tuple[int, int]],
) -> Array:
    space, key = _separable_boundary_source(trial_space, coeffs, trial, gi)
    return space.bndvals[key]


def _assemble_separable_bilinear_form(
    coeffs: CoeffDict,
    mats: list[_BilinearMat],
    trial: list[OrthogonalSpace],
    has_bcs: bool,
    context: _InnerContext,
) -> _InnerTerm:
    aresult: BaseMatrix | None = None
    bresult: GlobalArray | None = None
    mats_: list[Matrix | DiaMatrix] = [cast(Matrix | DiaMatrix, m[0]) for m in mats]
    gi = [m[1] for m in mats]

    assert isinstance(context.test_space, TensorProductSpace | VectorTensorProductSpace)
    if has_bcs:
        space, key = _separable_boundary_source(context.trial_space, coeffs, trial, gi)
        sign = _linear_sign(context.all_linear)
        if context.boundary_blocks is not None:
            context.boundary_blocks.append(
                _BoundaryBlock("separable", tuple(mats_), space, key, sign, gi[0][0])
            )
            return aresult, bresult
        res = _contract_separable(mats_, sign, space.bndvals[key])
        bresult = GlobalArray(gi[0][0], res)

    else:
        if "linear" in coeffs:
            sign = _linear_sign(context.all_linear)
            res = (
                sign
                * coeffs["linear"].get("scale", 1)
                * coeffs["linear"]["jaxcoeff"].array
            )
            for i, mat in enumerate(mats_):
                res = mat.matvec(res, axis=i)
            bresult = GlobalArray(gi[0][0], res)

        if "bilinear" in coeffs:
            assert isinstance(
                context.test_space, TensorProductSpace | VectorTensorProductSpace
            )
            assert isinstance(
                context.trial_space, TensorProductSpace | CartesianTensorProductSpace
            )
            aresult = TPMatrix(
                [cast(BaseMatrix, m[0]) for m in mats], 1, global_indices=gi[0]
            )

    return aresult, bresult


def _linear_form_scale(
    b0: InnerResultDict, has_bilinear_forms: bool
) -> float | complex:
    sc = sp.sympify(b0["coeff"])
    scale = float(sc) if sc.is_real else complex(sc)
    if has_bilinear_forms:
        scale = scale * (-1)
    return scale


def _assemble_linear_factor(
    bi: sp.Expr,
    scale: float | complex,
    is_multivar: bool,
    num_quad_points: _NumQuadPoints,
) -> tuple[Array | tuple[Array, ...], int]:
    v, _ = get_basisfunctions(bi)
    assert v is not None, "Test function required in linear form"
    assert _has_functionspace(v)
    vf = v.functionspace
    assert isinstance(vf, OrthogonalSpace)
    assert _has_globalindex(v)
    quads = _quad_points_for_space(num_quad_points, vf)
    z = inner_linear(bi, vf, scale, is_multivar, quads)
    return z, v.global_index


def _assemble_linear_form(
    b0: InnerResultDict, context: _InnerContext
) -> GlobalArray | None:
    scale = _linear_form_scale(b0, len(context.a_forms) > 0)
    bs: list[_LinearFactor] = []
    global_index = 0
    is_multivar = "multivar" in b0 or "jaxfunction" in b0
    quads = context.num_quad_points

    for bi in _form_factors(b0):
        z, global_index = _assemble_linear_factor(bi, scale, is_multivar, quads)
        scale = 1
        bs.append(z)

    test_space = context.test_space
    if isinstance(test_space, OrthogonalSpace):
        return GlobalArray(global_index, cast(Array, bs[0]))
    if (
        isinstance(test_space, TensorProductSpace | VectorTensorProductSpace)
        and len(test_space) == 2
    ):
        return _assemble_linear_tensor2d(b0, bs, test_space, quads, global_index)
    if isinstance(test_space, TensorProductSpace) and len(test_space) == 3:
        return _assemble_linear_tensor3d(b0, bs, test_space, quads, global_index)
    return None


def _assemble_linear_tensor2d(
    b0: InnerResultDict,
    bs: list[_LinearFactor],
    test_space: TensorProductSpace | VectorTensorProductSpace,
    num_quad_points: _NumQuadPoints,
    global_index: int,
) -> GlobalArray:
    if isinstance(bs[0], tuple):
        assert isinstance(num_quad_points, tuple)
        uj = jnp.array(1.0)
        if "multivar" in b0:
            s = test_space.system.base_scalars()
            uj *= lambdify(s, b0["multivar"], modules="jax")(
                *test_space.mesh(N=num_quad_points)
            )
        if "jaxfunction" in b0:
            uj *= evaluate_jaxfunction_expr_quad(b0["jaxfunction"], N=num_quad_points)
        if "jaxfunction" not in b0 and "multivar" not in b0:
            raise ValueError("Expected multivar or jaxfunction key in b0")
        res = bs[0][0].T @ uj @ bs[1][0]
        return GlobalArray(global_index, res)

    res = jnp.multiply.outer(cast(Array, bs[0]), cast(Array, bs[1]))
    return GlobalArray(global_index, res)


def _assemble_linear_tensor3d(
    b0: InnerResultDict,
    bs: list[_LinearFactor],
    test_space: TensorProductSpace | VectorTensorProductSpace,
    num_quad_points: _NumQuadPoints,
    global_index: int,
) -> GlobalArray:
    if isinstance(bs[0], tuple):
        assert isinstance(num_quad_points, tuple)
        if "multivar" in b0:
            s = test_space.system.base_scalars()
            uj = lambdify(s, b0["multivar"], modules="jax")(
                *test_space.mesh(N=num_quad_points)
            )
        elif "jaxfunction" in b0:
            uj = evaluate_jaxfunction_expr_quad(b0["jaxfunction"], N=num_quad_points)
        else:
            raise ValueError("Expected multivar or jaxfunction key in b0")
        res = jnp.einsum("il,jm,kn,ijk->lmn", bs[0][0], bs[1][0], bs[2][0], uj)
        return GlobalArray(global_index, res)

    res = jnp.multiply.outer(
        jnp.multiply.outer(cast(Array, bs[0]), cast(Array, bs[1])), cast(Array, bs[2])
    )
    return GlobalArray(global_index, res)


def _finalize_inner_result(
    aresults: list[BaseMatrix | GlobalMatrix],
    bresults: list[GlobalArray],
    test_space: ComputationalSpaceType,
    trial_space: TrialSpaceType | None,
    sparse: bool,
    sparse_tol: int,
) -> GalerkinAssembledForm:
    """Finalize assembly results (sum terms, optional sparsify).

    Args:
        aresults: List of bilinear matrices (dense or structured holders).
        bresults: List of load vectors / tensors.
        dims: Spatial dimension (1 => sum; >1 keep tensor structure).
        sparse: If True, sparsify (only when applicable).
        sparse_tol: Zero tolerance for sparsification.

    Returns:
        Matrix, vector, (matrix, vector), tensor-product operator, or None.
    """
    assert test_space is not None
    dims: int = test_space.dims
    test_leaf = test_space.leaf
    rank: RankTag = RankTag.SCALAR if test_leaf is None else test_leaf.rank

    bresult: Array | BlockArray | None = None

    if len(bresults) > 0:
        if rank == RankTag.SCALAR:
            bresult = jnp.sum(jnp.array([d.data for d in bresults]), axis=0)
        else:
            bresult = BlockArray(
                cast(CartesianTensorProductSpace, test_leaf), indexed_arrays=bresults
            )

    aresult: BaseMatrix | None = None

    if dims == 1:
        if len(aresults) > 0:
            if test_leaf is not None:
                # 1D CartesianProductSpace bilinear forms
                iresults = cast(list[GlobalMatrix], aresults)
                indexed_mats: list[GlobalMatrix] = []
                for im in iresults:
                    m = (
                        im.data.tosparse(tol=sparse_tol)
                        if sparse
                        else im.data.to_matrix()
                    )
                    indexed_mats.append(GlobalMatrix(im.global_indices, m))
                assert trial_space is not None
                trial_leaf = getattr(trial_space, "leaf", trial_space)
                aresult = BlockMatrix(
                    indexed_mats,
                    cast(CartesianProductSpace, test_leaf),
                    cast(CartesianProductSpace, trial_leaf),
                )
            else:
                plain_results = cast(list[BaseMatrix], aresults)
                amats: list[Matrix | DiaMatrix] = []
                for a in plain_results:
                    amats.append(
                        a.tosparse(tol=sparse_tol) if sparse else a.to_matrix()
                    )
                aresult = sum(amats[1:], amats[0])

        if aresult is None:
            return cast(Array, bresult)
        if bresult is None:
            return aresult

        return aresult, cast(Array, bresult)

    assert not isinstance(test_space, OrthogonalSpace)
    if len(jax.devices()) > 1 and bresult is not None:
        # `place`, not a bare `device_put`: a size the mesh cannot divide is a
        # reason to leave the array alone, not to refuse to assemble it.
        bresult = place(cast(Array, bresult), test_space._spectral_sharding)

    if len(aresults) > 0:
        # aresults is an empty list or a list of TPMatrix/TensorMatrix objects.
        assert trial_space is not None
        assert not isinstance(trial_space, OrthogonalSpace | DirectSum)

        if all(isinstance(a, TPMatrix) for a in aresults):
            tpresults = cast(list[TPMatrix], aresults)
            for a0 in tpresults:
                if sparse:
                    a0.mats = nnx.List(mat.tosparse(tol=sparse_tol) for mat in a0.mats)
                else:
                    a0.mats = nnx.List(mat.to_matrix() for mat in a0.mats)

            if rank == RankTag.SCALAR:
                aresult = tpresults[0] if len(tpresults) == 1 else TPMatrices(tpresults)
            else:
                aresult = BlockMatrix(
                    tpresults,
                    cast(CartesianTensorProductSpace, test_space.leaf),
                    cast(CartesianTensorProductSpace, trial_space.leaf),
                )

        elif all(isinstance(a, TensorMatrix) for a in aresults):
            if rank != RankTag.SCALAR:
                raise NotImplementedError("Rank >0 TensorMatrix not implemented")
            tensor_results = cast(list[TensorMatrix], aresults)
            aresult = sum(tensor_results[1:], tensor_results[0])

        else:
            raise ValueError("Inconsistent matrix types in aresults")

    if aresult is None:
        if bresult is None:
            raise ValueError("No bilinear or linear forms found in expression")
        return bresult
    if bresult is None:
        return aresult
    return aresult, bresult


@overload
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: Literal[False],
    num_quad_points: int,
    use_precomputed_matrices: bool,
) -> Matrix | DiaMatrix: ...
@overload
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: Literal[True],
    num_quad_points: int,
    use_precomputed_matrices: bool,
) -> tuple[Array, Array]: ...
def inner_bilinear(
    ai: sp.Expr,
    v: OrthogonalSpace,
    u: OrthogonalSpace,
    sc: float | complex,
    multivar: bool,
    num_quad_points: int,
    use_precomputed_matrices: bool,
) -> Matrix | DiaMatrix | tuple[Array, Array]:
    """Assemble single bilinear form contribution term.

    Detects derivative orders on test/trial factors, applies optional
    symbolic coefficient (possibly sampled at quadrature points) and
    returns dense matrix or tuple (multivar separated factors).

    Args:
        ai: SymPy sub-expression containing basis factors.
        v: Test function space (Orthogonal or Composite).
        u: Trial function space.
        sc: Scalar bilinear coefficient (after linear split).
        multivar: True if coefficient not separable (handled upstream).
        num_quad_points: Number of quadrature points.
    Returns:
        Matrix/DiaMatrix, or (Pi, Pj) tuple for multivar separation.
    """
    vo = v.orthogonal
    uo = u.orthogonal
    N = num_quad_points
    xj, wj = vo.quad_points_and_weights(N=N)
    df = float(vo.domain_factor)
    i, j = 0, 0
    scale = jnp.array([sc])
    poly_scale: int = 0
    x = vo.system.base_scalars()[0]

    for aii in ai.args:
        found_basis = False
        for p in sp.core.traversal.preorder_traversal(aii):
            if get_arg(p) in (ArgumentTag.TEST, ArgumentTag.TRIAL):
                found_basis = True
                break
        if found_basis:
            if isinstance(aii, sp.Derivative):
                arg = get_arg(aii.args[0])
                if arg is ArgumentTag.TEST:
                    assert i == 0
                    i = int(aii.derivative_count)
                elif arg is ArgumentTag.TRIAL:
                    assert j == 0
                    j = int(aii.derivative_count)
            continue

        jaxfunction = get_jaxfunctions(aii)
        if len(jaxfunction) >= 1:
            scale *= evaluate_jaxfunction_expr_quad(aii, N=N)
            continue

        if len(aii.free_symbols) > 0:
            s = aii.free_symbols.pop()
            assert s == x
            aii = cast(sp.Expr, aii)
            if aii.is_polynomial(s) and sp.degree(aii, s) > 0:
                # just store degree, because any numeric scale will already be in scale.
                poly_scale = int(sp.degree(aii, s))
            else:
                scale *= lambdify(s, uo.map_expr_true_domain(aii))(xj)

        else:
            scale *= float(aii)  # ty:ignore[invalid-argument-type]

    z: DiaMatrix | Matrix | None = None
    if len(scale) == 1 and use_precomputed_matrices and not multivar:
        if isinstance(v, Composite):
            z = v.matrices(i, (u, j), q=poly_scale, scale=scale.item())
            if z is not None:
                return z
        z = vo.matrices(i, (uo, j), q=poly_scale, scale=scale.item())

    if z is None:
        if poly_scale != 0:
            scale *= lambdify(x, vo.map_expr_true_domain(x**poly_scale))(xj)

        w = wj * df ** (i + j - 1) * scale
        Pi = vo.evaluate_basis_derivative(xj, k=i)
        Pj = uo.evaluate_basis_derivative(xj, k=j)

        if multivar:
            multi: Array = v.apply_stencil_right(w[:, None] * Pi)
            multj: Array = u.apply_stencil_right(jnp.conj(Pj))
            return multi, multj

        z = Matrix(matmat(Pi.T * w[None, :], jnp.conj(Pj)))

    if isinstance(v, Composite) and u == v:
        z = v.apply_stencil_galerkin(z)
    elif isinstance(v, Composite) and isinstance(u, Composite):
        z = v.apply_stencils_petrovgalerkin(z, u.ST)
    elif isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    elif isinstance(u, Composite):
        z = u.apply_stencil_right(z)

    return z


def inner_linear(
    bi: sp.Expr,
    v: OrthogonalSpace,
    sc: float | complex,
    multivar: bool,
    num_quad_points: int,
) -> Array | tuple[Array]:
    """Assemble single linear form contribution.

    Args:
        bi: SymPy term with one test function (possibly derivative).
        v: Test function space.
        sc: Scalar coefficient (sign-adjusted if from a(u,v)-L(v)).
        global_index: Global index for vector-valued spaces.
        multivar: True if non-separable scaling (return tuple form).
        num_quad_points: Number of quadrature points to use
            (de-aliasing for nonlinear / non-constant coeffs).

    Returns:
        Vector (1D), tuple (Pi,) for multivar, or projected result.
    """
    vo = v.orthogonal
    N = num_quad_points
    xj, wj = vo.quad_points_and_weights(N=N)
    df = float(vo.domain_factor)
    i = 0
    uj = jnp.array([sc])  # incorporate scalar coefficient into first vector

    if get_arg(bi) is ArgumentTag.TEST:
        pass
    elif isinstance(bi, sp.Derivative):
        i = int(bi.derivative_count)
    else:
        for bii in bi.args:
            found_basis = False
            for p in sp.core.traversal.preorder_traversal(bii):
                if get_arg(p) is ArgumentTag.TEST:
                    found_basis = True
                    break
            if found_basis:
                if isinstance(bii, sp.Derivative):
                    assert i == 0
                    i = int(bii.derivative_count)
                continue

            jaxfunction = get_jaxfunctions(bii)
            if len(jaxfunction) >= 1:
                for jaxf in jaxfunction:
                    assert jaxf.functionspace.orthogonal.__class__ == vo.__class__, (
                        "JAXFunction space must match test function space"
                    )
                uj *= evaluate_jaxfunction_expr_quad(bii, N=N)
                continue
            # bii contains coordinates in the domain of v, e.g., (r, theta) for polar
            # Need to compute bii as bii(x(X)), since we use quadrature points
            if len(bii.free_symbols) > 0:
                s = bii.free_symbols.pop()
                bii = cast(sp.Expr, bii)
                uj *= lambdify(s, vo.map_expr_true_domain(bii))(xj)
            else:
                uj *= float(bii)  # ty:ignore[invalid-argument-type]

    Pi = vo.evaluate_basis_derivative(xj, k=i)
    w = wj * df ** (i - 1)  # Account for domain different from reference
    if multivar:
        return (v.apply_stencil_right((uj * w)[:, None] * jnp.conj(Pi)),)

    z = matmat(uj * w, jnp.conj(Pi))
    if isinstance(v, Composite):
        z = v.apply_stencil_left(z)
    return z


def contains_sympy_symbols(obj: object) -> TypeGuard[sp.Expr]:
    return len(sp.sympify(obj).free_symbols) > 0


def assemble_multivar(
    mats: list[tuple[Array, Array]],
    scale: sp.Expr | Array | list[sp.Expr | Array],
    test_space: TensorProductSpace,
) -> Array:
    """Contract separated multivariable factors into global matrix.

    Handles expressions like sqrt(x+y)*u*v where coefficient cannot be
    separated into pure x / y factors. Performs batched contraction.

    Args:
        mats: List [(P0,P1),(P2,P3)] of separated directional factors.
        scale: SymPy expression or numeric scalar/array.
        test_space: Tensor product space (for mesh / variable order).

    Returns:
        Dense matrix of shape (i, k, j, l) assembled from factors.
    """
    P0, P1 = mats[0]
    P2, P3 = mats[1]
    sci = jnp.array([1.0])
    for sc in scale if isinstance(scale, list) else [scale]:
        if not isinstance(sc, Array) and contains_sympy_symbols(sc):
            s = test_space.system.base_scalars()
            xj = test_space.mesh(N=(P0.shape[0], P1.shape[0]))
            sc = lambdify(s, sc, modules="jax")(*xj)
        elif isinstance(sc, float | complex):
            sc = jnp.ones((P0.shape[0], P1.shape[0])) * sc
        else:
            sc = cast(Array, sc)
        sci = sci * sc

    a = jnp.einsum("pi,pk,qj,ql,pq->ikjl", P0, P1, P2, P3, sci)
    return a


@overload
def _forward_at(V: ScalarSpaceType, uj: Array, t: float | Array | None) -> Array: ...
@overload
def _forward_at(
    V: FunctionSpaceType, uj: Array, t: float | Array | None
) -> Array | tuple[Array, ...]: ...
def _forward_at(
    V: FunctionSpaceType, uj: Array, t: float | Array | None
) -> Array | tuple[Array, ...]:
    """Forward transform, carrying `t` only to a space that lifts boundary data."""
    if t is None or not isinstance(V, DirectSum | DirectSumTPS):
        return V.forward(uj)
    return V.forward(uj, t=t)


def project1D(
    ue: sp.Expr,
    V: OrthogonalSpace | Composite | DirectSum,
    t: float | Array | None = None,
) -> Array:
    """Project scalar expression ue onto 1D space V.

    Args:
        ue: SymPy expression in physical coordinate.
        V: Orthogonal / Composite / DirectSum space.
        t: Time at which to evaluate `ue` and, when `V` is a direct sum, its own
            boundary lifting. `None` means neither depends on time.

    Returns:
        Coefficient vector uh.
    """
    if len(get_jaxfunctions(ue)) == 0:
        args = V.system.base_scalars()
        mesh = (V.mesh(),)
        if t is not None:
            args = (V.system.base_time(), *args)
            mesh = (t, *mesh)
        uj = lambdify(args, ue)(*mesh)
        uj = jnp.broadcast_to(uj, V.num_quad_points)
        return _forward_at(V, uj, t)

    u = TrialFunction(V)
    v = TestFunction(V)
    M, b = inner(v * (u - ue), kind=InnerKind.SYSTEM)
    uh = M.solve(b)
    return uh


@overload
def project(
    ue: sp.Tuple,
    V: CartesianProductSpace,
    t: float | Array | None = None,
) -> tuple[Array, ...]: ...
@overload
def project(
    ue: sp.Expr | sp.Tuple,
    V: CartesianTensorProductSpace,
    t: float | Array | None = None,
) -> tuple[Array, ...]: ...
@overload
def project(
    ue: sp.Expr, V: ScalarSpaceType, t: float | Array | None = None
) -> Array: ...
def project(
    ue: sp.Expr | sp.Tuple,
    V: FunctionSpaceType,
    t: float | Array | None = None,
) -> Array | tuple[Array, ...]:
    """Project expression onto (possibly tensor) space V.

    Args:
        ue: SymPy expression.
        V: Function space (may be tensor product/direct sum).
        t: Time at which to evaluate `ue` and, when `V` lifts boundary data, its
            own lifting. `None` means neither depends on time. Only supported
            for scalar spaces, which is all the boundary lifting needs.

    Returns:
        Coefficient array shaped to V.num_dofs.
    """
    from jaxfun.operators import Dot

    if V.dims == 1:
        if isinstance(V, CartesianProductSpace):
            assert isinstance(ue, sp.Tuple) and len(ue) == V.num_components
            spaces = V.flatten()
            return tuple(
                project(cast(sp.Expr, uei), cast(ScalarSpaceType, spaces[i]))
                for i, uei in enumerate(ue)
            )
        assert isinstance(V, OrthogonalSpace | DirectSum)
        assert isinstance(ue, sp.Expr)
        return project1D(ue, V, t)

    if len(get_jaxfunctions(ue if isinstance(ue, sp.Expr) else sum(ue))) == 0:
        assert not isinstance(V, OrthogonalSpace | DirectSum | CartesianProductSpace)
        if V.rank == RankTag.SCALAR:
            assert isinstance(ue, sp.Expr)
            args = V.system.base_scalars()
            mesh = V.mesh()
            if t is not None:
                args = (V.system.base_time(), *args)
                mesh = (t, *mesh)
            uj = lambdify(args, ue)(*mesh)
            uj = jnp.broadcast_to(uj, V.num_quad_points)
        else:
            assert t is None, "`t` is only supported for scalar spaces"
            s = V.system.base_scalars()
            bv = V.system.base_vectors()
            if V.rank == RankTag.VECTOR:  # VectorTensorProductSpace
                assert isinstance(ue, sp.Expr)
                uj = (lambdify(s, Dot(ue, n).doit())(*V.mesh()) for n in bv)
            else:
                assert isinstance(V, CartesianTensorProductSpace)
                assert isinstance(ue, sp.Tuple) and len(ue) == V.num_components
                uj = (lambdify(s, (uei).doit())(*V.mesh()) for uei in ue)
            uj = jnp.stack([jnp.broadcast_to(ui, V.num_quad_points) for ui in uj])
        return _forward_at(V, place(uj, V._physical_sharding), t)

    u = TrialFunction(V)
    v = TestFunction(V)
    if V.rank == RankTag.SCALAR:
        A, b = inner(v * (u - ue), kind=InnerKind.SYSTEM)
        uh = A.solve(b)

    elif V.rank == RankTag.VECTOR:
        assert isinstance(V, VectorTensorProductSpace)
        A, b = inner(Dot(v, (u - ue)), kind=InnerKind.SYSTEM)
        uh = A.solve(b)

    else:
        assert isinstance(V, CartesianTensorProductSpace)
        assert isinstance(ue, sp.Tuple) and len(ue) == V.num_components
        spaces = V.flatten()
        uh = tuple(project(cast(sp.Expr, uei), spaces[i]) for i, uei in enumerate(ue))

    return uh
