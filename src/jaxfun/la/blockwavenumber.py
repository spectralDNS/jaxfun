"""Factored solves of block (multi-field) tensor-product systems.

A coupled system over a mixed space -- velocity, pressure and temperature, say
-- assembles into a `BlockMatrix` whose terms are `TPMatrix` objects tagged with
the (test, trial) component they couple. `block_lu_factor` factorizes such an
operator once and returns a solver acting on one coefficient array per component.

When every component shares the same Fourier axes and at most one non-periodic
axis, the system decouples into one small block system per Fourier wavenumber.
`BlockWavenumberSolver` factorizes those dense, and `BlockBandedWavenumberSolver`
banded, after a reordering that interleaves the fields; both pivot, which the
zero diagonal of a saddle-point (pressure) block needs. Fourier along every axis,
each block only couples the components, and `BlockFourierSolver` inverts it. On
several devices the
wavenumbers are split across them as the state is, and each device assembles,
factorizes and solves only its own. Every other layout falls back to one global
sparse LU of the flattened system.

Fields that enter the others' equations one way only -- a temperature that drives
the momentum equation through buoyancy but is itself advected explicitly, say --
make the operator block triangular. `BlockTriangularSolver` then solves the
groups of fields one after the other, each with one of the solvers above.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from flax import nnx
from jax import Array, shard_map
from jax.sharding import PartitionSpec

from jaxfun.la.blockmatrix import BlockMatrix
from jaxfun.la.diamatrix import DiaMatrix, _lu_banded_kernel, _pivoted_forward
from jaxfun.la.matrixprotocol import BaseMatrix, SolverNotApplicable
from jaxfun.la.operators import GlobalMatrix
from jaxfun.la.tpmatrix import (
    TPMatricesWavenumberSolver,
    TPMatrix,
    _make_wavenumber_solve,
    _prune_zero_diagonals,
    tpmats_wavenumber_factor,
)

if TYPE_CHECKING:
    from jaxfun.galerkin import CartesianProductSpace, CartesianTensorProductSpace

    type BlockSpace = CartesianTensorProductSpace | CartesianProductSpace

type Pin = tuple[int, tuple[int, ...]]
type BlockTerm = TPMatrix | GlobalMatrix

_WAVENUMBER_METHODS = frozenset({"auto", "dense", "rcm", "banded"})

# `shard_map` specs: split over the wavenumbers, or the same on every device.
_SPLIT, _SAME = PartitionSpec("k"), PartitionSpec()


def block_lu_factor(
    mats: BlockMatrix | Sequence[BlockTerm],
    test_space: BlockSpace | None = None,
    trial_space: BlockSpace | None = None,
    pins: Sequence[Pin] = (),
    **solve_options: Any,
) -> (
    BlockFourierSolver
    | BlockWavenumberSolver
    | BlockBandedWavenumberSolver
    | BlockTriangularSolver
    | BlockGlobalSolver
):
    """Factorize a block operator for repeated solves.

    When some fields enter the others' equations but not the other way round,
    the operator is block triangular, and it is factorized and solved group by
    group; see `BlockTriangularSolver`.

    Args:
        mats: The operator, either a `BlockMatrix` or its terms, each tagged
            with ``global_indices = (test component, trial component)``, as in
            ``BlockMatrix.mats``.
        test_space: Mixed space of the rows (the test functions). Needed only
            for terms; a `BlockMatrix` brings its own.
        trial_space: Mixed space of the columns (the unknowns). Needed only for
            terms; a `BlockMatrix` brings its own.
        pins: ``(component, index)`` pairs. The row of that test component and
            coefficient index is replaced by an identity row on the same trial
            component and index, which fixes that unknown to zero. Use it for a
            nullspace, such as the constant pressure mode. A row and column that
            are both identically zero at the same component and index are
            pinned automatically.
        **solve_options: ``method`` picks how each wavenumber is factorized:
            ``"dense"``, ``"rcm"`` (banded, after a reverse Cuthill-McKee
            reordering; ``"banded"`` is the same) or ``"auto"``, the default,
            which takes the banded factorization when its band is narrow. A
            system that is Fourier along every axis is inverted per wavenumber
            whatever the method, and a field solved on its own takes the scalar
            wavenumber solver. Without Fourier structure the global sparse
            solve takes ``method`` and ``auto_threshold`` as
            `DiaMatrix.lu_solve` does.

    Returns:
        A solver whose ``solve`` maps a tuple of right-hand sides, one per test
        component, to a tuple of solutions, one per trial component.
    """
    terms, test_space, trial_space = _unpack(mats, test_space, trial_space)
    # A 1D mixed space assembles `GlobalMatrix` blocks; as one-axis tensor
    # products they take the same per-wavenumber path.
    tpmats = [
        TPMatrix([m.data], 1, global_indices=m.global_indices)
        if isinstance(m, GlobalMatrix)
        else m
        for m in terms
    ]
    test_dofs, trial_dofs = _num_dofs(test_space), _num_dofs(trial_space)
    method = solve_options.get("method", "auto")
    groups = _triangular_groups(tpmats, test_dofs, trial_dofs)
    try:
        if len(groups) > 1:
            return BlockTriangularSolver(
                tpmats, test_dofs, trial_dofs, pins, groups, method
            )
        system = _WavenumberSystem(tpmats, test_dofs, trial_dofs, pins)
        return _wavenumber_solver(system, method)
    except SolverNotApplicable:
        return BlockGlobalSolver(mats, test_space, trial_space, pins, **solve_options)


def _wavenumber_solver(
    system: _WavenumberSystem, method: str
) -> BlockFourierSolver | BlockWavenumberSolver | BlockBandedWavenumberSolver:
    """Return the per-wavenumber solver `method` picks for ``system``."""
    if method not in _WAVENUMBER_METHODS:
        raise ValueError(
            f"Unsupported method {method!r} for a per-wavenumber block solve; "
            f"use one of {sorted(_WAVENUMBER_METHODS)}."
        )
    if system.poly_axis is None:
        return BlockFourierSolver(system)
    if method == "dense" or (method == "auto" and not system.band_is_narrow):
        return BlockWavenumberSolver(system)
    return BlockBandedWavenumberSolver(system)


def _unpack(
    mats: BlockMatrix | Sequence[BlockTerm],
    test_space: BlockSpace | None,
    trial_space: BlockSpace | None,
) -> tuple[Sequence[BlockTerm], BlockSpace, BlockSpace]:
    """Return the terms and the (test, trial) spaces of a block operator."""
    if isinstance(mats, BlockMatrix):
        return list(mats.mats), mats.test_space, mats.trial_space
    if test_space is None or trial_space is None:
        raise TypeError("Block terms need their test and trial spaces.")
    return mats, test_space, trial_space


def _num_dofs(space: BlockSpace) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(n) for n in nd) for nd in space.num_dofs)


def _check_pins(
    pins: Sequence[Pin],
    test_dofs: tuple[tuple[int, ...], ...],
    trial_dofs: tuple[tuple[int, ...], ...],
) -> None:
    for c, index in pins:
        if test_dofs[c] != trial_dofs[c]:
            raise ValueError(
                f"Cannot pin component {c}: its test and trial spaces have "
                f"different shapes, {test_dofs[c]} and {trial_dofs[c]}."
            )
        if len(index) != len(test_dofs[c]) or any(
            not 0 <= i < n for i, n in zip(index, test_dofs[c], strict=True)
        ):
            raise IndexError(
                f"Pin index {index} out of range for component {c} with shape "
                f"{test_dofs[c]}."
            )


def _triangular_groups(
    mats: Sequence[TPMatrix],
    test_dofs: tuple[tuple[int, ...], ...],
    trial_dofs: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the components in groups that can be solved one after another.

    Equation c (test component) is taken to determine unknown c (trial
    component). The groups are the strongly connected components of "equation
    i holds unknown j", in an order where each group's equations hold only its
    own unknowns and those of earlier groups. It is one group, the whole system,
    when the components cannot be paired like that, when they are all coupled
    both ways, or when a group of one component has no term in its own unknown.
    """
    n = len(test_dofs)
    whole = (tuple(range(n)),)
    if test_dofs != trial_dofs:
        return whole
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    holds = np.zeros((n, n), dtype=bool)
    for tp in mats:
        i, j = (int(c) for c in tp.global_indices)
        holds[i, j] = True
    count, label = connected_components(
        csr_matrix(holds), directed=True, connection="strong"
    )
    groups = [tuple(int(c) for c in np.nonzero(label == g)[0]) for g in range(count)]
    if count == 1 or any(len(g) == 1 and not holds[g[0], g[0]] for g in groups):
        return whole
    needs = [
        {int(label[j]) for c in group for j in np.nonzero(holds[c])[0]} - {g}
        for g, group in enumerate(groups)
    ]
    order: list[int] = []
    while len(order) < count:
        ready = [g for g in range(count) if g not in order and needs[g] <= set(order)]
        order.append(min(ready, key=lambda g: groups[g][0]))
    return tuple(groups[g] for g in order)


class _WavenumberSystem:
    """A block operator that splits into one block system per Fourier wavenumber.

    Every axis but (at most) one is Fourier, so every term is diagonal there and
    the system at wavenumber k is

    .. math::

        A_k = \\sum_t s_t \\Bigl(\\prod_{a \\in \\text{Fourier}} F_t^{(a)}[k_a]
        \\Bigr) P_t,

    with each term's polynomial factor :math:`P_t` placed in its (test, trial)
    block. The unknowns of all components are concatenated along the polynomial
    axis, so :math:`A_k` has size :math:`n = \\sum_c n_c` where :math:`n_c` is the
    polynomial size of component :math:`c` (1 when there is no polynomial axis).

    Raises:
        SolverNotApplicable: The components do not share their Fourier axes, or
            there is more than one non-periodic axis.
    """

    def __init__(
        self,
        mats: Sequence[TPMatrix],
        test_dofs: tuple[tuple[int, ...], ...],
        trial_dofs: tuple[tuple[int, ...], ...],
        pins: Sequence[Pin],
    ) -> None:
        if not all(isinstance(m, TPMatrix) for m in mats):
            raise SolverNotApplicable("Wavenumber solver needs TPMatrix terms")
        ndim = len(test_dofs[0])
        if any(len(nd) != ndim for nd in test_dofs + trial_dofs):
            raise SolverNotApplicable("Components differ in dimension")
        _check_pins(pins, test_dofs, trial_dofs)

        def diagonal(tp: TPMatrix, axis: int) -> Array | None:
            mat = tp.mats[axis]
            if mat.shape[0] != mat.shape[1]:
                return None
            d = mat.diagonal_or_none()
            return None if d is None or d.ndim != 1 else d

        fourier = tuple(
            a for a in range(ndim) if all(diagonal(tp, a) is not None for tp in mats)
        )
        poly = tuple(a for a in range(ndim) if a not in fourier)
        if len(poly) > 1:
            raise SolverNotApplicable(
                f"Wavenumber solver needs at most one non-diagonal axis, got {poly}"
            )
        fourier_shape = tuple(test_dofs[0][a] for a in fourier)
        if any(
            tuple(nd[a] for a in fourier) != fourier_shape
            for nd in test_dofs + trial_dofs
        ):
            raise SolverNotApplicable("Components do not share their Fourier shape")

        poly_axis = poly[0] if poly else None

        def sizes(dofs: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
            return tuple(1 if poly_axis is None else nd[poly_axis] for nd in dofs)

        row_sizes, col_sizes = sizes(test_dofs), sizes(trial_dofs)
        if sum(row_sizes) != sum(col_sizes):
            raise ValueError(
                f"Block system is not square per wavenumber: {sum(row_sizes)} rows "
                f"and {sum(col_sizes)} columns."
            )
        self.fourier = fourier
        self.poly_axis = poly_axis
        self.fourier_shape = fourier_shape
        self.n_F = int(np.prod(fourier_shape)) if fourier_shape else 1
        self.n = sum(row_sizes)
        self.row_sizes, self.col_sizes = row_sizes, col_sizes
        self.row_starts = tuple(int(s) for s in np.cumsum((0,) + row_sizes[:-1]))
        self.col_starts = tuple(int(s) for s in np.cumsum((0,) + col_sizes[:-1]))
        self.trial_dofs = trial_dofs
        self.dtype = _working_dtype(mats)
        # The wavenumbers are split over the devices as the state is (see
        # `jaxfun.sharding.state_sharding`): along axis 0, when that is a Fourier
        # axis the devices divide. Otherwise every device holds them all.
        n_dev = jax.device_count()
        self.sharded = (
            n_dev > 1
            and ndim >= 2
            and fourier[:1] == (0,)
            and fourier_shape[0] % n_dev == 0
        )

        # Each term as (test component, trial component, weight per wavenumber,
        # polynomial factor). The weight is its coefficient times the product of
        # its Fourier diagonals, flattened C-order over the Fourier axes -- the
        # order the solvers lay the wavenumbers out in.
        self.terms: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for tp in mats:
            w = np.asarray(tp.coefficient, dtype=complex).reshape(1)
            for a in fourier:
                w = np.outer(w, np.asarray(diagonal(tp, a))).ravel()
            P = (
                np.ones((1, 1))
                if poly_axis is None
                else np.asarray(tp.mats[poly_axis].todense())
            )
            bi, bj = (int(i) for i in tp.global_indices)
            self.terms.append((bi, bj, w, P))
        self.pinned = self._pins(pins, test_dofs, trial_dofs)

    def _pins(
        self,
        pins: Sequence[Pin],
        test_dofs: tuple[tuple[int, ...], ...],
        trial_dofs: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, int, int], ...]:
        """Return the pinned ``(wavenumber, row, column)``, explicit and automatic.

        A row and a column that are both structurally zero leave that unknown
        undetermined and that equation empty. The constant pressure mode at
        k = 0 is one; so is it at a Nyquist wavenumber, whose odd derivatives
        Fourier eliminates.
        """

        def flat(index: tuple[int, ...]) -> tuple[int, int]:
            k = (
                int(
                    np.ravel_multi_index(
                        tuple(index[a] for a in self.fourier), self.fourier_shape
                    )
                )
                if self.fourier
                else 0
            )
            return k, 0 if self.poly_axis is None else index[self.poly_axis]

        pinned: set[tuple[int, int, int]] = set()
        for c, index in pins:
            k, j = flat(index)
            pinned.add((k, self.row_starts[c] + j, self.col_starts[c] + j))
        row_abs = np.zeros((self.n_F, self.n))
        col_abs = np.zeros((self.n_F, self.n))
        for bi, bj, w, P in self.terms:
            r0, c0 = self.row_starts[bi], self.col_starts[bj]
            row_abs[:, r0 : r0 + P.shape[0]] += np.outer(np.abs(w), np.abs(P).sum(1))
            col_abs[:, c0 : c0 + P.shape[1]] += np.outer(np.abs(w), np.abs(P).sum(0))
        for c, (r0, c0, n) in enumerate(
            zip(self.row_starts, self.col_starts, self.row_sizes, strict=True)
        ):
            if test_dofs[c] != trial_dofs[c]:
                continue
            empty = (row_abs[:, r0 : r0 + n] == 0) & (col_abs[:, c0 : c0 + n] == 0)
            pinned.update(
                (int(k), r0 + int(j), c0 + int(j)) for k, j in zip(*np.nonzero(empty))
            )
        return tuple(sorted(pinned))

    def phases(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return row and column phases that make every :math:`A_k` real, or None.

        A Fourier derivative makes a block's weights imaginary (ik) where an
        even one leaves them real (-k^2), so with phases :math:`i^{\\psi}` on the
        rows and :math:`i^{\\phi}` on the unknowns of each component,
        :math:`D_\\psi A_k D_\\phi` is real when :math:`\\psi_i + \\phi_j` has
        the parity of every block (i, j). That holds for Navier-Stokes, where u
        and its row take a phase and nothing else does. A block that mixes
        real and imaginary weights, like advection-diffusion's
        :math:`\\nu k^2 + i c k`, has no such phases.

        Returns None also when every phase is zero: the :math:`A_k` are real
        already, and there is nothing to apply.
        """
        parity: dict[tuple[int, int], int] = {}
        for bi, bj, w, P in self.terms:
            pw, pP = _parity(w), _parity(P)
            if pw is None or pP is None:
                return None
            if pw < 0 or pP < 0:
                continue
            if parity.setdefault((bi, bj), (pw + pP) % 2) != (pw + pP) % 2:
                return None
        psi = np.full(len(self.row_sizes), -1)
        phi = np.full(len(self.col_sizes), -1)
        for start in range(len(psi)):
            if psi[start] >= 0:
                continue
            psi[start] = 0
            stack = [("r", start)]
            while stack:
                side, i = stack.pop()
                for (bi, bj), s in parity.items():
                    if side == "r" and bi == i:
                        want, j, other, nxt = (psi[i] + s) % 2, bj, phi, "c"
                    elif side == "c" and bj == i:
                        want, j, other, nxt = (phi[i] + s) % 2, bi, psi, "r"
                    else:
                        continue
                    if other[j] < 0:
                        other[j] = want
                        stack.append((nxt, j))
                    elif other[j] != want:
                        return None
        phi[phi < 0] = 0
        if not psi.any() and not phi.any():
            return None
        return psi, phi

    def weights(self, phases: tuple[np.ndarray, np.ndarray] | None) -> Array:
        """Return every term's weight per wavenumber, ``(n_F, n_terms)``.

        Each is multiplied by its block's phase, if ``phases`` are given.
        """
        W = np.stack([w for _, _, w, _ in self.terms], axis=1)
        if phases is not None:
            psi, phi = phases
            W = W * np.array([1j ** int(psi[i] + phi[j]) for i, j, _, _ in self.terms])
        return self.place(W.astype(jnp.result_type(self.dtype, jnp.complex64)))

    def keep(self) -> Array | None:
        """Return ``(n_F, n)`` zeros on the pinned rows and ones elsewhere, or None."""
        if not self.pinned:
            return None
        keep = np.ones((self.n_F, self.n), dtype=jnp.finfo(self.dtype).dtype)
        for k, r, _ in self.pinned:
            keep[k, r] = 0
        return self.place(keep)

    def place(self, x: np.ndarray) -> Array:
        """Return ``x``, wavenumbers first, split over the devices if `sharded`."""
        if not self.sharded:
            return jnp.asarray(x)
        from jaxfun.sharding import spectral_sharding

        return jax.make_array_from_callback(
            x.shape, spectral_sharding, lambda index: x[index]
        )

    def map(self, fn: Callable, in_specs: Any, out_specs: Any) -> Callable:
        """Return ``fn`` jitted, run on each device's own wavenumbers if `sharded`.

        ``in_specs`` and ``out_specs`` mark the arguments and results that carry
        wavenumbers, `_SPLIT`, and those that are the same everywhere, `_SAME`.
        Per-device data has to come in as an argument: a closed-over array is
        the same on every device.
        """
        if self.sharded:
            from jaxfun.sharding import spmd_mesh

            fn = shard_map(
                fn,
                mesh=spmd_mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
        return jax.jit(fn)

    def assembler(
        self,
        dtype: Any,
        shape: tuple[int, int],
        position: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> Callable[[Array, Array | None], Array]:
        """Return a function that builds every :math:`A_k` from the weights.

        It takes `weights` and `keep`, for whichever wavenumbers it is handed,
        and returns them shaped ``(n_k, *shape)``, with ``position`` mapping a
        (row, column) of :math:`A_k` to its place in that layout. A pinned row
        is replaced by its identity row.
        """
        found = [
            (
                np.full(ii.size, t),
                self.row_starts[bi] + ii,
                self.col_starts[bj] + jj,
                P[ii, jj],
            )
            for t, (bi, bj, w, P) in enumerate(self.terms)
            if np.any(w)
            for ii, jj in [np.nonzero(P)]
        ]
        term, rows, cols, vals = (np.concatenate(a) for a in zip(*found, strict=True))
        at = position(rows, cols)
        pairs = sorted({(r, c) for _, r, c in self.pinned})
        pin_rows = np.array([r for r, _ in pairs], dtype=int)
        pin_at = position(pin_rows, np.array([c for _, c in pairs], dtype=int))
        real = not jnp.issubdtype(dtype, jnp.complexfloating)

        def assemble(W: Array, keep: Array | None) -> Array:
            v = W[:, term] * vals
            v = (v.real if real else v).astype(dtype)
            A = jnp.zeros((W.shape[0], *shape), dtype)
            if keep is None:
                return A.at[:, at[0], at[1]].add(v)
            A = A.at[:, at[0], at[1]].add(v * keep[:, rows])
            return A.at[:, pin_at[0], pin_at[1]].add(1 - keep[:, pin_rows])

        return assemble

    @functools.cached_property
    def band(self) -> tuple[np.ndarray, int, int]:
        """Return the reordering shared by all wavenumbers and its bandwidths.

        Every :math:`A_k` has the same sparsity pattern -- the Fourier weights
        only scale the blocks -- so one reverse Cuthill-McKee ordering of the
        union pattern serves them all. It interleaves the fields mode by mode.

        Returns:
            ``(perm, p, q)``: band position -> natural index, and the lower and
            upper bandwidths of the reordered pattern.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        pattern = np.zeros((self.n, self.n), dtype=bool)
        for bi, bj, w, P in self.terms:
            if np.any(w):
                ii, jj = np.nonzero(P)
                pattern[self.row_starts[bi] + ii, self.col_starts[bj] + jj] = True
        for _, r, c in self.pinned:
            pattern[r, c] = True
        perm = np.asarray(
            reverse_cuthill_mckee(csr_matrix(pattern | pattern.T), symmetric_mode=True)
        )
        i, j = np.nonzero(pattern[np.ix_(perm, perm)])
        return (
            perm,
            int(max((i - j).max(initial=0), 0)),
            int(max((j - i).max(initial=0), 0)),
        )

    @property
    def band_is_narrow(self) -> bool:
        """Whether the banded factorization beats the dense one.

        Per wavenumber, the dense substitution costs about n^2 and the banded one
        n (3p + q + 1) -- the pivoting widens the band. But a banded step is a
        sequential scan where the dense one is a blocked triangular solve, so the
        banded one has to do several times less arithmetic to win. A factor 4
        was on the safe side of CPU measurements; on an accelerator, where a
        scan step costs a kernel launch, ``method="dense"`` may well do better.
        """
        _, p, q = self.band
        return self.n >= 4 * (3 * p + q + 1)


def _parity(a: np.ndarray) -> int | None:
    """Return 0 if ``a`` is real, 1 if imaginary, -1 if zero, None if mixed."""
    re, im = bool(np.any(np.real(a) != 0)), bool(np.any(np.imag(a) != 0))
    if re and im:
        return None
    return 1 if im else (0 if re else -1)


def _stack(
    rhs: Sequence[Array],
    fourier: tuple[int, ...],
    poly_axis: int | None,
    sizes: tuple[int, ...],
) -> Array:
    """Return the components laid out ``(n_F, n)``, Fourier axes first."""
    order = fourier + (() if poly_axis is None else (poly_axis,))
    return jnp.concatenate(
        [
            jnp.transpose(jnp.asarray(r), order).reshape(-1, n)
            for r, n in zip(rhs, sizes, strict=True)
        ],
        axis=1,
    )


def _unstack(
    x: Array,
    fourier: tuple[int, ...],
    poly_axis: int | None,
    sizes: tuple[int, ...],
    dofs: tuple[tuple[int, ...], ...],
) -> tuple[Array, ...]:
    """Undo `_stack` for the trial components."""
    order = fourier + (() if poly_axis is None else (poly_axis,))
    back = tuple(int(i) for i in np.argsort(order))
    out = []
    start = 0
    for n, shape in zip(sizes, dofs, strict=True):
        # The leading axis is inferred: it is the split one on several devices.
        xc = x[:, start : start + n].reshape((-1, *(shape[a] for a in order[1:])))
        out.append(jnp.transpose(xc, back))
        start += n
    return tuple(out)


def _singular(diagonal: Array) -> Array:
    """Return one flag per wavenumber: whether a pivot of its factors is zero."""
    return jnp.any(~jnp.isfinite(diagonal) | (diagonal == 0), axis=1)


def _check_nonsingular(singular: Array) -> None:
    flags = _fetch(singular)
    if flags.any():
        k = int(np.nonzero(flags)[0][0])
        raise ValueError(
            f"Block system is singular at wavenumber {k} (flat index). Pin "
            "the undetermined unknown, e.g. a pressure mode, through `pins`."
        )


def _fetch(x: Array) -> np.ndarray:
    """Return ``x`` whole, gathering the parts that other processes hold."""
    if x.is_fully_addressable:
        return np.asarray(x)
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


class BlockWavenumberSolver(nnx.Pytree):
    """Per-wavenumber dense LU of a Fourier x polynomial block system.

    Each :math:`A_k` (see `_WavenumberSystem`) is factorized dense with partial
    pivoting, all at once. When phases make every :math:`A_k` real, the factors
    are real and a complex right-hand side is solved as two real ones.
    """

    def __init__(self, system: _WavenumberSystem) -> None:
        phases = system.phases()
        dtype = system.dtype if phases is None else jnp.finfo(system.dtype).dtype
        n = system.n
        assemble = system.assembler(dtype, (n, n), lambda r, c: (r, c))

        def factor(W: Array, keep: Array | None) -> tuple[Array, Array, Array]:
            lu, piv = jax.vmap(jsl.lu_factor)(assemble(W, keep))
            perm = jax.lax.linalg.lu_pivots_to_permutation(piv, n)
            return lu, perm, _singular(jnp.diagonal(lu, axis1=1, axis2=2))

        k = _SPLIT
        self.keep = system.keep()
        self.lu, self.perm, singular = system.map(factor, (k, k), (k, k, k))(
            system.weights(phases), self.keep
        )
        _check_nonsingular(singular)
        self.row_phase, self.col_phase = (
            (None, None) if phases is None else _phase_vectors(system, phases)
        )
        self.pinned_rows = tuple((k, r) for k, r, _ in system.pinned)

        def solve(
            lu: Array,
            perm: Array,
            keep: Array | None,
            row_phase: Array | None,
            col_phase: Array | None,
            rhs: tuple[Array, ...],
        ) -> tuple[Array, ...]:
            b = _stack(rhs, system.fourier, system.poly_axis, system.row_sizes)
            if keep is not None:
                b = b * keep
            if row_phase is not None:
                b = b * row_phase
            x = _dense_solve(lu, perm, b)
            if col_phase is not None:
                x = x * col_phase
            return _unstack(
                x, system.fourier, system.poly_axis, system.col_sizes, system.trial_dofs
            )

        self._solve = system.map(solve, (k, k, k, _SAME, _SAME, k), k)

    def solve(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block system.

        Args:
            rhs: One array per test component, each in its coefficient shape.

        Returns:
            One array per trial component.
        """
        return self._solve(
            self.lu, self.perm, self.keep, self.row_phase, self.col_phase, tuple(rhs)
        )


class BlockBandedWavenumberSolver(nnx.Pytree):
    """Per-wavenumber banded LU of a Fourier x polynomial block system.

    The unknowns are reordered once for all wavenumbers, by reverse
    Cuthill-McKee on the union of the :math:`A_k` patterns, which interleaves the
    fields mode by mode into a narrow band. The bands of all :math:`A_k` are then
    assembled straight from the terms, without the dense matrices, factorized
    with partial pivoting within the band as LAPACK's ``gbtrf`` does, and solved
    by banded substitution that replays the row swaps. As in
    `BlockWavenumberSolver`, phases make the factors real when they can.
    """

    def __init__(self, system: _WavenumberSystem) -> None:
        perm, p, q = system.band
        phases = system.phases()
        n = system.n
        dtype = system.dtype if phases is None else jnp.finfo(system.dtype).dtype
        inv = np.argsort(perm)
        center = p  # L's multipliers below, U widened by p for swap fill above
        assemble = system.assembler(
            dtype,
            (center + p + q + 1, n),
            lambda r, c: (center + inv[c] - inv[r], inv[c]),
        )

        def factor(W: Array, keep: Array | None) -> tuple[Array, ...]:
            band_lu, pivots = jax.vmap(
                lambda b: _lu_banded_kernel(b, p, p + q, center)
            )(assemble(W, keep))
            # Multipliers as (n_k, n, p), column t - 1 for the row t below; all p
            # kept, since a swap can reach that far.
            multipliers = jnp.flip(band_lu[:, :center], axis=1).transpose(0, 2, 1)
            return (
                multipliers,
                pivots,
                band_lu[:, center:],
                _singular(band_lu[:, center]),
            )

        k = _SPLIT
        self.keep = system.keep()
        self.multipliers, self.pivots, U, singular = system.map(
            factor, (k, k), (k, k, k, k)
        )(system.weights(phases), self.keep)
        _check_nonsingular(singular)
        # Which diagonals survive sets the substitution's shape, so it is decided
        # over all wavenumbers, on every device and process alike.
        _, self.U, _, U_offsets = _prune_zero_diagonals(
            U[:, :0], U, (), tuple(range(0, p + q + 1))
        )
        self.row_phase, self.col_phase = (
            (None, None) if phases is None else _phase_vectors(system, phases)
        )
        self.pinned_rows = tuple((k, r) for k, r, _ in system.pinned)
        backward = _make_wavenumber_solve((), U_offsets, n, dtype)

        def substitute(multipliers: Array, pivots: Array, U: Array, b: Array) -> Array:
            forward = jax.vmap(
                lambda m, piv, x: _pivoted_forward(m, piv, x[:, None])[:, 0]
            )
            return backward(
                U[:, :0], U, forward(multipliers.astype(b.dtype), pivots, b)
            )

        def solve(
            multipliers: Array,
            pivots: Array,
            U: Array,
            keep: Array | None,
            row_phase: Array | None,
            col_phase: Array | None,
            rhs: tuple[Array, ...],
        ) -> tuple[Array, ...]:
            b = _stack(rhs, system.fourier, system.poly_axis, system.row_sizes)
            if keep is not None:
                b = b * keep
            if row_phase is not None:
                b = b * row_phase
            b = b[:, perm]
            if jnp.iscomplexobj(b) and not jnp.iscomplexobj(U):
                # The real and imaginary parts go through as wavenumbers of their
                # own, one pass for both. A separate axis for the two parts,
                # trailing or vmapped, compiles to a markedly slower scan.
                n_k = b.shape[0]
                both = substitute(
                    jnp.concatenate([multipliers, multipliers]),
                    jnp.concatenate([pivots, pivots]),
                    jnp.concatenate([U, U]),
                    jnp.concatenate([b.real, b.imag]),
                )
                x = both[:n_k] + 1j * both[n_k:]
            else:
                x = substitute(multipliers, pivots, U, b.astype(jnp.result_type(U, b)))
            x = x[:, inv]
            if col_phase is not None:
                x = x * col_phase
            return _unstack(
                x, system.fourier, system.poly_axis, system.col_sizes, system.trial_dofs
            )

        self._solve = system.map(solve, (k, k, k, k, _SAME, _SAME, k), k)

    def solve(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block system; see `BlockWavenumberSolver.solve`."""
        return self._solve(
            self.multipliers,
            self.pivots,
            self.U,
            self.keep,
            self.row_phase,
            self.col_phase,
            tuple(rhs),
        )


class BlockFourierSolver(nnx.Pytree):
    """Per-wavenumber inverse of a block system that is Fourier along every axis.

    Every term is diagonal, so the system at each wavenumber couples only the
    components: one unknown each. Its inverse is computed once, and a solve is
    an elementwise sum of products over the components, each in its own layout
    -- the block counterpart of a diagonal solve. As in `BlockWavenumberSolver`,
    phases make the inverse real when they can.
    """

    def __init__(self, system: _WavenumberSystem) -> None:
        phases = system.phases()
        dtype = system.dtype if phases is None else jnp.finfo(system.dtype).dtype
        n = system.n
        assemble = system.assembler(dtype, (n, n), lambda r, c: (r, c))
        trailing = system.fourier_shape[1:]

        def factor(W: Array, keep: Array | None) -> tuple[Array, Array]:
            lu, piv = jax.vmap(jsl.lu_factor)(assemble(W, keep))
            eye = jnp.eye(n, dtype=dtype)
            inverse = jax.vmap(lambda f, p: jsl.lu_solve((f, p), eye))(lu, piv)
            if keep is not None:
                # A pinned row's right-hand side is ignored, so its unknown is 0.
                inverse = inverse * keep[:, None, :]
            # Back on the Fourier axes, with (unknown, row) last.
            return (
                inverse.reshape((-1, *trailing, n, n)),
                _singular(jnp.diagonal(lu, axis1=1, axis2=2)),
            )

        k = _SPLIT
        self.inverse, singular = system.map(factor, (k, k), (k, k))(
            system.weights(phases), system.keep()
        )
        _check_nonsingular(singular)
        self.pinned_rows = tuple((k, r) for k, r, _ in system.pinned)
        pairs = tuple(
            (int(i), int(j)) for i, j in zip(*np.nonzero(_nonzero_pairs(self.inverse)))
        )
        row_phase, col_phase = (
            ((1,) * n, (1,) * n)
            if phases is None
            else tuple(tuple(1j ** int(s) for s in ph) for ph in phases)
        )

        def solve(inverse: Array, rhs: tuple[Array, ...]) -> tuple[Array, ...]:
            b = [jnp.asarray(r) * s for r, s in zip(rhs, row_phase, strict=True)]
            x: list[Array | None] = [None] * n
            for i, j in pairs:
                term, partial = inverse[..., i, j] * b[j], x[i]
                x[i] = term if partial is None else partial + term
            # Every component has the same (Fourier) shape.
            zero = jnp.zeros_like(b[0], dtype=jnp.result_type(inverse, *b))
            return tuple(
                zero if xi is None else xi * s
                for xi, s in zip(x, col_phase, strict=True)
            )

        self._solve = system.map(solve, (k, k), k)

    def solve(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block system; see `BlockWavenumberSolver.solve`."""
        return self._solve(self.inverse, tuple(rhs))


def _nonzero_pairs(inverse: Array) -> np.ndarray:
    """Which entries of the per-wavenumber blocks are nonzero at any wavenumber.

    Reduced over each addressable block and OR-ed across processes, so that every
    process skips the same entries and compiles the same solve.
    """
    axes = tuple(range(inverse.ndim - 2))
    nonzero = np.zeros(inverse.shape[-2:], dtype=bool)
    for shard in inverse.addressable_shards:
        nonzero |= np.asarray(jnp.any(shard.data != 0, axis=axes))
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        gathered = multihost_utils.process_allgather(jnp.asarray(nonzero))
        nonzero = np.any(np.asarray(gathered), axis=0)
    return nonzero


def _phase_vectors(
    system: _WavenumberSystem, phases: tuple[np.ndarray, np.ndarray]
) -> tuple[Array, Array]:
    """Return the row and column phases per unknown, ``(n,)`` each."""
    psi, phi = phases
    row = np.concatenate(
        [np.full(n, 1j ** int(s)) for n, s in zip(system.row_sizes, psi, strict=True)]
    )
    col = np.concatenate(
        [np.full(n, 1j ** int(s)) for n, s in zip(system.col_sizes, phi, strict=True)]
    )
    return jnp.asarray(row), jnp.asarray(col)


def _dense_solve(lu: Array, perm: Array, b: Array) -> Array:
    """Solve every wavenumber's system from its dense LU factors.

    A complex right-hand side for real factors is solved as two real columns.
    """
    split = jnp.iscomplexobj(b) and not jnp.iscomplexobj(lu)
    if split:
        rhs = jnp.stack([b.real, b.imag], axis=-1)
    else:
        rhs = b.astype(jnp.result_type(lu, b))[..., None]
    rhs = jnp.take_along_axis(rhs, perm[..., None], axis=1)
    y = jax.lax.linalg.triangular_solve(
        lu, rhs, left_side=True, lower=True, unit_diagonal=True
    )
    x = jax.lax.linalg.triangular_solve(lu, y, left_side=True, lower=False)
    return x[..., 0] + 1j * x[..., 1] if split else x[..., 0]


class BlockTriangularSolver(nnx.Pytree):
    """Block forward substitution over fields that enter each other one way.

    The groups of `_triangular_groups` make the operator block lower
    triangular: the equations of each group hold only its own unknowns and
    those of the groups before it. In Rayleigh-Benard convection, say, the
    temperature enters the momentum equation through buoyancy, but no other
    field enters the temperature equation implicitly. Each group is factorized
    on its own, and a solve takes them in order, moving the terms that hold
    unknowns already solved for to the right-hand side. The answer is that of
    the whole system, from smaller ones: there, a pivoted saddle-point system
    without the temperature, and the temperature alone, which as a single field
    takes the scalar wavenumber solver.
    """

    def __init__(
        self,
        mats: Sequence[TPMatrix],
        test_dofs: tuple[tuple[int, ...], ...],
        trial_dofs: tuple[tuple[int, ...], ...],
        pins: Sequence[Pin],
        groups: tuple[tuple[int, ...], ...],
        method: str,
    ) -> None:
        group_of = {c: g for g, group in enumerate(groups) for c in group}
        local = {c: group.index(c) for group in groups for c in group}
        terms: list[list[TPMatrix]] = [[] for _ in groups]
        slots, couplings = [], []
        for tp in mats:
            i, j = (int(c) for c in tp.global_indices)
            if group_of[i] == group_of[j]:
                indices = (local[i], local[j])
                terms[group_of[i]].append(TPMatrix(tp.mats, tp.coefficient, indices))
            else:
                slots.append((i, j))
                couplings.append(tp)
        self.groups = groups
        self.slots = tuple(slots)
        self.couplings = nnx.data(tuple(couplings))
        self.solvers = nnx.data(
            tuple(
                _group_solver(
                    terms[g],
                    tuple(test_dofs[c] for c in group),
                    tuple(trial_dofs[c] for c in group),
                    [(local[c], index) for c, index in pins if group_of[c] == g],
                    method,
                )
                for g, group in enumerate(groups)
            )
        )

    def solve(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block system; see `BlockWavenumberSolver.solve`."""
        x: dict[int, Array] = {}
        for group, solver in zip(self.groups, self.solvers, strict=True):
            b = [jnp.asarray(rhs[c]) for c in group]
            for (i, j), op in zip(self.slots, self.couplings, strict=True):
                if i in group:
                    b[group.index(i)] = b[group.index(i)] - op @ x[j]
            if isinstance(solver, TPMatricesWavenumberSolver):
                xs = (solver.solve(b[0]),)
            else:
                xs = solver.solve(tuple(b))
            x.update(zip(group, xs, strict=True))
        return tuple(x[c] for c in range(len(x)))


def _group_solver(
    terms: Sequence[TPMatrix],
    test_dofs: tuple[tuple[int, ...], ...],
    trial_dofs: tuple[tuple[int, ...], ...],
    pins: Sequence[Pin],
    method: str,
) -> (
    TPMatricesWavenumberSolver
    | BlockFourierSolver
    | BlockWavenumberSolver
    | BlockBandedWavenumberSolver
):
    """Return the solver of one group of `BlockTriangularSolver`.

    A single field with nothing to pin is solved as a scalar equation would be,
    without pivoting and with its even and odd modes apart where they decouple.
    """
    system = _WavenumberSystem(terms, test_dofs, trial_dofs, pins)
    if (
        len(test_dofs) == 1
        and system.fourier
        and system.poly_axis is not None
        and not system.pinned
        and all(isinstance(m, DiaMatrix) for tp in terms for m in tp.mats)
    ):
        return tpmats_wavenumber_factor(list(terms))
    return _wavenumber_solver(system, method)


class BlockGlobalSolver(nnx.Pytree):
    """Global sparse LU of a flattened block system.

    The fallback for block layouts with no exploitable Fourier structure. The
    whole operator is assembled into one `DiaMatrix` and factorized with
    pivoting, as for the steady Stokes problem. The arguments are those of
    `block_lu_factor`.
    """

    def __init__(
        self,
        mats: BlockMatrix | Sequence[BlockTerm],
        test_space: BlockSpace | None = None,
        trial_space: BlockSpace | None = None,
        pins: Sequence[Pin] = (),
        **solve_options: Any,
    ) -> None:
        terms, test_space, trial_space = _unpack(mats, test_space, trial_space)
        test_dofs, trial_dofs = _num_dofs(test_space), _num_dofs(trial_space)
        _check_pins(pins, test_dofs, trial_dofs)
        bm = (
            mats
            if isinstance(mats, BlockMatrix)
            else BlockMatrix(list(terms), test_space, trial_space)
        )
        dia = bm.tosparse()

        row_starts = np.cumsum((0,) + tuple(test_space.block_sizes[:-1]))
        col_starts = np.cumsum((0,) + tuple(trial_space.block_sizes[:-1]))
        pinned = {
            int(row_starts[c] + np.ravel_multi_index(index, test_dofs[c]))
            for c, index in pins
        }
        if row_starts.tolist() == col_starts.tolist():
            absolute = DiaMatrix(jnp.abs(dia.data), dia.offsets, dia.shape)
            ones = jnp.ones(dia.shape[1], dtype=absolute.data.dtype)
            zero = np.asarray(absolute.matvec(ones) == 0) & np.asarray(
                absolute.T.matvec(ones) == 0
            )
            pinned |= {int(i) for i in np.nonzero(zero)[0]}

        self.matrix: BaseMatrix = nnx.data(
            dia.pin({i: 0.0 for i in sorted(pinned)}) if pinned else dia
        )
        self.solve_options = tuple(sorted(solve_options.items()))
        self.trial_dofs = trial_dofs
        # Factorize now, while the matrix is concrete: the solve path is picked
        # from its values, and inside a jitted step they are tracers.
        self.solve(tuple(jnp.zeros(nd) for nd in test_dofs))

    def solve(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block system; see `BlockWavenumberSolver.solve`."""
        b = jnp.concatenate([jnp.asarray(r).ravel() for r in rhs])
        matrix = cast(DiaMatrix, self.matrix)

        def solve(r: Array) -> Array:
            return matrix.lu_solve(r, pivot=True, **dict(self.solve_options))

        if jnp.iscomplexobj(b) and not jnp.issubdtype(
            matrix.dtype, jnp.complexfloating
        ):
            x = solve(b.real) + 1j * solve(b.imag)
        else:
            x = solve(b.astype(jnp.result_type(matrix.dtype, b)))
        splits = np.cumsum([int(np.prod(nd)) for nd in self.trial_dofs])[:-1]
        return tuple(
            xc.reshape(nd)
            for xc, nd in zip(jnp.split(x, splits), self.trial_dofs, strict=True)
        )


def _working_dtype(mats: Sequence[TPMatrix]) -> jnp.dtype:
    """Return the dtype to factorize in, real unless something is complex-valued.

    As in `tpmats_wavenumber_factor`: a Fourier diagonal is stored complex even
    when it is real (a second derivative), and factorizing in complex then
    doubles the cost for nothing.
    """
    parts = [
        m.data if isinstance(m, DiaMatrix) else m.todense()
        for tp in mats
        for m in tp.mats
    ]
    parts += [jnp.asarray(tp.coefficient) for tp in mats]
    dtype = jnp.result_type(*parts)
    if jnp.issubdtype(dtype, jnp.complexfloating) and not any(
        bool(jnp.any(jnp.imag(p) != 0)) for p in parts
    ):
        return jnp.finfo(dtype).dtype
    return dtype
