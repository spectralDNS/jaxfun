"""Factored solves of block (multi-field) tensor-product systems.

A coupled system over a mixed space -- velocity, pressure and temperature, say
-- assembles into a `BlockMatrix` whose terms are `TPMatrix` objects tagged with
the (test, trial) component they couple. `block_lu_factor` factorizes such an
operator once and returns a solver acting on one coefficient array per component.

When every component shares the same Fourier axes and at most one non-periodic
axis, the system decouples into one small block system per Fourier wavenumber.
`BlockWavenumberSolver` factorizes those dense, and `BlockBandedWavenumberSolver`
banded, after a reordering that interleaves the fields; both pivot, which the
zero diagonal of a saddle-point (pressure) block needs. Every other layout falls
back to one global sparse LU of the flattened system.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from flax import nnx
from jax import Array

from jaxfun.la.blockmatrix import BlockMatrix
from jaxfun.la.diamatrix import DiaMatrix, _lu_banded_kernel, _pivoted_forward
from jaxfun.la.matrixprotocol import BaseMatrix, SolverNotApplicable
from jaxfun.la.operators import GlobalMatrix
from jaxfun.la.tpmatrix import TPMatrix, _make_wavenumber_solve, _prune_zero_diagonals

if TYPE_CHECKING:
    from jaxfun.galerkin import CartesianProductSpace, CartesianTensorProductSpace

    type BlockSpace = CartesianTensorProductSpace | CartesianProductSpace

type Pin = tuple[int, tuple[int, ...]]

_WAVENUMBER_METHODS = frozenset({"auto", "dense", "rcm", "banded"})


def block_lu_factor(
    mats: Sequence[TPMatrix | GlobalMatrix],
    test_space: BlockSpace,
    trial_space: BlockSpace,
    pins: Sequence[Pin] = (),
    **solve_options: Any,
) -> BlockWavenumberSolver | BlockBandedWavenumberSolver | BlockGlobalSolver:
    """Factorize a block operator for repeated solves.

    Args:
        mats: The operator's terms, each tagged with ``global_indices = (test
            component, trial component)``, as in ``BlockMatrix.mats``.
        test_space: Mixed space of the rows (the test functions).
        trial_space: Mixed space of the columns (the unknowns).
        pins: ``(component, index)`` pairs. The row of that test component and
            coefficient index is replaced by an identity row on the same trial
            component and index, which fixes that unknown to zero. Use it for a
            nullspace, such as the constant pressure mode. A row and column that
            are both identically zero at the same component and index are
            pinned automatically.
        **solve_options: ``method`` picks how each wavenumber is factorized:
            ``"dense"``, ``"rcm"`` (banded, after a reverse Cuthill-McKee
            reordering; ``"banded"`` is the same) or ``"auto"``, the default,
            which takes the banded factorization when its band is narrow. Without
            Fourier structure the global sparse solve takes ``method`` and
            ``auto_threshold`` as `DiaMatrix.lu_solve` does.

    Returns:
        A solver whose ``solve`` maps a tuple of right-hand sides, one per test
        component, to a tuple of solutions, one per trial component.
    """
    # A 1D mixed space assembles `GlobalMatrix` blocks; as one-axis tensor
    # products they take the same per-wavenumber path.
    mats = [
        TPMatrix([m.data], 1, global_indices=m.global_indices)
        if isinstance(m, GlobalMatrix)
        else m
        for m in mats
    ]
    try:
        system = _WavenumberSystem(mats, test_space, trial_space, pins)
    except SolverNotApplicable:
        return BlockGlobalSolver(mats, test_space, trial_space, pins, **solve_options)
    method = solve_options.get("method", "auto")
    if method not in _WAVENUMBER_METHODS:
        raise ValueError(
            f"Unsupported method {method!r} for a per-wavenumber block solve; "
            f"use one of {sorted(_WAVENUMBER_METHODS)}."
        )
    if method == "dense" or (method == "auto" and not system.band_is_narrow):
        return BlockWavenumberSolver(system)
    return BlockBandedWavenumberSolver(system)


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
        test_space: BlockSpace,
        trial_space: BlockSpace,
        pins: Sequence[Pin],
    ) -> None:
        test_dofs, trial_dofs = _num_dofs(test_space), _num_dofs(trial_space)
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
        return psi, phi

    def dense(self, phases: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
        """Return every :math:`A_k`, ``(n_F, n, n)``, scaled by ``phases``."""
        A = np.zeros((self.n_F, self.n, self.n), dtype=complex)
        for bi, bj, w, P in self.terms:
            if phases is not None:
                w = w * 1j ** int(phases[0][bi] + phases[1][bj])
            r0, c0 = self.row_starts[bi], self.col_starts[bj]
            A[:, r0 : r0 + P.shape[0], c0 : c0 + P.shape[1]] += (
                w[:, None, None] * P[None]
            )
        if phases is None:
            return A.astype(self.dtype)
        assert not np.any(A.imag), "phases left an imaginary part"
        return A.real.astype(jnp.finfo(self.dtype).dtype)

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
        xc = x[:, start : start + n].reshape(tuple(shape[a] for a in order))
        out.append(jnp.transpose(xc, back))
        start += n
    return tuple(out)


def _check_nonsingular(diag: Array) -> None:
    magnitude = np.abs(np.asarray(diag))
    bad = ~np.isfinite(magnitude) | (magnitude == 0)
    if bad.any():
        k = int(np.nonzero(bad.any(axis=1))[0][0])
        raise ValueError(
            f"Block system is singular at wavenumber {k} (flat index). Pin "
            "the undetermined unknown, e.g. a pressure mode, through `pins`."
        )


class BlockWavenumberSolver(nnx.Pytree):
    """Per-wavenumber dense LU of a Fourier x polynomial block system.

    Each :math:`A_k` (see `_WavenumberSystem`) is factorized dense with partial
    pivoting, all at once. When phases make every :math:`A_k` real, the factors
    are real and a complex right-hand side is solved as two real ones.
    """

    def __init__(self, system: _WavenumberSystem) -> None:
        phases = system.phases()
        A = system.dense(phases)
        for k, r, c in system.pinned:
            A[k, r, :] = 0
            A[k, r, c] = 1
        lu, piv = jax.vmap(jsl.lu_factor)(jnp.asarray(A))
        _check_nonsingular(jnp.diagonal(lu, axis1=1, axis2=2))
        self.lu = lu
        self.perm = jax.lax.linalg.lu_pivots_to_permutation(piv, system.n)
        self.row_phase, self.col_phase = (
            (None, None) if phases is None else _phase_vectors(system, phases)
        )
        self.keep = _keep_mask(system)
        self.fourier = system.fourier
        self.poly_axis = system.poly_axis
        self.row_sizes = system.row_sizes
        self.col_sizes = system.col_sizes
        self.trial_dofs = system.trial_dofs
        self.pinned_rows = tuple((k, r) for k, r, _ in system.pinned)

    def solve(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block system.

        Args:
            rhs: One array per test component, each in its coefficient shape.

        Returns:
            One array per trial component.
        """
        b = _stack(rhs, self.fourier, self.poly_axis, self.row_sizes)
        if self.keep is not None:
            b = b * self.keep
        if self.row_phase is not None:
            b = b * self.row_phase
        x = _dense_solve(self.lu, self.perm, b)
        if self.col_phase is not None:
            x = x * self.col_phase
        return _unstack(
            x, self.fourier, self.poly_axis, self.col_sizes, self.trial_dofs
        )


class BlockBandedWavenumberSolver(nnx.Pytree):
    """Per-wavenumber banded LU of a Fourier x polynomial block system.

    The unknowns are reordered once for all wavenumbers, by reverse
    Cuthill-McKee on the union of the :math:`A_k` patterns, which interleaves the
    fields mode by mode into a narrow band. The bands of all :math:`A_k` are then
    assembled straight from the terms, without the dense matrices, factorized
    with partial pivoting within the band as LAPACK's ``gbtrf`` does, and solved
    by banded substitution that replays the row swaps.
    """

    def __init__(self, system: _WavenumberSystem) -> None:
        perm, p, q = system.band
        n, n_F = system.n, system.n_F
        inv = np.argsort(perm)
        center = p  # L's multipliers below, U widened by p for swap fill above
        band = np.zeros((n_F, center + p + q + 1, n), dtype=complex)
        rows, cols, vals = [], [], []
        for bi, bj, w, P in system.terms:
            ii, jj = np.nonzero(P)
            r = inv[system.row_starts[bi] + ii]
            c = inv[system.col_starts[bj] + jj]
            rows.append(center + c - r)
            cols.append(c)
            vals.append(np.outer(w, P[ii, jj]))
        # Scattered through a (band row, column, wavenumber) view of `band`.
        np.add.at(
            band.transpose(1, 2, 0),
            (np.concatenate(rows), np.concatenate(cols)),
            np.concatenate(vals, axis=1).T,
        )
        for k, r, c in system.pinned:
            rp, cp = inv[r], inv[c]
            for off in range(-p, q + 1):
                if 0 <= rp + off < n:
                    band[k, center + off, rp + off] = 0
            band[k, center + cp - rp, cp] = 1
        dtype = system.dtype
        band = band if jnp.issubdtype(dtype, jnp.complexfloating) else band.real
        band_lu, pivots = jax.jit(
            jax.vmap(lambda b: _lu_banded_kernel(b, p, p + q, center))
        )(jnp.asarray(band, dtype=dtype))
        _check_nonsingular(band_lu[:, center])
        # Multipliers as (n_F, n, p), column t - 1 for the row t below; all p
        # kept, since a swap can reach that far.
        self.multipliers = jnp.flip(band_lu[:, :center], axis=1).transpose(0, 2, 1)
        self.pivots = pivots
        _, U, _, U_offsets = _prune_zero_diagonals(
            band_lu[:, :0], band_lu[:, center:], (), tuple(range(0, p + q + 1))
        )
        self.U = U
        self._backward = _make_wavenumber_solve((), U_offsets, n, dtype)
        self.gather = jnp.asarray(perm)
        self.scatter = jnp.asarray(inv)
        self.keep = _keep_mask(system)
        self.fourier = system.fourier
        self.poly_axis = system.poly_axis
        self.row_sizes = system.row_sizes
        self.col_sizes = system.col_sizes
        self.trial_dofs = system.trial_dofs
        self.pinned_rows = tuple((k, r) for k, r, _ in system.pinned)

    def _substitute(self, b: Array) -> Array:
        forward = jax.vmap(lambda m, piv, x: _pivoted_forward(m, piv, x[:, None])[:, 0])
        y = forward(self.multipliers.astype(b.dtype), self.pivots, b)
        return self._backward(self.U[:, :0], self.U, y)

    def solve(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block system; see `BlockWavenumberSolver.solve`."""
        b = _stack(rhs, self.fourier, self.poly_axis, self.row_sizes)
        if self.keep is not None:
            b = b * self.keep
        b = b[:, self.gather]
        if jnp.iscomplexobj(b) and not jnp.iscomplexobj(self.U):
            x = self._substitute(b.real) + 1j * self._substitute(b.imag)
        else:
            x = self._substitute(b.astype(jnp.result_type(self.U, b)))
        return _unstack(
            x[:, self.scatter],
            self.fourier,
            self.poly_axis,
            self.col_sizes,
            self.trial_dofs,
        )


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


def _keep_mask(system: _WavenumberSystem) -> Array | None:
    """Return ``(n_F, n)`` zeros on the pinned rows and ones elsewhere, or None."""
    if not system.pinned:
        return None
    keep = np.ones((system.n_F, system.n))
    k, r, _ = (np.array(i) for i in zip(*system.pinned, strict=True))
    keep[k, r] = 0
    return jnp.asarray(keep, dtype=jnp.finfo(system.dtype).dtype)


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


class BlockGlobalSolver(nnx.Pytree):
    """Global sparse LU of a flattened block system.

    The fallback for block layouts with no exploitable Fourier structure. The
    whole operator is assembled into one `DiaMatrix` and factorized with
    pivoting, as for the steady Stokes problem.
    """

    def __init__(
        self,
        mats: Sequence[TPMatrix | GlobalMatrix],
        test_space: BlockSpace,
        trial_space: BlockSpace,
        pins: Sequence[Pin] = (),
        **solve_options: Any,
    ) -> None:
        test_dofs, trial_dofs = _num_dofs(test_space), _num_dofs(trial_space)
        _check_pins(pins, test_dofs, trial_dofs)
        bm = BlockMatrix(list(mats), test_space, trial_space)
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
