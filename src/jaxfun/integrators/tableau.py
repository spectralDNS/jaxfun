"""Butcher tableaux for implicit-explicit (IMEX) Runge-Kutta time integration."""

import math
from dataclasses import dataclass

_TOL = 1e-10


@dataclass(frozen=True)
class ButcherTableau:
    """A single (explicit or diagonally implicit) Runge-Kutta tableau.

    Args:
        A: Stage coefficients, ``A[i][j]`` weights stage ``j``'s contribution
            to stage ``i``. Square, ``stages x stages``.
        b: Output weights combining stage contributions into the final state.
        c: Stage abscissae. Must satisfy ``sum(A[i]) == c[i]`` for every row.
    """

    A: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]

    def __post_init__(self) -> None:
        s = len(self.c)
        if len(self.A) != s or any(len(row) != s for row in self.A):
            raise ValueError("Butcher tableau `A` must be square with `len(c)` rows")
        if len(self.b) != s:
            raise ValueError("Butcher tableau `b` must have `len(c)` entries")
        for i, row in enumerate(self.A):
            if abs(sum(row) - self.c[i]) > _TOL:
                raise ValueError(
                    f"Row {i} of `A` sums to {sum(row)}, expected c[{i}]={self.c[i]}"
                )

    @property
    def stages(self) -> int:
        """Number of stages `s`."""
        return len(self.c)


@dataclass(frozen=True)
class IMEXTableau:
    """A paired explicit/diagonally-implicit tableau for IMEX Runge-Kutta.

    The explicit tableau must be strictly lower triangular (``A[i][i] == 0``);
    the implicit tableau must be lower triangular (``A[i][j] == 0`` for
    ``j > i``), i.e. diagonally implicit. The two tableaux need not share
    abscissae `c` -- shared `c` is a convention used by some IMEX families
    (e.g. Kennedy-Carpenter ARK) but not required in general (e.g.
    Pareschi-Russo IMEX-SSP schemes use distinct explicit/implicit `c`).
    """

    explicit: ButcherTableau
    implicit: ButcherTableau

    def __post_init__(self) -> None:
        if self.explicit.stages != self.implicit.stages:
            raise ValueError("Explicit and implicit tableaux must have equal stages")
        for i, row in enumerate(self.explicit.A):
            if abs(row[i]) > _TOL:
                raise ValueError(
                    f"Explicit tableau must have zero diagonal, "
                    f"got A[{i}][{i}]={row[i]}"
                )
        for i, row in enumerate(self.implicit.A):
            if any(abs(row[j]) > _TOL for j in range(i + 1, len(row))):
                raise ValueError(
                    "Implicit tableau must be lower triangular (diagonally implicit)"
                )

    @property
    def stages(self) -> int:
        """Number of stages `s`, shared by both tableaux."""
        return self.explicit.stages

    @property
    def explicit_is_stiffly_accurate(self) -> bool:
        """Whether the explicit tableau alone satisfies ``A[-1] == b``."""
        return all(
            abs(a - bi) <= _TOL
            for a, bi in zip(self.explicit.A[-1], self.explicit.b, strict=True)
        )

    @property
    def implicit_is_stiffly_accurate(self) -> bool:
        """Whether the implicit tableau alone satisfies ``A[-1] == b``.

        Weaker than `is_stiffly_accurate` (which requires both tableaux) --
        Kennedy-Carpenter ARK schemes satisfy this but not the full GSA
        property, since their explicit tableau's own last row differs from
        `b`. When this holds but `is_stiffly_accurate` does not, `step` can
        still fold the ``b_i``-weighted final combination into the
        already-computed last stage, leaving only a ``(b_e - explicit.A[-1])``
        -weighted correction over the cached nonlinear stage values.
        """
        return all(
            abs(a - bi) <= _TOL
            for a, bi in zip(self.implicit.A[-1], self.implicit.b, strict=True)
        )

    @property
    def is_stiffly_accurate(self) -> bool:
        """Whether the last stage equals the accepted solution.

        True when the last row of both the explicit and the implicit `A`
        equals the corresponding `b` (globally stiffly accurate), letting
        `step` reuse the last stage value directly instead of recomputing a
        separate output combination.
        """
        return self.explicit_is_stiffly_accurate and self.implicit_is_stiffly_accurate

    @property
    def distinct_diagonal_coeffs(self) -> tuple[float, ...]:
        """Unique nonzero diagonal coefficients of the implicit tableau.

        Determines how many distinct implicit system operators `setup` must
        build and cache: stages that share a diagonal coefficient (common in
        ESDIRK-type schemes) reuse the same cached operator.
        """
        seen: list[float] = []
        for i in range(self.stages):
            a_ii = self.implicit.A[i][i]
            if abs(a_ii) > _TOL and not any(abs(a_ii - s) <= _TOL for s in seen):
                seen.append(a_ii)
        return tuple(seen)


IMEX_EULER = IMEXTableau(
    explicit=ButcherTableau(
        A=((0.0, 0.0), (1.0, 0.0)),
        b=(1.0, 0.0),
        c=(0.0, 1.0),
    ),
    implicit=ButcherTableau(
        A=((0.0, 0.0), (0.0, 1.0)),
        b=(0.0, 1.0),
        c=(0.0, 1.0),
    ),
)
"""First-order IMEX Euler: forward Euler (explicit) paired with backward Euler
(implicit). Degenerates exactly to :class:`~jaxfun.integrators.BackwardEuler`."""


def _imex_ssp2_222() -> IMEXTableau:
    """Return the Pareschi-Russo IMEX-SSP2(2,2,2) L-stable scheme."""
    gamma = 1.0 - 1.0 / math.sqrt(2.0)
    return IMEXTableau(
        explicit=ButcherTableau(
            A=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.5),
            c=(0.0, 1.0),
        ),
        implicit=ButcherTableau(
            A=((gamma, 0.0), (1.0 - gamma, gamma)),
            b=(0.5, 0.5),
            c=(gamma, 1.0),
        ),
    )


IMEX_SSP2_222 = _imex_ssp2_222()
"""Second-order, 2-stage, L-stable IMEX-SSP2(2,2,2) scheme (Pareschi & Russo,
J. Sci. Comput. 25, 2005), with gamma = 1 - 1/sqrt(2)."""


ARK3_2_4L2SA = IMEXTableau(
    explicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0),
            (0.8717330430169179988320389023871136850586, 0.0, 0.0, 0.0),
            (
                0.52758901197630041156180797140291790433,
                0.07241098802369958843819202859708209566999,
                0.0,
                0.0,
            ),
            (
                0.3990960076760701320627260736092142797856,
                -0.437557654613519443722846363831022571942,
                1.038461646937449311660120290221808292156,
                0.0,
            ),
        ),
        b=(
            0.1876410243467238251612921441668043913795,
            -0.5952974735769549480478230275858851737782,
            0.9717899277217721234705114322255239398694,
            0.4358665215084589994160194511935568425293,
        ),
        c=(0.0, 0.8717330430169179988320389023871136850586, 0.6, 1.0),
    ),
    implicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0),
            (
                0.4358665215084589994160194511935568425293,
                0.4358665215084589994160194511935568425293,
                0.0,
                0.0,
            ),
            (
                0.2576482460664272457999960162840797092643,
                -0.09351476757488624521601546747763655179361,
                0.4358665215084589994160194511935568425293,
                0.0,
            ),
            (
                0.1876410243467238251612921441668043913795,
                -0.5952974735769549480478230275858851737782,
                0.9717899277217721234705114322255239398694,
                0.4358665215084589994160194511935568425293,
            ),
        ),
        b=(
            0.1876410243467238251612921441668043913795,
            -0.5952974735769549480478230275858851737782,
            0.9717899277217721234705114322255239398694,
            0.4358665215084589994160194511935568425293,
        ),
        c=(0.0, 0.8717330430169179988320389023871136850586, 0.6, 1.0),
    ),
)
"""Kennedy-Carpenter ARK3(2)4L[2]SA: 4-stage, 3rd-order (2nd-order embedded),
L-stable, stiffly-accurate ESDIRK-paired additive RK scheme. Coefficients
transcribed from the SUNDIALS ARKODE source
(``ARKODE_ARK324L2SA_ERK_4_2_3`` / ``ARKODE_ARK324L2SA_DIRK_4_2_3``)."""


ARK4_3_6L2SA = IMEXTableau(
    explicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0 / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (13861.0 / 62500.0, 6889.0 / 62500.0, 0.0, 0.0, 0.0, 0.0),
            (
                -116923316275.0 / 2393684061468.0,
                -2731218467317.0 / 15368042101831.0,
                9408046702089.0 / 11113171139209.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                -451086348788.0 / 2902428689909.0,
                -2682348792572.0 / 7519795681897.0,
                12662868775082.0 / 11960479115383.0,
                3355817975965.0 / 11060851509271.0,
                0.0,
                0.0,
            ),
            (
                647845179188.0 / 3216320057751.0,
                73281519250.0 / 8382639484533.0,
                552539513391.0 / 3454668386233.0,
                3354512671639.0 / 8306763924573.0,
                4040.0 / 17871.0,
                0.0,
            ),
        ),
        b=(
            82889.0 / 524892.0,
            0.0,
            15625.0 / 83664.0,
            69875.0 / 102672.0,
            -2260.0 / 8211.0,
            1.0 / 4.0,
        ),
        c=(0.0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0),
    ),
    implicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0),
            (8611.0 / 62500.0, -1743.0 / 31250.0, 1.0 / 4.0, 0.0, 0.0, 0.0),
            (
                5012029.0 / 34652500.0,
                -654441.0 / 2922500.0,
                174375.0 / 388108.0,
                1.0 / 4.0,
                0.0,
                0.0,
            ),
            (
                15267082809.0 / 155376265600.0,
                -71443401.0 / 120774400.0,
                730878875.0 / 902184768.0,
                2285395.0 / 8070912.0,
                1.0 / 4.0,
                0.0,
            ),
            (
                82889.0 / 524892.0,
                0.0,
                15625.0 / 83664.0,
                69875.0 / 102672.0,
                -2260.0 / 8211.0,
                1.0 / 4.0,
            ),
        ),
        b=(
            82889.0 / 524892.0,
            0.0,
            15625.0 / 83664.0,
            69875.0 / 102672.0,
            -2260.0 / 8211.0,
            1.0 / 4.0,
        ),
        c=(0.0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0),
    ),
)
"""Kennedy-Carpenter ARK4(3)6L[2]SA: 6-stage, 4th-order (3rd-order embedded),
L-stable, stiffly-accurate ESDIRK-paired additive RK scheme. Coefficients
transcribed from the SUNDIALS ARKODE source
(``ARKODE_ARK436L2SA_ERK_6_3_4`` / ``ARKODE_ARK436L2SA_DIRK_6_3_4``)."""


ARS443 = IMEXTableau(
    explicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0 / 2.0, 0.0, 0.0, 0.0, 0.0),
            (11.0 / 18.0, 1.0 / 18.0, 0.0, 0.0, 0.0),
            (5.0 / 6.0, -5.0 / 6.0, 1.0 / 2.0, 0.0, 0.0),
            (1.0 / 4.0, 7.0 / 4.0, 3.0 / 4.0, -7.0 / 4.0, 0.0),
        ),
        b=(1.0 / 4.0, 7.0 / 4.0, 3.0 / 4.0, -7.0 / 4.0, 0.0),
        c=(0.0, 1.0 / 2.0, 2.0 / 3.0, 1.0 / 2.0, 1.0),
    ),
    implicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0 / 2.0, 0.0, 0.0, 0.0),
            (0.0, 1.0 / 6.0, 1.0 / 2.0, 0.0, 0.0),
            (0.0, -1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 0.0),
            (0.0, 3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0),
        ),
        b=(0.0, 3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0),
        c=(0.0, 1.0 / 2.0, 2.0 / 3.0, 1.0 / 2.0, 1.0),
    ),
)
"""Ascher-Ruuth-Spiteri ARS(4,4,3): 5-stage (1 trivial explicit-first-stage +
4 real stages), 3rd-order, L-stable, *globally* stiffly-accurate additive RK
scheme -- unlike the Kennedy-Carpenter ARK schemes above, both the explicit
and implicit tableaux independently satisfy ``A[-1] == b``
(``is_stiffly_accurate`` is True), so `step` can skip the final combination
and reuse the last stage directly. Coefficients transcribed from Ascher,
Ruuth & Spiteri, "Implicit-explicit Runge-Kutta methods for time-dependent
partial differential equations", Appl. Numer. Math. 25 (1997), Section 2.8,
"A four-stage, third-order combination (4,4,3)" -- cross-checked against the
PETSc ``TSARKIMEXARS443`` source for the `A`/`At` matrices (PETSc's `b`
default differs from the paper's and is *not* used here; the paper's own
printed `b` rows, each equal to their table's last row, are used directly)."""


def _ars222() -> IMEXTableau:
    """Return the Ascher-Ruuth-Spiteri (2,2,2) L-stable GSA scheme."""
    gamma = (2.0 - math.sqrt(2.0)) / 2.0
    delta = 1.0 - 1.0 / (2.0 * gamma)
    return IMEXTableau(
        explicit=ButcherTableau(
            A=(
                (0.0, 0.0, 0.0),
                (gamma, 0.0, 0.0),
                (delta, 1.0 - delta, 0.0),
            ),
            b=(delta, 1.0 - delta, 0.0),
            c=(0.0, gamma, 1.0),
        ),
        implicit=ButcherTableau(
            A=(
                (0.0, 0.0, 0.0),
                (0.0, gamma, 0.0),
                (0.0, 1.0 - gamma, gamma),
            ),
            b=(0.0, 1.0 - gamma, gamma),
            c=(0.0, gamma, 1.0),
        ),
    )


ARS222 = _ars222()
"""Ascher-Ruuth-Spiteri (2,2,2): 3-stage (1 trivial explicit-first-stage + 2
real stages), 2nd-order, L-stable, *globally* stiffly-accurate additive RK
scheme (``is_stiffly_accurate`` is True). Pairs the same 2-stage L-stable
DIRK used in Section 2.5 of the source paper with a 3-stage explicit
companion whose weights are chosen (via `delta = 1 - 1/(2*gamma)`) to equal
its own last row, satisfying the paper's condition (2.3) -- distinct from
`IMEX_SSP2_222` (Pareschi & Russo), which pairs the same DIRK with a
different, non-GSA 2-stage explicit table. Coefficients transcribed from
Ascher, Ruuth & Spiteri, "Implicit-explicit Runge-Kutta methods for
time-dependent partial differential equations", Appl. Numer. Math. 25
(1997), Section 2.6, "L-stable, two-stage, second-order DIRK (2,2,2)"."""


def _ark2_1_3l2sa() -> IMEXTableau:
    """Return the Kennedy-Carpenter ARK2(1)3L[2]SA scheme."""
    gamma = 1.0 - 1.0 / math.sqrt(2.0)
    alpha = (3.0 + 2.0 * math.sqrt(2.0)) / 6.0
    delta = 1.0 / (2.0 * math.sqrt(2.0))
    two_gamma = 2.0 * gamma
    return IMEXTableau(
        explicit=ButcherTableau(
            A=(
                (0.0, 0.0, 0.0),
                (two_gamma, 0.0, 0.0),
                (1.0 - alpha, alpha, 0.0),
            ),
            b=(delta, delta, gamma),
            c=(0.0, two_gamma, 1.0),
        ),
        implicit=ButcherTableau(
            A=(
                (0.0, 0.0, 0.0),
                (gamma, gamma, 0.0),
                (delta, delta, gamma),
            ),
            b=(delta, delta, gamma),
            c=(0.0, two_gamma, 1.0),
        ),
    )


ARK2_1_3L2SA = _ark2_1_3l2sa()
"""Kennedy-Carpenter ARK2(1)3L[2]SA: 3-stage, 2nd-order (1st-order embedded),
L-stable, ESDIRK-paired additive RK scheme. Like `ARK3_2_4L2SA`/
`ARK4_3_6L2SA`, only the implicit part is stiffly accurate
(``implicit_is_stiffly_accurate`` True, ``is_stiffly_accurate`` False), so
`step` uses the partial (implicit-only) shortcut rather than the full
last-stage reuse. Coefficients transcribed from the SUNDIALS ARKODE source
(``ARKODE_ARK2_ERK_3_1_2`` / ``ARKODE_ARK2_DIRK_3_1_2``)."""


ARK5_4_8L2SA = IMEXTableau(
    explicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (41.0 / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (
                367902744464.0 / 2072280473677.0,
                677623207551.0 / 8224143866563.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                1268023523408.0 / 10340822734521.0,
                0.0,
                1029933939417.0 / 13636558850479.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                14463281900351.0 / 6315353703477.0,
                0.0,
                66114435211212.0 / 5879490589093.0,
                -54053170152839.0 / 4284798021562.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                14090043504691.0 / 34967701212078.0,
                0.0,
                15191511035443.0 / 11219624916014.0,
                -18461159152457.0 / 12425892160975.0,
                -281667163811.0 / 9011619295870.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                19230459214898.0 / 13134317526959.0,
                0.0,
                21275331358303.0 / 2942455364971.0,
                -38145345988419.0 / 4862620318723.0,
                -1.0 / 8.0,
                -1.0 / 8.0,
                0.0,
                0.0,
            ),
            (
                -19977161125411.0 / 11928030595625.0,
                0.0,
                -40795976796054.0 / 6384907823539.0,
                177454434618887.0 / 12078138498510.0,
                782672205425.0 / 8267701900261.0,
                -69563011059811.0 / 9646580694205.0,
                7356628210526.0 / 4942186776405.0,
                0.0,
            ),
        ),
        b=(
            -872700587467.0 / 9133579230613.0,
            0.0,
            0.0,
            22348218063261.0 / 9555858737531.0,
            -1143369518992.0 / 8141816002931.0,
            -39379526789629.0 / 19018526304540.0,
            32727382324388.0 / 42900044865799.0,
            41.0 / 200.0,
        ),
        c=(
            0.0,
            41.0 / 100.0,
            2935347310677.0 / 11292855782101.0,
            1426016391358.0 / 7196633302097.0,
            92.0 / 100.0,
            24.0 / 100.0,
            3.0 / 5.0,
            1.0,
        ),
    ),
    implicit=ButcherTableau(
        A=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (41.0 / 200.0, 41.0 / 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (
                41.0 / 400.0,
                -567603406766.0 / 11931857230679.0,
                41.0 / 200.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                683785636431.0 / 9252920307686.0,
                0.0,
                -110385047103.0 / 1367015193373.0,
                41.0 / 200.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                3016520224154.0 / 10081342136671.0,
                0.0,
                30586259806659.0 / 12414158314087.0,
                -22760509404356.0 / 11113319521817.0,
                41.0 / 200.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                218866479029.0 / 1489978393911.0,
                0.0,
                638256894668.0 / 5436446318841.0,
                -1179710474555.0 / 5321154724896.0,
                -60928119172.0 / 8023461067671.0,
                41.0 / 200.0,
                0.0,
                0.0,
            ),
            (
                1020004230633.0 / 5715676835656.0,
                0.0,
                25762820946817.0 / 25263940353407.0,
                -2161375909145.0 / 9755907335909.0,
                -211217309593.0 / 5846859502534.0,
                -4269925059573.0 / 7827059040749.0,
                41.0 / 200.0,
                0.0,
            ),
            (
                -872700587467.0 / 9133579230613.0,
                0.0,
                0.0,
                22348218063261.0 / 9555858737531.0,
                -1143369518992.0 / 8141816002931.0,
                -39379526789629.0 / 19018526304540.0,
                32727382324388.0 / 42900044865799.0,
                41.0 / 200.0,
            ),
        ),
        b=(
            -872700587467.0 / 9133579230613.0,
            0.0,
            0.0,
            22348218063261.0 / 9555858737531.0,
            -1143369518992.0 / 8141816002931.0,
            -39379526789629.0 / 19018526304540.0,
            32727382324388.0 / 42900044865799.0,
            41.0 / 200.0,
        ),
        c=(
            0.0,
            41.0 / 100.0,
            2935347310677.0 / 11292855782101.0,
            1426016391358.0 / 7196633302097.0,
            92.0 / 100.0,
            24.0 / 100.0,
            3.0 / 5.0,
            1.0,
        ),
    ),
)
"""Kennedy-Carpenter ARK5(4)8L[2]SA: 8-stage, 5th-order (4th-order embedded),
L-stable, stiffly-accurate ESDIRK-paired additive RK scheme -- like
`ARK3_2_4L2SA`/`ARK4_3_6L2SA`/`ARK2_1_3L2SA`, only the implicit part is
stiffly accurate. Coefficients transcribed from the SUNDIALS ARKODE source
(``ARKODE_ARK548L2SA_ERK_8_4_5`` / ``ARKODE_ARK548L2SA_DIRK_8_4_5``), and
independently cross-checked via exact `fractions.Fraction` row-sum
arithmetic before being written here."""
