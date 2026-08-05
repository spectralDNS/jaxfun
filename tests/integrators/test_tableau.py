import pytest

from jaxfun.integrators.tableau import (
    ARK2_1_3L2SA,
    ARK3_2_4L2SA,
    ARK4_3_6L2SA,
    ARK5_4_8L2SA,
    ARS222,
    ARS443,
    IMEX_EULER,
    IMEX_SSP2_222,
    ButcherTableau,
    IMEXTableau,
)

ALL_SCHEMES = {
    "imex_euler": IMEX_EULER,
    "imex_ssp2_222": IMEX_SSP2_222,
    "ark2_1_3l2sa": ARK2_1_3L2SA,
    "ark3_2_4l2sa": ARK3_2_4L2SA,
    "ark4_3_6l2sa": ARK4_3_6L2SA,
    "ark5_4_8l2sa": ARK5_4_8L2SA,
    "ars222": ARS222,
    "ars443": ARS443,
}


@pytest.mark.parametrize("tableau", ALL_SCHEMES.values(), ids=ALL_SCHEMES.keys())
def test_row_sum_consistency(tableau: IMEXTableau) -> None:
    for a_matrix, c in (
        (tableau.explicit.A, tableau.explicit.c),
        (tableau.implicit.A, tableau.implicit.c),
    ):
        for i, row in enumerate(a_matrix):
            assert sum(row) == pytest.approx(c[i], abs=1e-9)


@pytest.mark.parametrize("tableau", ALL_SCHEMES.values(), ids=ALL_SCHEMES.keys())
def test_explicit_tableau_strictly_lower_triangular(tableau: IMEXTableau) -> None:
    for i, row in enumerate(tableau.explicit.A):
        assert row[i] == 0.0
        for j in range(i + 1, len(row)):
            assert row[j] == 0.0


@pytest.mark.parametrize("tableau", ALL_SCHEMES.values(), ids=ALL_SCHEMES.keys())
def test_implicit_tableau_lower_triangular(tableau: IMEXTableau) -> None:
    for i, row in enumerate(tableau.implicit.A):
        for j in range(i + 1, len(row)):
            assert row[j] == 0.0


@pytest.mark.parametrize("tableau", ALL_SCHEMES.values(), ids=ALL_SCHEMES.keys())
def test_weight_consistency_order_one(tableau: IMEXTableau) -> None:
    assert sum(tableau.explicit.b) == pytest.approx(1.0, abs=1e-9)
    assert sum(tableau.implicit.b) == pytest.approx(1.0, abs=1e-9)


def test_imex_euler_is_stiffly_accurate_and_matches_backward_euler_structure() -> None:
    assert IMEX_EULER.is_stiffly_accurate
    assert IMEX_EULER.distinct_diagonal_coeffs == (1.0,)


def test_imex_ssp2_222_is_not_stiffly_accurate() -> None:
    # SSP2(2,2,2) is L-stable but not stiffly accurate (b != last row of A).
    assert not IMEX_SSP2_222.is_stiffly_accurate
    assert len(IMEX_SSP2_222.distinct_diagonal_coeffs) == 1


@pytest.mark.parametrize(
    "tableau",
    [ARK2_1_3L2SA, ARK3_2_4L2SA, ARK4_3_6L2SA, ARK5_4_8L2SA, ARS222, ARS443],
    ids=[
        "ark2_1_3l2sa",
        "ark3_2_4l2sa",
        "ark4_3_6l2sa",
        "ark5_4_8l2sa",
        "ars222",
        "ars443",
    ],
)
def test_ark_schemes_share_single_diagonal_coefficient(tableau: IMEXTableau) -> None:
    # ESDIRK-paired ARK/ARS schemes: first stage explicit (a_11=0), remaining
    # stages share one repeated diagonal coefficient.
    assert len(tableau.distinct_diagonal_coeffs) == 1


@pytest.mark.parametrize("tableau", [ARS222, ARS443], ids=["ars222", "ars443"])
def test_ars_schemes_are_globally_stiffly_accurate(tableau: IMEXTableau) -> None:
    # Unlike the Kennedy-Carpenter ARK schemes, the ARS(2,2,2)/(4,4,3)
    # schemes are constructed so that BOTH tableaux independently satisfy
    # A[-1] == b (Ascher, Ruuth & Spiteri 1997, condition (2.3)), enabling
    # the last-stage shortcut.
    assert tableau.is_stiffly_accurate


@pytest.mark.parametrize(
    "tableau",
    [ARK2_1_3L2SA, ARK3_2_4L2SA, ARK4_3_6L2SA, ARK5_4_8L2SA],
    ids=["ark2_1_3l2sa", "ark3_2_4l2sa", "ark4_3_6l2sa", "ark5_4_8l2sa"],
)
def test_ark_schemes_are_implicit_only_stiffly_accurate(tableau: IMEXTableau) -> None:
    # Kennedy-Carpenter ARK schemes: DIRK part satisfies A[-1] == b (enabling
    # the partial "fold into last stage" shortcut in step()), but the
    # explicit table's own last row differs from b, so they are not globally
    # stiffly accurate.
    assert tableau.implicit_is_stiffly_accurate
    assert not tableau.explicit_is_stiffly_accurate
    assert not tableau.is_stiffly_accurate


def test_imex_ssp2_222_is_neither_form_of_stiffly_accurate() -> None:
    assert not IMEX_SSP2_222.implicit_is_stiffly_accurate
    assert not IMEX_SSP2_222.explicit_is_stiffly_accurate
    assert not IMEX_SSP2_222.is_stiffly_accurate


def test_butcher_tableau_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="square"):
        ButcherTableau(A=((0.0, 0.0),), b=(1.0, 0.0), c=(0.0, 1.0))
    with pytest.raises(ValueError, match="entries"):
        ButcherTableau(A=((0.0, 0.0), (1.0, 0.0)), b=(1.0,), c=(0.0, 1.0))
    with pytest.raises(ValueError, match="sums to"):
        ButcherTableau(A=((0.0, 0.0), (1.0, 0.0)), b=(1.0, 0.0), c=(0.0, 0.5))


def test_imex_tableau_rejects_nonzero_explicit_diagonal() -> None:
    bad_explicit = ButcherTableau(A=((1.0,),), b=(1.0,), c=(1.0,))
    implicit = ButcherTableau(A=((1.0,),), b=(1.0,), c=(1.0,))
    with pytest.raises(ValueError, match="zero diagonal"):
        IMEXTableau(explicit=bad_explicit, implicit=implicit)


def test_imex_tableau_rejects_non_lower_triangular_implicit() -> None:
    explicit = ButcherTableau(A=((0.0, 0.0), (1.0, 0.0)), b=(0.0, 1.0), c=(0.0, 1.0))
    bad_implicit = ButcherTableau(
        A=((0.0, 1.0), (0.0, 1.0)), b=(0.0, 1.0), c=(1.0, 1.0)
    )
    with pytest.raises(ValueError, match="lower triangular"):
        IMEXTableau(explicit=explicit, implicit=bad_implicit)
