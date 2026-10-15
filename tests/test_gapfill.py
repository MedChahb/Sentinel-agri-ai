"""Tests of the part 2 gap-filling logic, on small hand-made series."""

import numpy as np
import pandas as pd
import pytest

from sentinel_agri.gapfill import (
    despike,
    gap_days,
    harmonise_orbits,
    linear_fill,
    make_mask,
    neighbours,
)


@pytest.mark.parametrize("seed", range(5))
def test_make_mask_follows_its_rules(seed):
    observed = np.random.default_rng(seed).random(67) < 0.5
    hidden = make_mask(observed, np.random.default_rng(seed))
    obs = np.flatnonzero(observed)
    assert not (hidden & ~observed).any()  # only observed values are hidden
    assert not hidden[obs[0]] and not hidden[obs[-1]]  # the ends stay visible
    assert abs(hidden.sum() - 0.25 * len(obs)) <= 2  # about a quarter is hidden


def test_make_mask_leaves_short_series_alone():
    observed = np.zeros(67, dtype=bool)
    observed[[3, 10, 20]] = True
    assert not make_mask(observed, np.random.default_rng(0)).any()


def test_linear_fill_is_exact_on_a_straight_line():
    ndvi = np.linspace(0.2, 0.8, 10)[None, :]
    visible = np.zeros_like(ndvi, dtype=bool)
    visible[0, [0, 9]] = True
    np.testing.assert_allclose(linear_fill(ndvi, visible), ndvi)


def test_neighbours_and_gap_days():
    visible = np.array([[True, False, False, True, False, True]])
    prev, nxt = neighbours(visible)
    assert prev[0].tolist() == [-1, 0, 0, 0, 3, 3]
    assert nxt[0].tolist() == [3, 3, 3, 5, 5, -1]
    hidden = np.array([[False, True, True, False, True, False]])
    assert gap_days(hidden, visible)[0, 1] == 15  # steps 0 to 3, 5 days each


def test_harmonise_orbits_removes_a_constant_offset():
    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    s1 = pd.DataFrame(
        {
            "parcel_id": 1,
            "date": dates,
            "orbit": [37, 59] * 4,
            "vv_db": [-10.0, -8.0] * 4,  # orbit 59 reads 2 dB higher
            "vh_db": [-16.0, -15.0] * 4,
            "ratio_db": [-6.0, -7.0] * 4,
        }
    )
    harmonised = harmonise_orbits(s1)
    for column in ["vv_db", "vh_db", "ratio_db"]:
        assert harmonised[column].nunique() == 1


def test_despike_removes_isolated_drops_only():
    nan = np.nan
    ndvi = np.array(
        [
            [0.80, 0.55, 0.82, nan, 0.80],  # one bad date between two good ones: removed
            [0.80, 0.78, 0.25, 0.24, 0.22],  # harvest: the next values stay low, kept
            [0.80, 0.70, 0.62, 0.70, 0.80],  # gradual dip (flowering), kept
        ]
    )
    clean, flagged = despike(ndvi)
    assert flagged.tolist()[0] == [False, True, False, False, False]
    assert not flagged[1:].any()
    assert np.isnan(clean[0, 1]) and clean[1, 2] == 0.25
