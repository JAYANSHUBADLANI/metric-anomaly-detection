"""Tests for the detectors: hand-computed recursions, contracts and behaviour."""

import numpy as np
import pytest

from anomaly.datasets import make_contamination_demo, make_level_shift_demo, make_synthetic_kpi
from anomaly.evaluate import false_alarm_rate
from anomaly.methods import (
    _mad,
    cusum,
    cusum_arl,
    ewma_control_chart,
    isolation_forest_detector,
    rolling_zscore,
    seasonal_decompose,
    stl_residual_detector,
)


# --------------------------------------------------------------------------- #
# helpers / contracts
# --------------------------------------------------------------------------- #
def _all_detectors(x, period=7):
    return [
        rolling_zscore(x, window=7),
        rolling_zscore(x, window=7, robust=True),
        ewma_control_chart(x, warmup=20),
        cusum(x, warmup=20),
        stl_residual_detector(x, period=period),
        isolation_forest_detector(x, period=period, random_state=2026),
    ]


@pytest.mark.parametrize("robust", [False, True])
def test_output_contract(robust):
    x = np.sin(np.arange(200) / 3.0) + 100
    res = rolling_zscore(x, window=10, robust=robust)
    assert res.flags.dtype == bool
    assert res.flags.shape == (200,)
    assert res.score.shape == (200,)


def test_all_detectors_return_aligned_arrays():
    rng = np.random.default_rng(0)
    x = 100 + rng.normal(size=210) + 5 * np.sin(np.arange(210))
    for res in _all_detectors(x):
        assert res.flags.shape == (210,)
        assert res.score.shape == (210,)
        assert res.flags.dtype == bool


def test_mad_hand_value():
    # values 1..5 -> median 3, abs devs [2,1,0,1,2] -> median 1
    assert _mad([1, 2, 3, 4, 5]) == 1.0


# --------------------------------------------------------------------------- #
# rolling z-score
# --------------------------------------------------------------------------- #
def test_zscore_constant_series_no_alarms():
    res = rolling_zscore(np.full(50, 7.0), window=10, threshold=3.0)
    assert not res.flags.any()
    assert np.allclose(res.score, 0.0)


def test_zscore_flags_clear_outlier():
    x = np.array([1, -1, 1, -1, 1, -1, 1, -1, 10.0])  # window of ±1 then a spike
    res = rolling_zscore(x, window=8, threshold=3.0)
    assert res.flags[8]
    assert not res.flags[:8].any()          # warm-up region is silent
    assert res.score[8] > 8                 # (10 - 0) / ~1.07


def test_zscore_is_causal_during_warmup():
    x = np.arange(20.0)
    res = rolling_zscore(x, window=10)
    assert not res.flags[:10].any()         # need `window` past points first


def test_robust_zscore_survives_contaminated_history():
    demo = make_contamination_demo(seed=2026)
    tgt = demo.meta["target_idx"]
    classic = rolling_zscore(demo.x, window=30, threshold=3.0)
    robust = rolling_zscore(demo.x, window=30, threshold=3.0, robust=True)
    # classic z is masked by the inflated window std; robust z still fires.
    assert not classic.flags[tgt]
    assert robust.flags[tgt]
    assert abs(robust.score[tgt]) > abs(classic.score[tgt])


# --------------------------------------------------------------------------- #
# EWMA
# --------------------------------------------------------------------------- #
def test_ewma_recursion_matches_by_hand():
    x = np.full(5, 10.0)
    res = ewma_control_chart(x, lam=0.5, L=3.0, mu=0.0, sigma=1.0)
    expected = [5.0, 7.5, 8.75, 9.375, 9.6875]      # Z_t = 0.5*10 + 0.5*Z_{t-1}
    assert np.allclose(res.info["ewma"], expected)


def test_ewma_in_control_constant_series():
    x = np.full(80, 3.0)
    res = ewma_control_chart(x, lam=0.2, L=3.0)   # mu,sigma from warmup -> sigma 0 guarded
    assert not res.flags.any()


def test_ewma_detects_sustained_shift():
    x = np.concatenate([np.zeros(60), np.full(40, 5.0)])
    res = ewma_control_chart(x, lam=0.2, L=3.0, mu=0.0, sigma=1.0, warmup=40)
    assert res.flags[60:].any()


# --------------------------------------------------------------------------- #
# CUSUM
# --------------------------------------------------------------------------- #
def test_cusum_recursion_matches_by_hand():
    x = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    res = cusum(x, k=0.5, h=100.0, mu=0.0, sigma=1.0)   # h high -> never triggers
    assert np.allclose(res.info["c_pos"], [0.0, 0.0, 2.5, 2.0, 1.5])
    assert np.allclose(res.info["c_neg"], [0.0, 0.0, 0.0, 0.0, 0.0])
    assert not res.flags.any()


def test_cusum_resets_after_alarm():
    x = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    res = cusum(x, k=0.5, h=2.0, mu=0.0, sigma=1.0, reset=True)
    assert res.flags[2]                     # 2.5 > 2 -> alarm
    assert res.info["c_pos"][3] == 0.0      # reset wipes the accumulator


def test_cusum_quiet_when_in_control():
    rng = np.random.default_rng(2026)
    x = rng.normal(0.0, 1.0, size=300)
    res = cusum(x, k=0.5, h=5.0, mu=0.0, sigma=1.0)
    # a handful of alarms at most over 300 in-control samples
    assert res.flags.sum() <= 5


def test_cusum_catches_small_shift_that_zscore_misses():
    demo = make_level_shift_demo(seed=2026)
    start = demo.meta["shift_start"]
    z = rolling_zscore(demo.x, window=30, threshold=3.0)
    cu = cusum(demo.x, k=0.5, h=5.0, warmup=100)
    z_in = int(z.flags[start:].sum())
    cu_in = int(cu.flags[start:].sum())
    assert cu_in >= 5           # CUSUM repeatedly signals the shifted regime
    assert cu_in > z_in         # ... far more than the point-wise z-score


def test_cusum_arl_in_control_much_larger_than_out_of_control():
    arl0 = cusum_arl(k=0.5, h=5.0, shift=0.0, n_sims=400, seed=2026)
    arl1 = cusum_arl(k=0.5, h=5.0, shift=1.0, n_sims=400, seed=2026)
    assert arl0 > 100           # rare false alarms in control
    assert arl1 < 30            # quick detection of a 1-sigma shift
    assert arl0 > 5 * arl1


# --------------------------------------------------------------------------- #
# STL / seasonal decomposition
# --------------------------------------------------------------------------- #
def test_seasonal_decompose_leaves_flat_residual_on_clean_signal():
    t = np.arange(210)
    x = 0.1 * t + np.array([2.0, -1, 0, 1, -2, 0, 0])[t % 7]  # trend + seasonality
    dec = seasonal_decompose(x, period=7, train_frac=0.5)
    # Trend and seasonality are removed, so no *structure* is left: after warm-up
    # the residual is flat (a trailing-mean level lags a linear trend by a
    # constant, which the detector later cancels by median-centring).
    resid = dec["residual"][7:]
    assert resid.max() - resid.min() < 1e-6


def test_stl_flags_spike_on_seasonal_series():
    t = np.arange(210)
    x = 50 + np.array([5.0, -3, 0, 2, -4, 0, 0])[t % 7].astype(float)
    x[150] += 40.0
    res = stl_residual_detector(x, period=7, threshold=3.0)
    assert res.flags[150]


# --------------------------------------------------------------------------- #
# IsolationForest baseline
# --------------------------------------------------------------------------- #
def test_isolation_forest_is_deterministic():
    rng = np.random.default_rng(1)
    x = 100 + rng.normal(size=200)
    a = isolation_forest_detector(x, period=7, random_state=2026)
    b = isolation_forest_detector(x, period=7, random_state=2026)
    assert np.array_equal(a.flags, b.flags)


def test_isolation_forest_flags_obvious_outlier():
    rng = np.random.default_rng(2)
    x = 100 + rng.normal(size=200)
    x[100] += 60.0
    res = isolation_forest_detector(x, period=7, contamination=0.05, random_state=2026)
    assert res.flags[100]


# --------------------------------------------------------------------------- #
# no-anomaly series -> near-zero false alarms at the chosen thresholds
# --------------------------------------------------------------------------- #
def _clean_series():
    kpi = make_synthetic_kpi(seed=2026)
    return kpi.components["clean"], kpi.period


def test_statistical_detectors_have_near_zero_false_alarms_when_clean():
    x, period = _clean_series()
    y = np.zeros(len(x), dtype=bool)
    resid = seasonal_decompose(x, period=period, train_frac=0.2, level_window=8 * period)["residual"]
    checks = {
        "rolling z": rolling_zscore(x, window=14, threshold=3.0),
        "robust z": rolling_zscore(x, window=14, threshold=3.0, robust=True),
        "ewma": ewma_control_chart(resid, lam=0.2, L=3.0, warmup=150),
        "cusum": cusum(resid, k=0.5, h=5.0, warmup=150),
        "stl": stl_residual_detector(x, period=period, threshold=3.0),
    }
    for name, res in checks.items():
        assert false_alarm_rate(y, res.flags) < 0.03, name


def test_isolation_forest_false_alarm_rate_tracks_contamination():
    x, period = _clean_series()
    y = np.zeros(len(x), dtype=bool)
    res = isolation_forest_detector(x, period=period, contamination=0.03, random_state=2026)
    # IsolationForest flags roughly `contamination` of points by construction.
    assert false_alarm_rate(y, res.flags) <= 0.06
