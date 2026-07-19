"""Tests for the synthetic generators and the NYC-taxi loader (offline, cached)."""

import os

import numpy as np

from anomaly.datasets import (
    make_contamination_demo,
    make_level_shift_demo,
    make_synthetic_kpi,
    load_nyc_taxi,
)


def test_synthetic_kpi_is_reproducible():
    a = make_synthetic_kpi(seed=2026)
    b = make_synthetic_kpi(seed=2026)
    assert np.array_equal(a.series.to_numpy(), b.series.to_numpy())
    assert np.array_equal(a.y_true, b.y_true)


def test_synthetic_kpi_shape_and_labels():
    kpi = make_synthetic_kpi(seed=2026, n_days=730)
    assert len(kpi.series) == 730
    assert kpi.y_true.dtype == bool
    assert kpi.y_true.any()                       # some anomalies exist
    assert not kpi.y_true.all()                   # but most of the series is normal
    assert np.isfinite(kpi.series.to_numpy()).all()


def test_synthetic_events_lie_inside_labelled_region():
    kpi = make_synthetic_kpi(seed=2026)
    for event in kpi.events:
        s, e = int(event["start"]), int(event["end"])
        assert kpi.y_true[s : e + 1].all()
    # every anomaly type is represented
    types = {e["type"] for e in kpi.events}
    assert {"spike", "level_shift", "seasonality_break"} <= types


def test_synthetic_clean_component_has_no_labels_baked_in():
    kpi = make_synthetic_kpi(seed=2026)
    # the "clean" component is the series before anomalies were injected
    assert kpi.components["clean"].shape == kpi.series.to_numpy().shape
    assert not np.array_equal(kpi.components["clean"], kpi.series.to_numpy())


def test_level_shift_demo_is_reproducible_and_labelled():
    a = make_level_shift_demo(seed=2026)
    b = make_level_shift_demo(seed=2026)
    assert np.array_equal(a.x, b.x)
    start = a.meta["shift_start"]
    assert a.y_true[start:].all()
    assert not a.y_true[:start].any()


def test_contamination_demo_marks_target_and_spikes():
    demo = make_contamination_demo(seed=2026)
    assert demo.y_true[demo.meta["target_idx"]]
    for i in demo.meta["contam_idx"]:
        assert demo.y_true[i]


def test_load_nyc_taxi_from_cache(tmp_path):
    # A tiny stand-in CSV so the loader never touches the network.
    csv = tmp_path / "nyc_taxi.csv"
    csv.write_text(
        "timestamp,value\n"
        "2014-07-01 00:00:00,100\n"   # outside every window -> normal
        "2014-11-02 12:00:00,200\n"   # inside the NYC-marathon window -> anomaly
        "2015-01-27 00:00:00,300\n"   # inside the blizzard window -> anomaly
    )
    ds = load_nyc_taxi(cache_path=str(csv), download=False)
    assert len(ds.series) == 3
    assert ds.y_true.tolist() == [False, True, True]
    assert ds.series.index.is_monotonic_increasing
    assert len(ds.windows) == 5


def test_load_nyc_taxi_missing_cache_without_download(tmp_path):
    missing = tmp_path / "nope.csv"
    try:
        load_nyc_taxi(cache_path=str(missing), download=False)
        assert False, "expected an error when the cache is missing and download is off"
    except (FileNotFoundError, RuntimeError):
        pass
