"""Datasets for the benchmarks: a controllable synthetic KPI and real taxi demand.

Two layers of validation are supported:

* **Synthetic** -- a daily business KPI with trend, weekly seasonality and noise
  into which anomalies of *known* location and type (spikes, level shifts and
  seasonality breaks) are injected. Because the ground truth is exact, this is
  where precision/recall/delay numbers can be trusted. Everything is driven by a
  single seed so the series is bit-for-bit reproducible.

* **Real** -- the NYC taxi demand series from the Numenta Anomaly Benchmark
  (NAB), a 10,320-point stream of 30-minute passenger counts. NAB ships
  hand-labelled anomaly windows around five real events (the NYC marathon,
  Thanksgiving, Christmas, New Year's Day and the January 2015 blizzard).

Plus two small focused scenarios used to demonstrate specific behaviours
(CUSUM vs. z-score on a small shift; robust vs. classic z-score under a
contaminated history).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# synthetic KPI
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticKPI:
    series: pd.Series               # value indexed by daily DatetimeIndex
    y_true: np.ndarray              # bool, True on anomalous days
    events: List[Dict[str, object]] # each: {type, start, end}
    components: Dict[str, np.ndarray] = field(default_factory=dict)
    noise_sigma: float = 0.0
    period: int = 7


# A weekly shape for a B2B-style KPI: quiet start of week, busy Friday, low
# weekend. Centred to sum to zero so it is pure seasonality (no level change).
_WEEKLY_PROFILE = np.array([10.0, 20.0, 25.0, 35.0, 55.0, -70.0, -75.0])


def make_synthetic_kpi(
    n_days: int = 730,
    seed: int = 2026,
    base_level: float = 1000.0,
    daily_growth: float = 0.5,
    noise_sigma: float = 40.0,
    start_date: str = "2021-01-04",  # a Monday, so phase 0 == Monday
) -> SyntheticKPI:
    """Generate a reproducible daily KPI with labelled anomalies.

    The clean series is ``level (linear trend) + weekly seasonality + Gaussian
    noise``. Anomalies are then injected at fixed offsets (all comfortably after
    the warm-up region the detectors use to learn their baselines):

    * **spikes** -- single-day shocks of +-6 to 7 sigma;
    * **level shifts** -- ~3 weeks held ~1.5 sigma above/below the trend, the
      regime a point-wise z-score adapts to and misses but CUSUM accumulates;
    * **seasonality breaks** -- ~3 weeks where the weekly amplitude is inflated,
      so every day departs from its expected seasonal value (worst on weekends),
      the case the decomposition detector is built for.
    """
    rng = np.random.default_rng(seed)
    n = int(n_days)
    t = np.arange(n)

    trend = base_level + daily_growth * t
    phase = t % 7
    seasonal = _WEEKLY_PROFILE[phase]
    noise = rng.normal(0.0, noise_sigma, size=n)
    clean = trend + seasonal + noise
    value = clean.copy()

    y_true = np.zeros(n, dtype=bool)
    events: List[Dict[str, object]] = []

    def add_spike(day: int, mag: float) -> None:
        value[day] += mag
        y_true[day] = True
        events.append({"type": "spike", "start": day, "end": day})

    def add_level_shift(start: int, end: int, mag: float) -> None:
        value[start : end + 1] += mag
        y_true[start : end + 1] = True
        events.append({"type": "level_shift", "start": start, "end": end})

    def add_seasonality_break(start: int, end: int, factor: float) -> None:
        # Inflate the seasonal swing over the window: every day moves away from
        # its normal seasonal value, weekends most of all.
        idx = np.arange(start, end + 1)
        value[idx] += (factor - 1.0) * _WEEKLY_PROFILE[idx % 7]
        y_true[idx] = True
        events.append({"type": "seasonality_break", "start": start, "end": end})

    s = noise_sigma
    add_spike(200, +6.0 * s)
    add_spike(240, -6.0 * s)
    add_level_shift(300, 320, +2.3 * s)      # small sustained up-shift (~1.5 global sigma)
    add_spike(380, +7.0 * s)
    add_seasonality_break(430, 450, 3.0)     # weekly amplitude tripled
    add_level_shift(520, 540, -2.6 * s)      # small sustained down-shift
    add_spike(600, -6.0 * s)
    add_seasonality_break(640, 660, 3.0)

    index = pd.date_range(start=start_date, periods=n, freq="D")
    series = pd.Series(value, index=index, name="kpi")
    return SyntheticKPI(
        series=series,
        y_true=y_true,
        events=events,
        components={"trend": trend, "seasonal": seasonal, "noise": noise, "clean": clean},
        noise_sigma=noise_sigma,
        period=7,
    )


# --------------------------------------------------------------------------- #
# focused demonstration scenarios
# --------------------------------------------------------------------------- #
@dataclass
class DemoSeries:
    x: np.ndarray
    y_true: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


def make_level_shift_demo(
    seed: int = 2026, n: int = 300, shift_start: int = 150, shift_sigma: float = 0.8
) -> DemoSeries:
    """Near-stationary stream with one small sustained shift.

    Used to show that a point-wise z-score never sees a 0.8-sigma step (each
    single point stays well within its limits) while CUSUM accumulates the small
    bias and eventually fires.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    x[shift_start:] += shift_sigma
    y_true = np.zeros(n, dtype=bool)
    y_true[shift_start:] = True
    return DemoSeries(x=x, y_true=y_true, meta={"shift_start": shift_start, "shift_sigma": shift_sigma})


def make_contamination_demo(
    seed: int = 2026,
    n: int = 200,
    contam_idx: Tuple[int, ...] = (100, 102, 104),
    contam_mag: float = 15.0,
    target_idx: int = 122,
    target_mag: float = 5.0,
) -> DemoSeries:
    """Stream whose recent history is contaminated by a few big spikes.

    The classic rolling z-score's window standard deviation is inflated by the
    earlier spikes, so a genuine moderate anomaly a little later is masked; the
    robust median/MAD version is unaffected and still flags it.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    for i in contam_idx:
        x[i] += contam_mag
    x[target_idx] += target_mag
    y_true = np.zeros(n, dtype=bool)
    y_true[list(contam_idx)] = True
    y_true[target_idx] = True
    return DemoSeries(
        x=x,
        y_true=y_true,
        meta={"contam_idx": list(contam_idx), "target_idx": target_idx, "target_mag": target_mag},
    )


# --------------------------------------------------------------------------- #
# real data: NYC taxi demand (Numenta Anomaly Benchmark)
# --------------------------------------------------------------------------- #
NYC_TAXI_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/"
    "data/realKnownCause/nyc_taxi.csv"
)

# Hand-labelled anomaly windows for realKnownCause/nyc_taxi.csv, taken from NAB's
# labels/combined_windows.json, together with the documented real-world cause of
# each. (Timestamps are the inclusive window bounds NAB scores against.)
NYC_TAXI_WINDOWS: List[Tuple[str, str, str]] = [
    ("2014-10-30 15:30:00", "2014-11-03 22:30:00", "NYC Marathon"),
    ("2014-11-25 12:00:00", "2014-11-29 19:00:00", "Thanksgiving"),
    ("2014-12-23 11:30:00", "2014-12-27 18:30:00", "Christmas"),
    ("2014-12-29 21:30:00", "2015-01-03 04:30:00", "New Year's Day"),
    ("2015-01-24 20:30:00", "2015-01-29 03:30:00", "Jan 2015 blizzard"),
]


@dataclass
class RealDataset:
    series: pd.Series                       # value indexed by 30-min DatetimeIndex
    y_true: np.ndarray                      # bool, True inside a labelled window
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]]
    period: int = 48                        # 48 half-hours == one day
    sampling_interval_min: float = 30.0


def _labels_from_windows(index: pd.DatetimeIndex) -> np.ndarray:
    y = np.zeros(len(index), dtype=bool)
    for start, end, _cause in NYC_TAXI_WINDOWS:
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        y |= np.asarray(mask)
    return y


def load_nyc_taxi(
    cache_path: Optional[str] = None,
    url: str = NYC_TAXI_URL,
    timeout: float = 30.0,
    download: bool = True,
) -> RealDataset:
    """Load the NAB NYC-taxi series, downloading and caching it on first use.

    Parameters
    ----------
    cache_path:
        Where to read/write the CSV. If the file exists it is used directly
        (keeping the benchmark reproducible offline); otherwise the CSV is
        fetched from ``url`` and saved there.
    download:
        If ``False`` and the cache is missing, a ``FileNotFoundError`` is raised
        instead of hitting the network.

    Raises
    ------
    RuntimeError
        If the data can be neither read from cache nor downloaded. The caller
        (see ``experiments.py``) treats this as the signal to fall back to the
        synthetic-only benchmark.
    """
    df: Optional[pd.DataFrame] = None
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    elif download:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "metric-anomaly-detection"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "wb") as fh:
                    fh.write(raw)
            from io import BytesIO

            df = pd.read_csv(BytesIO(raw))
        except Exception as exc:  # network blocked, offline, URL moved, ...
            raise RuntimeError(f"could not download NYC taxi data: {exc}") from exc
    else:
        raise FileNotFoundError(f"cache not found and download disabled: {cache_path}")

    if df is None or not {"timestamp", "value"}.issubset(df.columns):
        raise RuntimeError("unexpected NYC taxi CSV format")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    series = pd.Series(df["value"].astype(float).to_numpy(),
                       index=pd.DatetimeIndex(df["timestamp"]), name="taxi_demand")

    y_true = _labels_from_windows(series.index)
    windows = [
        (pd.Timestamp(s), pd.Timestamp(e), cause) for s, e, cause in NYC_TAXI_WINDOWS
    ]
    return RealDataset(series=series, y_true=y_true, windows=windows)
