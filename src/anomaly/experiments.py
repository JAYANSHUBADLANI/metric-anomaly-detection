"""Run the full benchmark and write everything under ``results/``.

    python -m anomaly.experiments

produces, from a single seed (2026):

* ``results/synthetic_metrics.csv`` / ``results/nyc_taxi_metrics.csv`` -- per
  method precision, recall, F1, detection delay and false-alarm rate;
* ``results/synthetic_by_type.csv`` -- recall broken down by anomaly type,
  which is where "CUSUM catches the small level shifts a z-score misses" shows
  up numerically;
* ``results/method_comparison.md`` -- the same tables in Markdown, plus the two
  focused demonstrations, ready to paste into the README;
* PNG overlays for the synthetic benchmark, the NYC-taxi benchmark and the two
  demos.

If the NYC-taxi data cannot be loaded (no cache and no network) the real-data
section is skipped and the run continues with the synthetic benchmark only, as
noted in the console output and the Markdown report.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from anomaly import datasets
from anomaly.evaluate import evaluate, segments_from_labels
from anomaly.methods import (
    cusum,
    cusum_arl,
    ewma_control_chart,
    isolation_forest_detector,
    rolling_zscore,
    seasonal_decompose,
    stl_residual_detector,
)

SEED = 2026

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Order the detectors are reported in everywhere.
METHOD_ORDER = [
    "rolling z-score",
    "robust z-score (MAD)",
    "EWMA control chart",
    "CUSUM",
    "STL residual",
    "IsolationForest",
]


# --------------------------------------------------------------------------- #
# detector registries (fixed, conventional parameters -- never tuned to labels)
# --------------------------------------------------------------------------- #
Detector = Callable[[np.ndarray], "object"]


# The rolling z-score family uses a trailing window that adapts to trend and
# seasonality on its own, so it runs on the raw series. The global control charts
# (EWMA, CUSUM) assume a stationary target, so they run on the deseasonalized
# residual. The STL detector does its own decomposition; IsolationForest gets the
# raw series plus engineered seasonal features.
def synthetic_detectors(period: int = 7) -> Dict[str, Detector]:
    warmup = 150  # first 150 days are anomaly-free by construction
    # A multi-period level window keeps sustained shifts alive in the residual so
    # CUSUM/EWMA can accumulate them (a one-period window would absorb them).
    resid = lambda x: seasonal_decompose(x, period=period, train_frac=0.2, level_window=8 * period)["residual"]
    return {
        "rolling z-score": lambda x: rolling_zscore(x, window=14, threshold=3.0),
        "robust z-score (MAD)": lambda x: rolling_zscore(x, window=14, threshold=3.0, robust=True),
        "EWMA control chart": lambda x: ewma_control_chart(resid(x), lam=0.2, L=3.0, warmup=warmup),
        "CUSUM": lambda x: cusum(resid(x), k=0.5, h=5.0, warmup=warmup),
        "STL residual": lambda x: stl_residual_detector(x, period=period, threshold=3.0, train_frac=0.2),
        "IsolationForest": lambda x: isolation_forest_detector(x, period=period, contamination=0.03, random_state=SEED),
    }


def real_detectors(period: int = 48) -> Dict[str, Detector]:
    warmup = 2016  # ~6 weeks of anomaly-free history at 30-min cadence
    resid = lambda x: seasonal_decompose(x, period=period, train_frac=0.2, level_window=8 * period)["residual"]
    return {
        "rolling z-score": lambda x: rolling_zscore(x, window=period, threshold=3.0),
        "robust z-score (MAD)": lambda x: rolling_zscore(x, window=period, threshold=3.0, robust=True),
        "EWMA control chart": lambda x: ewma_control_chart(resid(x), lam=0.2, L=3.0, warmup=warmup),
        "CUSUM": lambda x: cusum(resid(x), k=0.5, h=5.0, warmup=warmup),
        "STL residual": lambda x: stl_residual_detector(x, period=period, threshold=4.0, train_frac=0.2),
        "IsolationForest": lambda x: isolation_forest_detector(x, period=period, contamination=0.03, random_state=SEED),
    }


# --------------------------------------------------------------------------- #
# running / scoring
# --------------------------------------------------------------------------- #
def run_benchmark(
    x: np.ndarray,
    y_true: np.ndarray,
    detectors: Dict[str, Detector],
    sampling_interval_min: float,
) -> (pd.DataFrame, Dict[str, object]):
    rows = []
    results = {}
    for name in METHOD_ORDER:
        res = detectors[name](x)
        results[name] = res
        m = evaluate(y_true, res.flags, sampling_interval_min=sampling_interval_min)
        rows.append(
            {
                "method": name,
                "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3),
                "f1": round(m["f1"], 3),
                "false_alarm_rate": round(m["false_alarm_rate"], 4),
                "mean_delay_samples": (
                    round(m["mean_delay_samples"], 1)
                    if np.isfinite(m["mean_delay_samples"])
                    else np.nan
                ),
                "n_detected": int(m["n_detected"]),
                "n_events": int(m["n_events"]),
                "n_flagged": int(m["n_flagged"]),
            }
        )
    return pd.DataFrame(rows), results


def recall_by_type(
    events: List[Dict[str, object]],
    results: Dict[str, object],
) -> pd.DataFrame:
    """Fraction of events of each type that each method detects (>=1 hit inside)."""
    types = ["spike", "level_shift", "seasonality_break"]
    rows = []
    for name in METHOD_ORDER:
        flags = results[name].flags
        row = {"method": name}
        for typ in types:
            evs = [e for e in events if e["type"] == typ]
            if not evs:
                row[typ] = np.nan
                continue
            hit = sum(bool(flags[int(e["start"]) : int(e["end"]) + 1].any()) for e in evs)
            row[typ] = round(hit / len(evs), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _shade_segments(ax, index, y_true, label="labelled anomaly"):
    first = True
    for s, e in segments_from_labels(y_true):
        ax.axvspan(index[s], index[e], color="orange", alpha=0.25, lw=0,
                   label=label if first else None)
        first = False


def plot_overlay(index, x, y_true, results, title, path, max_methods=None):
    names = METHOD_ORDER if max_methods is None else METHOD_ORDER[:max_methods]
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.7 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ax.plot(index, x, color="#3b6ea5", lw=0.7)
        _shade_segments(ax, index, y_true)
        flags = results[name].flags
        ax.scatter(np.asarray(index)[flags], np.asarray(x)[flags], color="crimson",
                   s=12, zorder=3, label="detected")
        ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
    axes[0].set_title(title, fontsize=11)
    handles = [
        Patch(color="orange", alpha=0.25, label="labelled anomaly"),
        plt.Line2D([], [], marker="o", ls="", color="crimson", label="detected point"),
        plt.Line2D([], [], color="#3b6ea5", label="metric"),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_level_shift_demo(demo, path):
    x, y = demo.x, demo.y_true
    z = rolling_zscore(x, window=30, threshold=3.0)
    cu = cusum(x, k=0.5, h=5.0, warmup=100)
    start = demo.meta["shift_start"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax1.plot(x, color="#3b6ea5", lw=0.8, label="series")
    ax1.axvspan(start, len(x) - 1, color="orange", alpha=0.2, lw=0,
                label=f"+{demo.meta['shift_sigma']}σ level shift")
    ax1.scatter(np.where(z.flags)[0], x[z.flags], color="green", s=18, label="z-score alarm")
    ax1.scatter(np.where(cu.flags)[0], x[cu.flags], color="crimson", s=18, marker="x", label="CUSUM alarm")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Small sustained shift: point-wise z-score misses it, CUSUM accumulates it", fontsize=11)

    ax2.plot(np.abs(z.score), color="green", lw=0.9, label="|z-score|")
    ax2.axhline(3.0, color="green", ls=":", lw=1)
    ax2.plot(cu.score, color="crimson", lw=0.9, label="CUSUM statistic")
    ax2.axhline(5.0, color="crimson", ls=":", lw=1)
    ax2.axvspan(start, len(x) - 1, color="orange", alpha=0.2, lw=0)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_xlabel("time (samples)")
    ax2.set_ylabel("statistic")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return z, cu


def plot_contamination_demo(demo, path):
    x, y = demo.x, demo.y_true
    std = rolling_zscore(x, window=30, threshold=3.0)
    rob = rolling_zscore(x, window=30, threshold=3.0, robust=True)
    tgt = demo.meta["target_idx"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax1.plot(x, color="#3b6ea5", lw=0.8, label="series")
    for i in demo.meta["contam_idx"]:
        ax1.axvline(i, color="grey", ls=":", lw=0.8)
    ax1.axvline(tgt, color="orange", lw=1.4, label="target anomaly")
    ax1.scatter(np.where(std.flags)[0], x[std.flags], color="purple", s=16, label="classic z alarm")
    ax1.scatter(np.where(rob.flags)[0], x[rob.flags], color="crimson", s=28, marker="x", label="robust z alarm")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Contaminated history: classic z-score's inflated σ masks the target; robust z survives", fontsize=11)

    ax2.plot(np.abs(std.score), color="purple", lw=0.9, label="|classic z|")
    ax2.plot(np.abs(rob.score), color="crimson", lw=0.9, label="|robust z|")
    ax2.axhline(3.0, color="grey", ls=":", lw=1, label="threshold")
    ax2.axvline(tgt, color="orange", lw=1.4)
    ax2.set_ylim(0, 12)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_xlabel("time (samples)")
    ax2.set_ylabel("|z| at each point")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return std, rob


# --------------------------------------------------------------------------- #
# markdown report
# --------------------------------------------------------------------------- #
def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    md: List[str] = ["# Benchmark results", ""]
    md.append(f"Produced by `python -m anomaly.experiments` (seed = {SEED}).")
    md.append("")

    # ------------------------------------------------------------------ #
    # 1. synthetic benchmark
    # ------------------------------------------------------------------ #
    print("[1/4] synthetic KPI benchmark ...")
    kpi = datasets.make_synthetic_kpi(seed=SEED)
    x = kpi.series.to_numpy()
    syn_df, syn_res = run_benchmark(x, kpi.y_true, synthetic_detectors(kpi.period),
                                    sampling_interval_min=24 * 60)
    syn_df.to_csv(os.path.join(RESULTS_DIR, "synthetic_metrics.csv"), index=False)
    by_type = recall_by_type(kpi.events, syn_res)
    by_type.to_csv(os.path.join(RESULTS_DIR, "synthetic_by_type.csv"), index=False)
    plot_overlay(kpi.series.index, x, kpi.y_true, syn_res,
                 "Synthetic KPI - detections by method", os.path.join(RESULTS_DIR, "synthetic_overlay.png"))
    print(syn_df.to_string(index=False))

    md += ["## 1. Synthetic KPI benchmark", "",
           f"Daily series, {len(x)} days, {int(kpi.y_true.sum())} anomalous days across "
           f"{len(kpi.events)} injected events (spikes, level shifts, seasonality breaks). "
           "Metrics are point-adjusted; mean detection delay is in samples (1 sample = 1 day).", "",
           _df_to_md(syn_df), "",
           "### Recall by anomaly type", "",
           _df_to_md(by_type), ""]

    # ------------------------------------------------------------------ #
    # 2. focused demonstrations
    # ------------------------------------------------------------------ #
    print("\n[2/4] focused demonstrations ...")
    ls = datasets.make_level_shift_demo(seed=SEED)
    z_ls, cu_ls = plot_level_shift_demo(ls, os.path.join(RESULTS_DIR, "demo_level_shift.png"))
    z_flag_in = int(z_ls.flags[ls.meta["shift_start"]:].sum())
    cu_hits = np.where(cu_ls.flags[ls.meta["shift_start"]:])[0]
    cu_delay = int(cu_hits[0]) if cu_hits.size else None
    print(f"  level-shift demo: z-score alarms in shift = {z_flag_in}; "
          f"CUSUM first alarm delay = {cu_delay} samples")

    ct = datasets.make_contamination_demo(seed=SEED)
    std_ct, rob_ct = plot_contamination_demo(ct, os.path.join(RESULTS_DIR, "demo_contamination.png"))
    tgt = ct.meta["target_idx"]
    print(f"  contamination demo: classic |z| at target = {abs(std_ct.score[tgt]):.2f} "
          f"(flag={bool(std_ct.flags[tgt])}); robust |z| at target = {abs(rob_ct.score[tgt]):.2f} "
          f"(flag={bool(rob_ct.flags[tgt])})")

    md += ["## 2. Focused demonstrations", "",
           "**Small level shift (CUSUM vs. z-score).** A +"
           f"{ls.meta['shift_sigma']}σ sustained shift is injected at sample "
           f"{ls.meta['shift_start']}. The point-wise z-score raises "
           f"{z_flag_in} alarms inside the shifted region; CUSUM detects it "
           f"{cu_delay} samples after onset. See `demo_level_shift.png`.", "",
           "**Contaminated history (robust vs. classic z-score).** Three large spikes are placed "
           f"just before a genuine +{ct.meta['target_mag']}σ anomaly at sample {tgt}. "
           f"The classic z-score sees |z| = {abs(std_ct.score[tgt]):.2f} there "
           f"(flag = {bool(std_ct.flags[tgt])}) because the spikes inflated its window "
           f"standard deviation; the robust MAD version sees |z| = {abs(rob_ct.score[tgt]):.2f} "
           f"(flag = {bool(rob_ct.flags[tgt])}). See `demo_contamination.png`.", ""]

    # ------------------------------------------------------------------ #
    # 3. CUSUM ARL sanity check
    # ------------------------------------------------------------------ #
    print("\n[3/4] CUSUM ARL (simulation) ...")
    arl0 = cusum_arl(k=0.5, h=5.0, shift=0.0, seed=SEED)
    arl1 = cusum_arl(k=0.5, h=5.0, shift=1.0, seed=SEED)
    print(f"  k=0.5, h=5:  in-control ARL0 = {arl0:.0f}, 1-sigma ARL1 = {arl1:.1f}")
    md += ["## 3. CUSUM average run length (simulated)", "",
           f"Two-sided chart, k = 0.5, h = 5 (seed {SEED}, 3000 runs): in-control "
           f"ARL0 = {arl0:.0f} samples between false alarms, versus ARL1 = {arl1:.1f} "
           "samples to detect a 1σ shift. The large gap is exactly why CUSUM is the tool "
           "for small persistent shifts.", ""]

    # ------------------------------------------------------------------ #
    # 4. real data: NYC taxi
    # ------------------------------------------------------------------ #
    print("\n[4/4] NYC taxi (NAB) benchmark ...")
    cache = os.path.join(DATA_DIR, "nyc_taxi.csv")
    try:
        real = datasets.load_nyc_taxi(cache_path=cache)
        xr = real.series.to_numpy()
        real_df, real_res = run_benchmark(xr, real.y_true, real_detectors(real.period),
                                          sampling_interval_min=real.sampling_interval_min)
        real_df.to_csv(os.path.join(RESULTS_DIR, "nyc_taxi_metrics.csv"), index=False)
        plot_overlay(real.series.index, xr, real.y_true, real_res,
                     "NYC taxi demand (NAB) - detections by method",
                     os.path.join(RESULTS_DIR, "nyc_taxi_overlay.png"))
        print(real_df.to_string(index=False))
        causes = ", ".join(w[2] for w in real.windows)
        md += ["## 4. NYC taxi demand benchmark (NAB)", "",
               f"{len(xr)} half-hourly observations (2014-07-01 to 2015-01-31). "
               f"Five labelled windows: {causes}. Metrics are point-adjusted; mean detection "
               "delay is in samples (1 sample = 30 min).", "",
               _df_to_md(real_df), ""]
    except Exception as exc:
        print(f"  !! NYC taxi data unavailable ({exc}). Skipping real-data section.")
        md += ["## 4. NYC taxi demand benchmark (NAB)", "",
               f"*Skipped: the dataset could not be loaded ({exc}). "
               "Re-run with network access (or place `nyc_taxi.csv` in `data/`) to populate "
               "this section.*", ""]

    with open(os.path.join(RESULTS_DIR, "method_comparison.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
