# Metric Anomaly Detection

Statistical anomaly detection for business KPIs and time series, with the core
methods written **from scratch** on top of NumPy / pandas / SciPy. The point of
the project is not to wrap a library but to understand *why* each classic
detector behaves the way it does, and to measure that behaviour honestly on
data where I know the right answer.

Everything is driven by a single seed (`2026`) and the results tables below are
the actual numbers written to `results/` by the experiment script: nothing in
this README is hand-typed or aspirational.

---

## The problem

If you own a KPI (sign-ups per day, checkout latency, taxi trips per hour) you
usually watch it on a dashboard and hope you notice when something breaks. That
does not scale. You want an alarm that fires when the metric does something it
should not, and stays quiet the other 99% of the time. In practice "something it
should not" comes in a few flavours:

- **spikes**: a single reading shoots up or down (a logging bug, a viral post);
- **level shifts**: the metric settles at a new baseline and stays there (a
  pricing change, a broken integration that silently drops 15% of events);
- **seasonality breaks**: the daily/weekly *shape* changes (a holiday, a demand
  shock).

No single detector is best at all three, and the interesting engineering
question is *which method to reach for when*, and how to keep the false-alarm
rate low enough that people still trust the alerts. That trade-off is what this
repo measures.

---

## The methods

All four statistical detectors share one interface (`DetectionResult` with a
boolean `flags` array and a real-valued `score`), so the experiment code can
treat them uniformly. Implementations live in `src/anomaly/methods.py`.

### 1. Rolling z-score (classic + robust)

For each point I estimate the mean and spread of the *preceding* window (the
current point is excluded, so it cannot hide itself) and standardise:

```
z_t = (x_t - mean_of_last_w) / std_of_last_w        # classic
z_t = (x_t - median_of_last_w) / (1.4826 * MAD)     # robust
```

`MAD` is the median absolute deviation; the `1.4826` makes it match the standard
deviation for Gaussian data. The robust variant matters when the recent history
is itself dirty: a couple of past spikes inflate the ordinary standard deviation
and *mask* the next real event, while the median/MAD barely move.

### 2. EWMA control chart

An exponentially weighted moving average smooths through single-point noise but
reacts to small persistent shifts:

```
Z_t = λ·x_t + (1-λ)·Z_{t-1}
control limits:  μ₀ ± L·σ₀·sqrt( (λ/(2-λ)) · (1 - (1-λ)^{2t}) )
```

Small `λ` (0.1-0.3) makes it sensitive to slow drifts. Because the chart assumes
a stationary target, I feed it the deseasonalised residual (see below), not the
raw KPI.

### 3. CUSUM (tabular)

CUSUM accumulates standardised deviations, so tiny biases that a point-wise test
never notices eventually add up and trip an alarm:

```
C⁺_t = max(0, C⁺_{t-1} + y_t - k)      y_t = (x_t - μ₀)/σ₀
C⁻_t = max(0, C⁻_{t-1} - y_t - k)      alarm when C⁺ or C⁻ > h
```

`k` is the slack (with `k = 0.5` the chart is tuned for ~1σ shifts) and `h` is
the decision threshold. Its behaviour is summarised by the **Average Run
Length**: how long between false alarms (in-control) versus how long to detect
(out-of-control). `cusum_arl()` estimates both by simulation; for `k=0.5, h=5`
I measure ARL₀ ≈ **462** samples between false alarms against ARL₁ ≈ **10.4**
samples to catch a 1σ shift, in line with the textbook values.

### 4. STL-style seasonal-decomposition residual detector

A lightweight, causal decomposition: a one-period trailing average estimates the
level, a fixed per-phase profile (learned on a clean training slice) captures
the seasonal shape, and the leftover residual is thresholded with a robust
z-score. Because seasonality is modelled explicitly, this is the detector that
sees a *seasonality break* (a weekend that suddenly behaves like a weekday),
which the raw control charts smear into their limits.

### Baseline: IsolationForest

A thin wrapper around scikit-learn's `IsolationForest` on a few causal features
(value, trailing mean/std, deviation from the local level, change versus the
same phase one cycle earlier). It is here purely as an off-the-shelf
machine-learning reference point; unlike the statistical detectors it has no
interpretable control limit and its `contamination` parameter fixes the alarm
rate up front.

### How each detector is fed

The rolling z-scores use a trailing window that adapts to trend and seasonality
on their own, so they run on the **raw** series. The global control charts
(EWMA, CUSUM) assume a stationary target, so they run on the **deseasonalised
residual**. The STL detector does its own decomposition; IsolationForest gets
the raw series plus the engineered features. Thresholds are fixed, conventional
values (z = 3, EWMA L = 3, CUSUM k = 0.5/h = 5), never tuned against the labels.

---

## How I score detectors

Real anomaly labels come as *ranges* ("that whole weekend was weird"), and an
alert is useful if it fires *somewhere* inside the event without crying wolf the
rest of the time. So `src/anomaly/evaluate.py` reports:

- **Point-adjusted precision / recall / F1**: if a detector flags at least one
  point inside a labelled segment, the whole segment counts as caught (Xu et
  al., 2018); points flagged outside any segment still count against precision.
- **Detection delay**: samples between the start of an event and the first
  alarm inside it.
- **False-alarm rate**: the fraction of genuinely normal points that get
  flagged.

---

## Results

Two layers of validation: a synthetic KPI where I control the ground truth
exactly, and the real NYC-taxi series from the Numenta Anomaly Benchmark.

### A. Synthetic KPI (known ground truth)

730 daily points = trend + weekly seasonality + noise, with 8 injected events
(spikes, level shifts, seasonality breaks) covering ~10% of the series. Metrics
are point-adjusted; delay is in samples (1 sample = 1 day).

| method | precision | recall | F1 | false-alarm rate | mean delay |
| --- | --- | --- | --- | --- | --- |
| rolling z-score | 0.800 | 0.273 | 0.407 | 0.009 | 0.5 |
| robust z-score (MAD) | 0.878 | 0.739 | 0.802 | 0.014 | 4.0 |
| EWMA control chart | 0.752 | 0.966 | 0.846 | 0.044 | 3.4 |
| **CUSUM** | 0.924 | 0.966 | **0.944** | 0.011 | 1.8 |
| STL residual | 0.868 | 0.523 | 0.652 | 0.011 | 0.7 |
| IsolationForest | 0.918 | 0.511 | 0.657 | 0.006 | 0.6 |

The averages hide the real story, which is **recall broken down by anomaly
type**:

| method | spikes | level shifts | seasonality breaks |
| --- | --- | --- | --- |
| rolling z-score | 0.75 | **0.00** | 0.50 |
| robust z-score (MAD) | 0.50 | 0.50 | 1.00 |
| EWMA control chart | 0.25 | **1.00** | 1.00 |
| CUSUM | 0.25 | **1.00** | 1.00 |
| STL residual | 1.00 | 0.00 | 1.00 |
| IsolationForest | 0.75 | 0.00 | 1.00 |

This is the headline finding, and it matches the theory: the point-wise
**rolling z-score catches 0% of the small level shifts** (its trailing window
just re-baselines to the new level), while **CUSUM catches 100% of them** by
accumulating the small persistent bias. Meanwhile the STL residual detector is
the only method that catches every spike, because it strips out trend and
seasonality before looking for outliers.

### B. Two focused demonstrations

**Small level shift: CUSUM vs. z-score** (`results/demo_level_shift.png`). A
+0.8σ shift is injected halfway through an otherwise stationary stream. The
point-wise z-score raises only **2** alarms in the entire shifted region (and
those are just noise); CUSUM locks on **11 samples** after the shift begins and
stays lit.

**Contaminated history: robust vs. classic z-score**
(`results/demo_contamination.png`). Three large spikes are placed just before a
genuine +5σ anomaly. The classic z-score sees the target at only **|z| = 0.60**
(no alarm) because those spikes inflated its window standard deviation; the
robust MAD version sees **|z| = 3.64** and fires. Same data, one line of code
different, opposite outcome.

### C. Real data: NYC taxi demand (NAB)

10,320 half-hourly passenger counts (2014-07-01 → 2015-01-31) from the
[Numenta Anomaly Benchmark](https://github.com/numenta/NAB). NAB ships
hand-labelled windows around five real events: the **NYC Marathon**,
**Thanksgiving**, **Christmas**, **New Year's Day** and the **January 2015
blizzard**. Metrics are point-adjusted; delay is in samples (1 sample = 30 min,
so ×0.5 for hours).

| method | precision | recall | F1 | false-alarm rate | mean delay |
| --- | --- | --- | --- | --- | --- |
| rolling z-score | 1.000 | 0.200 | 0.333 | 0.000 | 139.0 |
| robust z-score (MAD) | 0.747 | 0.600 | 0.666 | 0.023 | 57.3 |
| EWMA control chart | 0.403 | 1.000 | 0.575 | 0.165 | 38.8 |
| CUSUM | 0.707 | 1.000 | 0.828 | 0.046 | 29.8 |
| STL residual | 0.630 | 1.000 | 0.773 | 0.066 | 70.0 |
| **IsolationForest** | 0.909 | 1.000 | **0.952** | 0.011 | 97.4 |

On messy real data the ranking flips: the machine-learning baseline
(**IsolationForest, F1 = 0.95**) and **CUSUM (F1 = 0.83)** come out on top,
catching all five holidays. The raw point-wise z-score is nearly useless here
(recall 0.20) because a single half-hour reading almost never looks extreme
against the enormous daily swing, and **EWMA over-alarms** (false-alarm rate
0.165) because the real residual is autocorrelated rather than white, an honest
reminder that a textbook chart's assumptions matter.

### Figures

`results/` contains, all regenerated on every run:

- `synthetic_overlay.png`: the KPI with every method's detections;
- `nyc_taxi_overlay.png`: the same for the taxi series;
- `demo_level_shift.png`, `demo_contamination.png`: the two demonstrations;
- `synthetic_metrics.csv`, `synthetic_by_type.csv`, `nyc_taxi_metrics.csv`,
  `method_comparison.md`: the numbers above.

---

## Which method when

| If you care about… | reach for | why |
| --- | --- | --- |
| single-point spikes | rolling z-score / STL residual | a point-wise test is all you need |
| dirty history / outlier-heavy data | **robust z-score (MAD)** | median/MAD ignore the contamination |
| small, sustained level shifts | **CUSUM** (or EWMA) | it accumulates a bias a point test misses |
| seasonality / shape breaks | **STL residual** | it models the seasonal shape explicitly |
| messy real streams, no time to tune | **IsolationForest** | strong general baseline, but no control limit |

A practical rule of thumb from these experiments: deseasonalise first, then run
a **CUSUM** for drifts and a **robust z-score** (or STL residual) for spikes, and
keep an IsolationForest around as a sanity check.

---

## Project structure

```
metric-anomaly-detection/
├── src/anomaly/
│   ├── methods.py        # the detectors (from scratch) + IsolationForest wrapper
│   ├── evaluate.py       # point-adjusted P/R/F1, detection delay, false-alarm rate
│   ├── datasets.py       # synthetic KPI generator, demos, NYC-taxi loader + labels
│   └── experiments.py    # runs everything and writes results/
├── tests/                # 47 pytest tests (hand-computed cases, ARL, metrics, ...)
├── results/              # CSV + Markdown + PNG, all reproducible
├── data/                 # cached nyc_taxi.csv
├── requirements.txt
├── pyproject.toml
└── LICENSE               # MIT
```

## Setup & usage

```bash
# 1. install
pip install -r requirements.txt          # or: pip install -e .

# 2. reproduce every table and figure in results/
python -m anomaly.experiments            # needs src on the path; see below

# 3. run the tests
pytest
```

If you are not installing the package, put `src` on the path first, e.g.
`PYTHONPATH=src python -m anomaly.experiments`. The taxi data is cached in
`data/nyc_taxi.csv`; if it is missing the loader downloads it from the NAB
repository, and if that download fails the run falls back to the synthetic-only
benchmark and says so.

Using the detectors directly:

```python
import numpy as np
from anomaly.methods import cusum, rolling_zscore
from anomaly.evaluate import evaluate

x = np.load_your_series()
result = cusum(x, k=0.5, h=5.0)          # -> DetectionResult(flags, score, ...)
print(result.flags.nonzero()[0])          # indices that alarmed
```

## Reproducibility

Every random draw goes through `numpy.random.default_rng(2026)`, so the synthetic
series, the injected anomalies and the IsolationForest are identical run to run.
The real taxi series is a fixed public dataset. Re-running `python -m
anomaly.experiments` overwrites `results/` with byte-stable CSVs.

## Data & credits

The real series and its anomaly labels come from the Numenta Anomaly Benchmark
(`realKnownCause/nyc_taxi.csv` and `labels/combined_windows.json`). The
point-adjustment evaluation protocol follows Xu et al., "Unsupervised Anomaly
Detection via Variational Auto-Encoder…" (WWW 2018). Everything else is my own.

## Limitations & next steps

- The decomposition is deliberately simple (trailing means, a fixed seasonal
  profile). A full STL or a model that handles multiple seasonalities
  simultaneously would help on the taxi data, which has both daily and weekly
  cycles.
- Thresholds are fixed conventional values; a production system would calibrate
  them to a target false-alarm budget per stream.
- Everything here is univariate. Multivariate KPI monitoring (correlated
  metrics) is the natural extension.

## License

MIT, see [LICENSE](LICENSE).
