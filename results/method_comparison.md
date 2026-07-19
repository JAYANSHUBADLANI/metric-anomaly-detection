# Benchmark results

Produced by `python -m anomaly.experiments` (seed = 2026).

## 1. Synthetic KPI benchmark

Daily series, 730 days, 88 anomalous days across 8 injected events (spikes, level shifts, seasonality breaks). Metrics are point-adjusted; mean detection delay is in samples (1 sample = 1 day).

| method | precision | recall | f1 | false_alarm_rate | mean_delay_samples | n_detected | n_events | n_flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling z-score | 0.8 | 0.273 | 0.407 | 0.0093 | 0.5 | 4 | 8 | 10 |
| robust z-score (MAD) | 0.878 | 0.739 | 0.802 | 0.014 | 4.0 | 5 | 8 | 17 |
| EWMA control chart | 0.752 | 0.966 | 0.846 | 0.0436 | 3.4 | 5 | 8 | 69 |
| CUSUM | 0.924 | 0.966 | 0.944 | 0.0109 | 1.8 | 5 | 8 | 24 |
| STL residual | 0.868 | 0.523 | 0.652 | 0.0109 | 0.7 | 6 | 8 | 21 |
| IsolationForest | 0.918 | 0.511 | 0.657 | 0.0062 | 0.6 | 5 | 8 | 22 |

### Recall by anomaly type

| method | spike | level_shift | seasonality_break |
| --- | --- | --- | --- |
| rolling z-score | 0.75 | 0.0 | 0.5 |
| robust z-score (MAD) | 0.5 | 0.5 | 1.0 |
| EWMA control chart | 0.25 | 1.0 | 1.0 |
| CUSUM | 0.25 | 1.0 | 1.0 |
| STL residual | 1.0 | 0.0 | 1.0 |
| IsolationForest | 0.75 | 0.0 | 1.0 |

## 2. Focused demonstrations

**Small level shift (CUSUM vs. z-score).** A +0.8σ sustained shift is injected at sample 150. The point-wise z-score raises 2 alarms inside the shifted region; CUSUM detects it 11 samples after onset. See `demo_level_shift.png`.

**Contaminated history (robust vs. classic z-score).** Three large spikes are placed just before a genuine +5.0σ anomaly at sample 122. The classic z-score sees |z| = 0.60 there (flag = False) because the spikes inflated its window standard deviation; the robust MAD version sees |z| = 3.64 (flag = True). See `demo_contamination.png`.

## 3. CUSUM average run length (simulated)

Two-sided chart, k = 0.5, h = 5 (seed 2026, 3000 runs): in-control ARL0 = 462 samples between false alarms, versus ARL1 = 10.4 samples to detect a 1σ shift. The large gap is exactly why CUSUM is the tool for small persistent shifts.

## 4. NYC taxi demand benchmark (NAB)

10320 half-hourly observations (2014-07-01 to 2015-01-31). Five labelled windows: NYC Marathon, Thanksgiving, Christmas, New Year's Day, Jan 2015 blizzard. Metrics are point-adjusted; mean detection delay is in samples (1 sample = 30 min).

| method | precision | recall | f1 | false_alarm_rate | mean_delay_samples | n_detected | n_events | n_flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling z-score | 1.0 | 0.2 | 0.333 | 0.0 | 139.0 | 1 | 5 | 1 |
| robust z-score (MAD) | 0.747 | 0.6 | 0.666 | 0.0226 | 57.3 | 3 | 5 | 250 |
| EWMA control chart | 0.403 | 1.0 | 0.575 | 0.1649 | 38.8 | 5 | 5 | 1893 |
| CUSUM | 0.707 | 1.0 | 0.828 | 0.0462 | 29.8 | 5 | 5 | 530 |
| STL residual | 0.63 | 1.0 | 0.773 | 0.0656 | 70.0 | 5 | 5 | 709 |
| IsolationForest | 0.909 | 1.0 | 0.952 | 0.0112 | 97.4 | 5 | 5 | 310 |

