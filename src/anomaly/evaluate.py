"""Evaluation metrics for time-series anomaly detection.

Point anomaly labels in real KPI data almost always come as *ranges* ("demand
was abnormal that weekend"), and a monitoring system is useful as long as it
fires *somewhere* inside the event and does not cry wolf the rest of the time.
The metrics here reflect that:

* **point-adjusted precision / recall / F1** -- the widely used protocol where a
  ground-truth anomaly segment counts as fully detected if the detector flags at
  least one point inside it (Xu et al., 2018). Points flagged outside any
  segment stay counted as false positives, so precision still punishes noisy
  detectors.
* **detection delay** -- how long after an event begins the first alarm arrives.
* **false-alarm rate** -- the fraction of genuinely normal points that are
  flagged.

All functions take plain boolean/0-1 arrays so they are trivial to unit-test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _as_bool(a) -> np.ndarray:
    return np.asarray(a).astype(bool).ravel()


def segments_from_labels(y_true) -> List[Tuple[int, int]]:
    """Return contiguous anomaly segments as ``(start, end)`` inclusive indices."""
    y = _as_bool(y_true)
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(y):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(y) - 1))
    return segments


def point_adjust(pred, y_true) -> np.ndarray:
    """Apply the point-adjustment protocol to a prediction array.

    If any point inside a ground-truth segment is flagged, the whole segment is
    marked as detected. Predictions outside every segment are left untouched.
    """
    pred = _as_bool(pred)
    adjusted = pred.copy()
    for start, end in segments_from_labels(y_true):
        if pred[start : end + 1].any():
            adjusted[start : end + 1] = True
    return adjusted


def precision_recall_f1(y_true, pred, adjust: bool = True) -> Dict[str, float]:
    """Precision, recall and F1, optionally with point adjustment.

    Conventions for degenerate cases: precision is 1.0 when nothing is predicted,
    recall is 1.0 when there are no true anomalies, and F1 is the harmonic mean
    (0.0 when precision + recall is 0).
    """
    y = _as_bool(y_true)
    p = point_adjust(pred, y) if adjust else _as_bool(pred)

    tp = int(np.sum(p & y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))

    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def detection_delay(
    y_true, pred, sampling_interval_min: Optional[float] = None
) -> Dict[str, float]:
    """Per-event detection delay.

    For each ground-truth segment the delay is the offset (in samples) of the
    first flagged point from the start of the segment. Segments with no flagged
    point are counted as missed and excluded from the mean.

    Returns the mean/median delay over detected segments (in samples, and in
    minutes/hours if ``sampling_interval_min`` is given) plus detection counts.
    """
    y = _as_bool(y_true)
    p = _as_bool(pred)
    delays: List[int] = []
    n_missed = 0
    for start, end in segments_from_labels(y):
        hits = np.nonzero(p[start : end + 1])[0]
        if hits.size:
            delays.append(int(hits[0]))
        else:
            n_missed += 1

    out: Dict[str, float] = {
        "n_events": float(len(segments_from_labels(y))),
        "n_detected": float(len(delays)),
        "n_missed": float(n_missed),
        "mean_delay_samples": float(np.mean(delays)) if delays else float("nan"),
        "median_delay_samples": float(np.median(delays)) if delays else float("nan"),
        "max_delay_samples": float(np.max(delays)) if delays else float("nan"),
    }
    if sampling_interval_min is not None and delays:
        out["mean_delay_hours"] = float(np.mean(delays)) * sampling_interval_min / 60.0
    return out


def false_alarm_rate(y_true, pred) -> float:
    """Fraction of truly-normal points that are flagged (per-point false positives).

    Computed on raw predictions (point adjustment only ever affects points inside
    anomaly segments, so it cannot change this number).
    """
    y = _as_bool(y_true)
    p = _as_bool(pred)
    negatives = int(np.sum(~y))
    if negatives == 0:
        return 0.0
    fp = int(np.sum(p & ~y))
    return fp / negatives


def evaluate(
    y_true,
    pred,
    sampling_interval_min: Optional[float] = None,
) -> Dict[str, float]:
    """Bundle every metric into one flat dictionary for reporting.

    Includes point-adjusted precision/recall/F1 (the headline numbers), the raw
    point-wise F1 for reference, detection-delay statistics and the false-alarm
    rate.
    """
    adj = precision_recall_f1(y_true, pred, adjust=True)
    raw = precision_recall_f1(y_true, pred, adjust=False)
    delay = detection_delay(y_true, pred, sampling_interval_min)
    far = false_alarm_rate(y_true, pred)

    out = {
        "precision": adj["precision"],
        "recall": adj["recall"],
        "f1": adj["f1"],
        "raw_f1": raw["f1"],
        "false_alarm_rate": far,
        "n_flagged": int(np.sum(_as_bool(pred))),
        "tp": adj["tp"],
        "fp": adj["fp"],
        "fn": adj["fn"],
    }
    out.update(delay)
    return out
