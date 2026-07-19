"""Tests for the evaluation metrics, checked against hand-computed toy cases."""

import numpy as np

from anomaly.evaluate import (
    detection_delay,
    evaluate,
    false_alarm_rate,
    point_adjust,
    precision_recall_f1,
    segments_from_labels,
)


def test_segments_from_labels_multiple_runs():
    y = [0, 1, 1, 0, 1, 0, 0, 1]
    assert segments_from_labels(y) == [(1, 2), (4, 4), (7, 7)]


def test_segments_from_labels_empty():
    assert segments_from_labels([0, 0, 0]) == []


def test_segments_from_labels_all_true():
    assert segments_from_labels([1, 1, 1]) == [(0, 2)]


def test_point_adjust_marks_whole_segment_on_single_hit():
    y = [0, 1, 1, 1, 0]
    pred = [0, 0, 1, 0, 0]
    assert list(point_adjust(pred, y)) == [False, True, True, True, False]


def test_point_adjust_no_hit_leaves_segment_untouched():
    y = [0, 1, 1, 0]
    pred = [0, 0, 0, 0]
    assert list(point_adjust(pred, y)) == [False, False, False, False]


def test_point_adjust_keeps_outside_false_positives():
    y = [0, 1, 1, 0, 0]
    pred = [0, 0, 1, 0, 1]  # one hit inside, one false positive outside
    assert list(point_adjust(pred, y)) == [False, True, True, False, True]


def test_prf_perfect_detection():
    y = [0, 1, 1, 0, 1]
    m = precision_recall_f1(y, y)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_prf_handcomputed_with_point_adjust():
    # segment (1,3); one point flagged inside -> whole segment TP (3 points),
    # plus one false positive outside.
    y = [0, 1, 1, 1, 0, 0]
    pred = [0, 0, 1, 0, 0, 1]
    m = precision_recall_f1(y, pred, adjust=True)
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 0
    assert m["precision"] == 3 / 4
    assert m["recall"] == 1.0
    assert abs(m["f1"] - 2 * (0.75 * 1.0) / (0.75 + 1.0)) < 1e-9


def test_prf_no_predictions_conventions():
    y = [0, 1, 1, 0]
    m = precision_recall_f1(y, [0, 0, 0, 0])
    assert m["precision"] == 1.0  # nothing predicted -> no false positives
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_prf_raw_vs_adjusted_differs():
    y = [0, 1, 1, 1, 0]
    pred = [0, 0, 1, 0, 0]
    raw = precision_recall_f1(y, pred, adjust=False)
    adj = precision_recall_f1(y, pred, adjust=True)
    assert adj["recall"] > raw["recall"]


def test_detection_delay_basic():
    y = [0, 0, 1, 1, 1, 1, 0]
    pred = [0, 0, 0, 0, 1, 0, 0]  # first hit at index 4, segment starts at 2
    d = detection_delay(y, pred)
    assert d["mean_delay_samples"] == 2.0
    assert d["n_detected"] == 1 and d["n_missed"] == 0


def test_detection_delay_missed_event():
    y = [0, 1, 1, 0]
    pred = [0, 0, 0, 0]
    d = detection_delay(y, pred)
    assert d["n_missed"] == 1
    assert np.isnan(d["mean_delay_samples"])


def test_detection_delay_zero_when_caught_at_onset():
    y = [0, 1, 1, 0]
    pred = [0, 1, 0, 0]
    d = detection_delay(y, pred)
    assert d["mean_delay_samples"] == 0.0


def test_detection_delay_hours_conversion():
    y = [0, 1, 1, 1]
    pred = [0, 0, 1, 0]  # delay 1 sample
    d = detection_delay(y, pred, sampling_interval_min=30.0)
    assert abs(d["mean_delay_hours"] - 0.5) < 1e-9


def test_false_alarm_rate_basic():
    y = [0, 0, 0, 0, 1]        # four negatives
    pred = [1, 0, 0, 0, 0]     # one false positive among the negatives
    assert false_alarm_rate(y, pred) == 0.25


def test_false_alarm_rate_no_negatives():
    assert false_alarm_rate([1, 1, 1], [1, 0, 1]) == 0.0


def test_evaluate_bundle_has_expected_keys():
    y = [0, 1, 1, 0, 0]
    pred = [0, 1, 0, 0, 1]
    m = evaluate(y, pred, sampling_interval_min=60.0)
    for key in ["precision", "recall", "f1", "false_alarm_rate",
                "mean_delay_samples", "n_events", "n_flagged"]:
        assert key in m
