"""Statistical anomaly detection for business KPIs and time series.

The package bundles a small family of detectors that a KPI-monitoring team
would actually reach for, each implemented from scratch on top of NumPy/pandas
so the mechanics are visible and testable:

    * rolling z-score (classic and robust median/MAD variant)
    * EWMA control chart
    * tabular CUSUM
    * seasonal-decomposition residual detector (STL-style)

An IsolationForest wrapper is included as a machine-learning baseline.

See ``anomaly.evaluate`` for point-adjusted precision/recall/F1, detection
delay and false-alarm-rate metrics, and ``anomaly.experiments`` for the
reproducible synthetic and real-data benchmarks.
"""

from anomaly.methods import (
    DetectionResult,
    rolling_zscore,
    ewma_control_chart,
    cusum,
    stl_residual_detector,
    seasonal_decompose,
    isolation_forest_detector,
    cusum_arl,
)
from anomaly.evaluate import (
    evaluate,
    point_adjust,
    segments_from_labels,
    precision_recall_f1,
    detection_delay,
    false_alarm_rate,
)

__version__ = "0.1.0"

__all__ = [
    "DetectionResult",
    "rolling_zscore",
    "ewma_control_chart",
    "cusum",
    "stl_residual_detector",
    "seasonal_decompose",
    "isolation_forest_detector",
    "cusum_arl",
    "evaluate",
    "point_adjust",
    "segments_from_labels",
    "precision_recall_f1",
    "detection_delay",
    "false_alarm_rate",
    "__version__",
]
