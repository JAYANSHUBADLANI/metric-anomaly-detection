"""Anomaly detectors for univariate KPI time series.

Every method returns a :class:`DetectionResult` so the experiment and plotting
code can treat them uniformly. The four statistical detectors are written from
scratch (NumPy/pandas only) so the mechanics stay inspectable; the
IsolationForest wrapper is a thin adapter around scikit-learn and is only used
as a machine-learning baseline.

Conventions
-----------
* ``x`` is a 1-D array of observations ordered in time.
* ``flags[t] == True`` means "raise an alarm at time ``t``".
* ``score[t]`` is the detector's decision statistic. It is signed where a sign
  is meaningful (z-score, EWMA, residual detector) and non-negative where it is
  not (CUSUM, IsolationForest). Comparing ``|score|`` (or ``score`` for the
  one-sided statistics) against ``threshold`` reproduces ``flags``.
* The z-score, EWMA and CUSUM detectors are *causal*: the statistic at time
  ``t`` only uses observations up to ``t``. The seasonal-decomposition detector
  and the IsolationForest baseline use a training slice that is assumed to be
  in-control; this is stated in their docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    """Container for a detector's output.

    Attributes
    ----------
    flags:
        Boolean array, ``True`` where an alarm is raised.
    score:
        The decision statistic, same length as the input.
    threshold:
        The threshold the statistic is compared against (scalar). Stored for
        plotting and reporting; it does not need to be re-applied.
    name:
        Human-readable detector name.
    info:
        Optional extra arrays/parameters (control limits, baseline mean, ...).
    """

    flags: np.ndarray
    score: np.ndarray
    threshold: float
    name: str = ""
    info: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.flags = np.asarray(self.flags, dtype=bool)
        self.score = np.asarray(self.score, dtype=float)
        if self.flags.shape != self.score.shape:
            raise ValueError("flags and score must have the same shape")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_MAD_TO_STD = 1.4826  # scale that makes MAD a consistent estimator of sigma
#                       for normally distributed data (1 / Phi^-1(0.75)).


def _mad(a: np.ndarray) -> float:
    """Median absolute deviation of a 1-D array (unscaled)."""
    a = np.asarray(a, dtype=float)
    return float(np.median(np.abs(a - np.median(a))))


def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("input series is empty")
    return arr


# --------------------------------------------------------------------------- #
# (a) rolling z-score, classic and robust
# --------------------------------------------------------------------------- #
def rolling_zscore(
    x,
    window: int = 14,
    threshold: float = 3.0,
    robust: bool = False,
    min_periods: Optional[int] = None,
) -> DetectionResult:
    r"""Trailing-window z-score detector.

    For each point the location and scale of the *preceding* ``window`` values
    are estimated (the current point is excluded, so the statistic is causal and
    a point cannot mask itself):

    .. math::

        z_t = \frac{x_t - \mu_{t}}{\sigma_{t}}, \qquad
        \mu_t, \sigma_t \ \text{estimated on } x_{t-w}, \dots, x_{t-1}.

    With ``robust=True`` the mean/standard deviation are replaced by the median
    and a scaled median absolute deviation, ``sigma = 1.4826 * MAD``. The robust
    variant keeps working when the history window is contaminated by earlier
    spikes: a few large values inflate the ordinary standard deviation (and so
    shrink later z-scores, masking real events), whereas the median/MAD barely
    move.

    Parameters
    ----------
    window:
        Number of trailing observations used to estimate location/scale.
    threshold:
        Alarm when ``|z_t| > threshold``.
    robust:
        Use median/MAD instead of mean/std.
    min_periods:
        Minimum number of valid trailing points required before a score is
        produced (defaults to ``window``). During warm-up the score is 0 and no
        alarm is raised.
    """
    x = _as_1d(x)
    if window < 2:
        raise ValueError("window must be >= 2")
    if min_periods is None:
        min_periods = window

    s = pd.Series(x)
    past = s.shift(1).rolling(window=window, min_periods=min_periods)

    if robust:
        center = past.median()
        scale = _MAD_TO_STD * past.apply(_mad, raw=True)
    else:
        center = past.mean()
        scale = past.std(ddof=1)

    center = center.to_numpy()
    scale = scale.to_numpy()

    z = np.zeros_like(x)
    valid = np.isfinite(center) & np.isfinite(scale) & (scale > 0)
    z[valid] = (x[valid] - center[valid]) / scale[valid]

    flags = np.abs(z) > threshold
    name = "robust z-score (MAD)" if robust else "rolling z-score"
    return DetectionResult(
        flags=flags,
        score=z,
        threshold=threshold,
        name=name,
        info={"center": center, "scale": scale, "window": window, "robust": robust},
    )


# --------------------------------------------------------------------------- #
# (b) EWMA control chart
# --------------------------------------------------------------------------- #
def ewma_control_chart(
    x,
    lam: float = 0.2,
    L: float = 3.0,
    warmup: Optional[int] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
) -> DetectionResult:
    r"""Exponentially weighted moving-average control chart.

    The chart statistic is the usual EWMA recursion

    .. math::

        Z_t = \lambda\, x_t + (1 - \lambda)\, Z_{t-1}, \qquad Z_0 = \mu_0,

    monitored against time-varying control limits

    .. math::

        \mu_0 \pm L\,\sigma_0 \sqrt{\tfrac{\lambda}{2 - \lambda}
                \bigl(1 - (1-\lambda)^{2t}\bigr)}.

    The in-control mean ``mu_0`` and standard deviation ``sigma_0`` are estimated
    from the first ``warmup`` observations unless supplied explicitly. Small
    ``lambda`` (0.1-0.3) makes the chart sensitive to small, sustained shifts in
    the mean while smoothing through single-point noise.

    Parameters
    ----------
    lam:
        Smoothing constant :math:`\lambda \in (0, 1]`.
    L:
        Control-limit width in standard-deviation units.
    warmup:
        Number of initial (assumed in-control) points used to estimate
        ``mu_0``/``sigma_0``. Defaults to ``max(20, len(x) // 10)``.
    mu, sigma:
        Optionally provide the in-control mean/std directly.
    """
    x = _as_1d(x)
    n = x.size
    if not (0.0 < lam <= 1.0):
        raise ValueError("lam must be in (0, 1]")
    if warmup is None:
        warmup = max(20, n // 10)
    warmup = int(min(max(warmup, 2), n))

    if mu is None:
        mu = float(np.mean(x[:warmup]))
    if sigma is None:
        sigma = float(np.std(x[:warmup], ddof=1))
    if sigma == 0:
        sigma = 1e-12

    z = np.empty(n)
    ucl = np.empty(n)
    lcl = np.empty(n)
    prev = mu
    ratio = lam / (2.0 - lam)
    for t in range(n):
        prev = lam * x[t] + (1.0 - lam) * prev
        z[t] = prev
        half = L * sigma * np.sqrt(ratio * (1.0 - (1.0 - lam) ** (2 * (t + 1))))
        ucl[t] = mu + half
        lcl[t] = mu - half

    flags = (z > ucl) | (z < lcl)
    # Report a standardized statistic so it is comparable across series.
    asymptotic_sd = sigma * np.sqrt(ratio)
    score = (z - mu) / asymptotic_sd
    return DetectionResult(
        flags=flags,
        score=score,
        threshold=L,
        name="EWMA control chart",
        info={"ewma": z, "ucl": ucl, "lcl": lcl, "mu": mu, "sigma": sigma, "lam": lam},
    )


# --------------------------------------------------------------------------- #
# (c) tabular CUSUM
# --------------------------------------------------------------------------- #
def cusum(
    x,
    k: float = 0.5,
    h: float = 5.0,
    warmup: Optional[int] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    reset: bool = True,
) -> DetectionResult:
    r"""Two-sided tabular CUSUM.

    Observations are standardized with an in-control mean/std and accumulated in
    two one-sided sums that detect upward and downward shifts:

    .. math::

        C^{+}_t = \max\bigl(0,\; C^{+}_{t-1} + y_t - k\bigr), \qquad
        C^{-}_t = \max\bigl(0,\; C^{-}_{t-1} - y_t - k\bigr),

    where :math:`y_t = (x_t - \mu_0)/\sigma_0`. An alarm is raised when either
    sum exceeds the decision interval ``h``. The reference value ``k`` is the
    slack per step: with ``k = 0.5`` the chart is tuned to detect shifts of about
    :math:`1\sigma` (``k`` is half the target shift in sigma units).

    Because it accumulates evidence, CUSUM catches small *sustained* level shifts
    that a point-wise z-score misses, at the cost of ignoring isolated spikes.

    Average run length (ARL)
    ------------------------
    For the standard one-sided chart with ``k = 0.5`` the textbook ARLs are
    roughly:

    ======  ==========  ===================
    ``h``   in-control  1-sigma shift
    ======  ==========  ===================
    4       ~168        ~8.4
    5       ~465        ~10.4
    ======  ==========  ===================

    A large in-control ARL means rare false alarms; a small out-of-control ARL
    means fast detection. The two-sided chart used here has roughly half the
    in-control ARL because either arm can trigger. Use :func:`cusum_arl` to
    estimate these numbers by simulation for a given ``(k, h)``.

    Parameters
    ----------
    k:
        Reference value (slack) in standard-deviation units.
    h:
        Decision interval / alarm threshold in standard-deviation units.
    warmup:
        Number of initial (assumed in-control) points for ``mu_0``/``sigma_0``.
        Defaults to ``max(20, len(x) // 10)``.
    mu, sigma:
        Optionally provide the in-control mean/std directly.
    reset:
        If ``True`` the triggering sum is reset to 0 after each alarm (the usual
        tabular-CUSUM behaviour) so consecutive alarms need fresh evidence.
    """
    x = _as_1d(x)
    n = x.size
    if warmup is None:
        warmup = max(20, n // 10)
    warmup = int(min(max(warmup, 2), n))

    if mu is None:
        mu = float(np.mean(x[:warmup]))
    if sigma is None:
        sigma = float(np.std(x[:warmup], ddof=1))
    if sigma == 0:
        sigma = 1e-12

    y = (x - mu) / sigma
    c_pos = np.zeros(n)
    c_neg = np.zeros(n)
    flags = np.zeros(n, dtype=bool)
    sp = 0.0
    sn = 0.0
    for t in range(n):
        sp = max(0.0, sp + y[t] - k)
        sn = max(0.0, sn - y[t] - k)
        c_pos[t] = sp
        c_neg[t] = sn
        if sp > h or sn > h:
            flags[t] = True
            if reset:
                sp = 0.0
                sn = 0.0

    score = np.maximum(c_pos, c_neg)
    return DetectionResult(
        flags=flags,
        score=score,
        threshold=h,
        name="CUSUM",
        info={"c_pos": c_pos, "c_neg": c_neg, "mu": mu, "sigma": sigma, "k": k},
    )


def cusum_arl(
    k: float = 0.5,
    h: float = 5.0,
    shift: float = 0.0,
    two_sided: bool = True,
    n_sims: int = 3000,
    max_len: int = 20000,
    seed: int = 2026,
) -> float:
    """Estimate the average run length of a CUSUM chart by simulation.

    Draws standard-normal streams with an optional mean ``shift`` (in sigma
    units) and returns the mean number of samples until the first alarm. With
    ``shift=0`` this is the in-control ARL (want it large); with ``shift>0`` it
    is the out-of-control ARL for that shift (want it small).

    Runs that do not alarm within ``max_len`` samples are censored at
    ``max_len``, so very large in-control ARLs are slightly underestimated.
    """
    rng = np.random.default_rng(seed)
    run_lengths = np.empty(n_sims)
    for i in range(n_sims):
        sp = 0.0
        sn = 0.0
        rl = max_len
        # Draw in reasonably sized blocks to keep the loop fast.
        drawn = 0
        block = min(max_len, 4096)
        found = False
        while drawn < max_len and not found:
            size = min(block, max_len - drawn)
            y = rng.normal(shift, 1.0, size=size)
            for j in range(size):
                sp = max(0.0, sp + y[j] - k)
                trig = sp > h
                if two_sided:
                    sn = max(0.0, sn - y[j] - k)
                    trig = trig or (sn > h)
                if trig:
                    rl = drawn + j + 1
                    found = True
                    break
            drawn += size
        run_lengths[i] = rl
    return float(np.mean(run_lengths))


# --------------------------------------------------------------------------- #
# (d) STL-style seasonal decomposition residual detector
# --------------------------------------------------------------------------- #
def seasonal_decompose(
    x,
    period: int = 7,
    train_frac: float = 0.2,
    n_train: Optional[int] = None,
    level_window: Optional[int] = None,
) -> Dict[str, object]:
    r"""Causal level/seasonal/residual decomposition (a lightweight STL).

    * **level** -- a trailing moving average over ``level_window`` observations
      (a multiple of ``period``), ``L_t = mean(x_{t-w+1}, ..., x_t)``. Averaging
      whole periods cancels the seasonal swing and tracks the slow trend without
      peeking into the future.
    * **seasonal** -- a fixed per-phase profile ``s[p]`` learned on the training
      slice as the median of ``x_t - L_t`` for each phase ``p = t mod P`` and
      centred to sum to zero.
    * **residual** -- ``r_t = x_t - L_t - s[t mod P]``.

    The residual is an approximately stationary, deseasonalized stream, which is
    exactly what the control charts (EWMA, CUSUM) need as input. The width of
    ``level_window`` sets a trade-off: one period (the default) reacts quickly
    but absorbs sustained level shifts within a cycle, so it suits a spike/
    seasonality-break detector; several periods keeps sustained shifts visible in
    the residual, which is what a CUSUM needs to accumulate them. Returned as a
    dict of arrays plus the fitted seasonal profile and the training size used.
    """
    x = _as_1d(x)
    n = x.size
    if period < 2:
        raise ValueError("period must be >= 2")
    if n_train is None:
        n_train = int(round(train_frac * n))
    n_train = int(min(max(n_train, period * 2), n))
    if level_window is None:
        level_window = period

    s = pd.Series(x)
    level = s.rolling(window=level_window, min_periods=period).mean().to_numpy()
    # Back-fill the warm-up so residuals are defined from the first sample.
    level = pd.Series(level).bfill().to_numpy()
    detrended = x - level

    phase = np.arange(n) % period
    seasonal_profile = np.zeros(period)
    for p in range(period):
        mask = (phase == p) & (np.arange(n) < n_train) & np.isfinite(detrended)
        if mask.any():
            seasonal_profile[p] = np.median(detrended[mask])
    seasonal_profile -= seasonal_profile.mean()
    seasonal = seasonal_profile[phase]

    residual = x - level - seasonal
    return {
        "level": level,
        "seasonal": seasonal,
        "residual": residual,
        "seasonal_profile": seasonal_profile,
        "period": period,
        "n_train": n_train,
        "level_window": level_window,
    }


def stl_residual_detector(
    x,
    period: int = 7,
    threshold: float = 3.0,
    train_frac: float = 0.2,
    n_train: Optional[int] = None,
) -> DetectionResult:
    r"""Seasonal-decomposition residual detector (a lightweight STL).

    Runs :func:`seasonal_decompose` and raises an alarm on residuals that are
    large relative to the in-control noise. Residuals are standardized with the
    median and scaled MAD of the *training* residuals (a static, robust control
    limit), and an alarm fires when
    ``|r_t - median| / (1.4826 * MAD) > threshold``. Because seasonality is
    modelled explicitly, this detector flags *seasonality breaks* -- e.g. a
    weekend that suddenly behaves like a weekday -- that a raw control chart
    would smear into its limits.

    Parameters
    ----------
    period:
        Length of one seasonal cycle (7 for weekly seasonality on daily data,
        48 for daily seasonality on 30-minute data).
    threshold:
        Alarm when the robust residual z-score exceeds this value.
    train_frac, n_train:
        Size of the leading in-control slice used to learn the seasonal profile
        and residual baseline. ``n_train`` takes precedence if given.
    """
    dec = seasonal_decompose(x, period=period, train_frac=train_frac, n_train=n_train)
    residual = dec["residual"]
    n_train = int(dec["n_train"])

    train_res = residual[:n_train]
    med = float(np.median(train_res))
    scale = _MAD_TO_STD * _mad(train_res)
    if scale == 0:
        scale = float(np.std(train_res)) or 1e-12

    score = (residual - med) / scale
    flags = np.abs(score) > threshold
    return DetectionResult(
        flags=flags,
        score=score,
        threshold=threshold,
        name="STL residual",
        info=dec,
    )


# --------------------------------------------------------------------------- #
# IsolationForest wrapper (machine-learning baseline)
# --------------------------------------------------------------------------- #
def _seasonal_features(x: np.ndarray, period: int) -> np.ndarray:
    """Build a small causal feature matrix for the IsolationForest baseline."""
    n = x.size
    s = pd.Series(x)
    roll_mean = s.rolling(period, min_periods=1).mean()
    roll_std = s.rolling(period, min_periods=1).std(ddof=0).fillna(0.0)
    dev = s - roll_mean                        # deviation from local level
    lag = s.shift(period)
    season_diff = (s - lag).fillna(0.0)        # change vs same phase last cycle
    feats = np.column_stack(
        [
            x,
            roll_mean.to_numpy(),
            roll_std.to_numpy(),
            dev.to_numpy(),
            season_diff.to_numpy(),
        ]
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def isolation_forest_detector(
    x,
    period: int = 7,
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 2026,
) -> DetectionResult:
    """IsolationForest baseline (scikit-learn).

    A handful of causal features (value, trailing mean/std, deviation from the
    local level, and the change versus the same phase one cycle earlier) are fed
    to an :class:`sklearn.ensemble.IsolationForest`. This is included only as an
    off-the-shelf machine-learning reference point; unlike the statistical
    detectors it has no interpretable control limit, and ``contamination`` fixes
    the alarm rate up front rather than a false-alarm probability.

    The score returned is the negative of the model's ``decision_function`` so
    that, as with the other detectors, larger means more anomalous.
    """
    from sklearn.ensemble import IsolationForest

    x = _as_1d(x)
    feats = _seasonal_features(x, period)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(feats)
    raw = model.decision_function(feats)   # higher = more normal
    score = -raw                            # higher = more anomalous
    flags = model.predict(feats) == -1
    threshold = 0.0                         # predict flags exactly where this score goes positive
    return DetectionResult(
        flags=flags,
        score=score,
        threshold=threshold,
        name="IsolationForest",
        info={"contamination": contamination, "period": period},
    )
