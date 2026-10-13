"""Pairwise EXP crossover (rival overtake) labels and features.

All inputs are cumulative EXP arrays (level-aware, see ``baselines``) indexed by
day.  A pair is described at an origin day ``t`` using only information up to
``t``; the label looks at days ``t+1 .. t+horizon``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


LOOKBACK_DAYS = 30
HORIZON_DAYS = 30

FEATURE_NAMES = (
    "log_gap_ratio",
    "log_gap_over_lead_30d",
    "gap_over_lead_30d_gain",
    "log_lead_gain_30d",
    "log_lag_gain_30d",
    "log_lead_gain_7d",
    "log_lag_gain_7d",
    "log_lead_gain_14d",
    "log_lag_gain_14d",
    "lead_active_ratio",
    "lag_active_ratio",
    "lead_trend_7_30",
    "lag_trend_7_30",
    "rate_diff_ratio",
    "linear_eta_days",
    "lead_level",
    "lag_level",
    "level_diff",
    "origin_weekday",
)


@dataclass(frozen=True)
class CrossoverSample:
    pair_index: int
    origin: int
    leader: int
    lagger: int
    crossed: int
    days_to_cross: int | None
    features: tuple[float, ...]


def daily_gains(totals: np.ndarray) -> np.ndarray:
    """Day-over-day gains; negative values (data glitches) are clipped to 0."""

    return np.clip(np.diff(totals), 0, None)


def days_to_crossover(
    lead_totals: np.ndarray, lag_totals: np.ndarray, origin: int, horizon: int
) -> int | None:
    """First day offset k in 1..horizon where lagger >= leader, else None."""

    lead = lead_totals[origin + 1 : origin + 1 + horizon]
    lag = lag_totals[origin + 1 : origin + 1 + horizon]
    if len(lead) < horizon:
        raise ValueError("Origin too late: horizon is not fully observed")
    hits = np.nonzero(lag >= lead)[0]
    return int(hits[0]) + 1 if len(hits) else None


def linear_eta(gap: float, lead_rate: float, lag_rate: float) -> float:
    """Days until the lagger catches up if both keep their mean daily rates.

    Returns ``inf`` when the lagger is not faster.
    """

    closing = lag_rate - lead_rate
    if closing <= 0:
        return math.inf
    return gap / closing


def _log(value: float) -> float:
    return math.log1p(max(value, 0.0))


def pair_features(
    lead_totals: np.ndarray,
    lag_totals: np.ndarray,
    lead_levels: np.ndarray,
    lag_levels: np.ndarray,
    origin: int,
    origin_weekday: int,
    lookback: int = LOOKBACK_DAYS,
) -> tuple[float, ...]:
    """Features for leader/lagger as observed at ``origin`` (inclusive)."""

    if origin < lookback:
        raise ValueError("Not enough history before origin")
    window = slice(origin - lookback, origin + 1)
    lead_gain = daily_gains(lead_totals[window])
    lag_gain = daily_gains(lag_totals[window])

    gap = float(lead_totals[origin] - lag_totals[origin])
    lead_sum = {n: float(lead_gain[-n:].sum()) for n in (7, 14, 30)}
    lag_sum = {n: float(lag_gain[-n:].sum()) for n in (7, 14, 30)}
    lead_rate = lead_sum[30] / 30
    lag_rate = lag_sum[30] / 30
    eta = linear_eta(gap, lead_rate, lag_rate)

    def trend(sums: dict[int, float]) -> float:
        # recent weekly pace relative to 30-day pace; 1.0 means unchanged
        baseline = sums[30] / 30
        return (sums[7] / 7) / baseline if baseline > 0 else 0.0

    return (
        math.log(float(lag_totals[origin]) / float(lead_totals[origin])),
        _log(gap) - _log(lead_sum[30]),
        gap / lead_sum[30] if lead_sum[30] > 0 else 1e6,
        _log(lead_sum[30]),
        _log(lag_sum[30]),
        _log(lead_sum[7]),
        _log(lag_sum[7]),
        _log(lead_sum[14]),
        _log(lag_sum[14]),
        float((lead_gain > 0).mean()),
        float((lag_gain > 0).mean()),
        trend(lead_sum),
        trend(lag_sum),
        (lag_rate - lead_rate) / max(lead_rate, 1.0),
        min(eta, 1e4),
        float(lead_levels[origin]),
        float(lag_levels[origin]),
        float(lead_levels[origin] - lag_levels[origin]),
        float(origin_weekday),
    )


def build_samples(
    totals: np.ndarray,
    levels: np.ndarray,
    weekdays: np.ndarray,
    pairs: list[tuple[int, int]],
    origins: range,
    horizon: int = HORIZON_DAYS,
) -> list[CrossoverSample]:
    """Create samples for each pair at each origin.

    ``totals``/``levels`` have shape (characters, days); ``weekdays`` has shape
    (days,).  Pairs where the totals tie at the origin, or where neither
    character gained EXP in the lookback window, are skipped.
    """

    samples: list[CrossoverSample] = []
    for pair_index, (a, b) in enumerate(pairs):
        for origin in origins:
            if totals[a, origin] == totals[b, origin]:
                continue
            leader, lagger = (a, b) if totals[a, origin] > totals[b, origin] else (b, a)
            window = slice(origin - LOOKBACK_DAYS, origin + 1)
            if daily_gains(totals[leader, window]).sum() == 0 and daily_gains(
                totals[lagger, window]
            ).sum() == 0:
                continue
            eta = days_to_crossover(totals[leader], totals[lagger], origin, horizon)
            samples.append(
                CrossoverSample(
                    pair_index=pair_index,
                    origin=origin,
                    leader=leader,
                    lagger=lagger,
                    crossed=int(eta is not None),
                    days_to_cross=eta,
                    features=pair_features(
                        totals[leader],
                        totals[lagger],
                        levels[leader],
                        levels[lagger],
                        origin,
                        int(weekdays[origin]),
                    ),
                )
            )
    return samples


def neighbor_pairs(
    totals_at_anchor: np.ndarray, indices: np.ndarray, neighbors: int
) -> list[tuple[int, int]]:
    """Pair each character with the next ``neighbors`` characters when sorted
    by cumulative EXP at the anchor day (rivals are close in progress)."""

    order = indices[np.argsort(totals_at_anchor[indices], kind="stable")]
    pairs = []
    for position, a in enumerate(order):
        for b in order[position + 1 : position + 1 + neighbors]:
            pairs.append((int(a), int(b)))
    return pairs
