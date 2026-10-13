"""Simple, dependency-free baselines for daily EXP gain forecasting."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Callable, Iterable, Sequence

from maplerival.experience import required_exp


WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


@dataclass(frozen=True)
class DailyGain:
    """EXP earned on one calendar day."""

    day: date
    gain: int


@lru_cache(maxsize=None)
def level_start_experience(level: int) -> int:
    """Return cumulative EXP at the beginning of ``level``.

    Character API EXP resets on level-up, so raw ``experience`` values cannot
    be differenced directly across a level boundary.
    """

    if level < 200:
        raise ValueError(f"Level {level} is below the supported EXP table")
    return sum(required_exp(current) for current in range(200, level))


def records_to_daily_gains(records: Sequence[dict]) -> list[DailyGain]:
    """Convert one character's consecutive snapshots to daily EXP gains."""

    if len(records) < 2:
        return []
    ordered = sorted(records, key=lambda item: item["date"])
    parsed_dates = [date.fromisoformat(item["date"]) for item in ordered]
    if len(set(parsed_dates)) != len(parsed_dates):
        raise ValueError("Duplicate snapshot dates found")

    totals: list[int] = []
    for item, snapshot_date in zip(ordered, parsed_dates):
        supplied_weekday = item.get("weekday")
        expected_weekday = WEEKDAY_NAMES[snapshot_date.weekday()]
        if supplied_weekday is not None and supplied_weekday != expected_weekday:
            raise ValueError(
                f"Weekday mismatch on {snapshot_date}: "
                f"expected {expected_weekday}, got {supplied_weekday}"
            )
        level = int(item["level"])
        totals.append(level_start_experience(level) + int(item["experience"]))

    gains: list[DailyGain] = []
    for index in range(1, len(ordered)):
        previous_date = parsed_dates[index - 1]
        current_date = parsed_dates[index]
        if (current_date - previous_date).days != 1:
            raise ValueError(
                f"Non-consecutive snapshots: {previous_date} -> {current_date}"
            )
        gain = totals[index] - totals[index - 1]
        if gain < 0:
            raise ValueError(f"Negative EXP gain on {current_date}: {gain}")
        gains.append(DailyGain(day=current_date, gain=gain))
    return gains


def simple_moving_average(
    history: Sequence[DailyGain], target_dates: Sequence[date]
) -> list[float]:
    """Repeat the mean of the observed window for every forecast date."""

    if not history:
        raise ValueError("Moving average history cannot be empty")
    average = statistics.fmean(item.gain for item in history)
    return [average] * len(target_dates)


def weekday_moving_average(
    history: Sequence[DailyGain], target_dates: Sequence[date]
) -> list[float]:
    """Predict each date using the history mean for the same weekday."""

    if not history:
        raise ValueError("Moving average history cannot be empty")
    by_weekday: dict[int, list[int]] = defaultdict(list)
    for item in history:
        by_weekday[item.day.weekday()].append(item.gain)
    fallback = statistics.fmean(item.gain for item in history)
    weekday_means = {
        weekday: statistics.fmean(values) for weekday, values in by_weekday.items()
    }
    return [weekday_means.get(day.weekday(), fallback) for day in target_dates]


PREDICTORS: dict[
    str, Callable[[Sequence[DailyGain], Sequence[date]], list[float]]
] = {
    "sma_30": simple_moving_average,
    "weekday_sma_30": weekday_moving_average,
}


def regression_metrics(
    actual: Sequence[int | float], predicted: Sequence[int | float]
) -> dict[str, float]:
    """Calculate pooled MAE, RMSE, and R2 score.

    Constant targets follow scikit-learn's finite convention: a perfect model
    receives R2=1 and an imperfect model receives R2=0.
    """

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ")
    if not actual:
        raise ValueError("Metrics require at least one prediction")
    errors = [
        float(estimate) - float(observed)
        for observed, estimate in zip(actual, predicted)
    ]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    mean_actual = statistics.fmean(float(value) for value in actual)
    total_sum_squares = sum((float(value) - mean_actual) ** 2 for value in actual)
    residual_sum_squares = sum(squared_errors)
    if total_sum_squares == 0:
        r2_score = 1.0 if residual_sum_squares == 0 else 0.0
    else:
        r2_score = 1.0 - residual_sum_squares / total_sum_squares
    return {
        "mae": statistics.fmean(absolute_errors),
        "rmse": math.sqrt(statistics.fmean(squared_errors)),
        "r2_score": r2_score,
    }


def _summarize_character_metrics(
    metrics_by_character: Iterable[dict[str, float]],
) -> dict[str, float]:
    values = list(metrics_by_character)
    return {
        "mean_mae": statistics.fmean(item["mae"] for item in values),
        "median_mae": statistics.median(item["mae"] for item in values),
        "mean_rmse": statistics.fmean(item["rmse"] for item in values),
        "median_rmse": statistics.median(item["rmse"] for item in values),
    }


def evaluate_forecaster(
    series_by_character: dict[str, Sequence[DailyGain]],
    model_name: str,
    *,
    lookback_days: int = 30,
    horizon_days: int = 30,
    latest_origin_only: bool = False,
) -> dict:
    """Run character-level rolling-origin evaluation for a baseline model."""

    try:
        predictor = PREDICTORS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline model: {model_name}") from exc
    return evaluate_predictor(
        series_by_character,
        predictor,
        model_name,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        latest_origin_only=latest_origin_only,
    )


def evaluate_predictor(
    series_by_character: dict[str, Sequence[DailyGain]],
    predictor: Callable[[Sequence[DailyGain], Sequence[date]], list[float]],
    model_name: str,
    *,
    lookback_days: int = 30,
    horizon_days: int = 30,
    latest_origin_only: bool = False,
) -> dict:
    """Evaluate any predictor with the same character-level protocol."""

    if lookback_days <= 0 or horizon_days <= 0:
        raise ValueError("Lookback and horizon must be positive")

    all_actual: list[int] = []
    all_predicted: list[float] = []
    actual_by_horizon: list[list[int]] = [[] for _ in range(horizon_days)]
    predicted_by_horizon: list[list[float]] = [[] for _ in range(horizon_days)]
    character_metrics: list[dict[str, float]] = []
    window_count = 0

    for character_id, observations in sorted(series_by_character.items()):
        required_length = lookback_days + horizon_days
        possible_windows = len(observations) - required_length + 1
        if possible_windows <= 0:
            raise ValueError(
                f"Character {character_id} has {len(observations)} daily gains; "
                f"at least {required_length} are required"
            )
        starts = [possible_windows - 1] if latest_origin_only else range(possible_windows)
        character_actual: list[int] = []
        character_predicted: list[float] = []
        for start in starts:
            history = observations[start : start + lookback_days]
            target = observations[
                start + lookback_days : start + lookback_days + horizon_days
            ]
            predictions = predictor(history, [item.day for item in target])
            actual = [item.gain for item in target]
            character_actual.extend(actual)
            character_predicted.extend(predictions)
            all_actual.extend(actual)
            all_predicted.extend(predictions)
            for offset, (observed, estimate) in enumerate(zip(actual, predictions)):
                actual_by_horizon[offset].append(observed)
                predicted_by_horizon[offset].append(estimate)
            window_count += 1
        character_metrics.append(
            regression_metrics(character_actual, character_predicted)
        )

    horizon_metrics = []
    for index, (actual, predicted) in enumerate(
        zip(actual_by_horizon, predicted_by_horizon), start=1
    ):
        horizon_metrics.append(
            {"horizon_day": index, **regression_metrics(actual, predicted)}
        )
    return {
        "model": model_name,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "origin_policy": "latest_only" if latest_origin_only else "all_rolling_origins",
        "character_count": len(series_by_character),
        "window_count": window_count,
        "prediction_count": len(all_actual),
        "metrics": regression_metrics(all_actual, all_predicted),
        "per_character_summary": _summarize_character_metrics(character_metrics),
        "metrics_by_horizon": horizon_metrics,
    }
