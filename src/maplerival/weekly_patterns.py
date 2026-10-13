"""Causal weekly-content pattern model for daily EXP forecasts."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from maplerival.baselines import DailyGain


GROUPS = ("early", "weekend", "late")
WEEKDAYS = tuple(range(7))
THURSDAY = 3
GROUP_BY_WEEKDAY = {
    3: "early",
    4: "early",
    5: "weekend",
    6: "weekend",
    0: "late",
    1: "late",
    2: "late",
}


@dataclass(frozen=True)
class WeeklyEvidence:
    """Soft evidence for a character's preferred weekly-content group."""

    weights: dict[str, float]
    primary_group: str | None
    confidence: float
    valid_week_count: int


@dataclass(frozen=True)
class WeeklyProfiles:
    """Weekday factors learned from train characters up to one cutoff."""

    cutoff: date
    weekday_shift: int
    global_factors: dict[int, float]
    group_factors: dict[str, dict[int, float]]
    training_group_weight: dict[str, float]
    active_week_count: int


def activity_date(day: date, weekday_shift: int) -> date:
    return day + timedelta(days=weekday_shift)


def maple_week_start(day: date) -> date:
    """Return the Thursday starting the MapleStory week containing ``day``."""

    return day - timedelta(days=(day.weekday() - THURSDAY) % 7)


def complete_maple_weeks(
    observations: Sequence[DailyGain], weekday_shift: int = 0
) -> list[list[tuple[date, int]]]:
    """Return complete Thursday-Wednesday weeks from an observation window."""

    buckets: dict[date, list[tuple[date, int]]] = defaultdict(list)
    for observation in observations:
        mapped_day = activity_date(observation.day, weekday_shift)
        buckets[maple_week_start(mapped_day)].append((mapped_day, observation.gain))
    weeks: list[list[tuple[date, int]]] = []
    for start, values in sorted(buckets.items()):
        by_day = {day: gain for day, gain in values}
        expected = [start + timedelta(days=offset) for offset in range(7)]
        if all(day in by_day for day in expected):
            weeks.append([(day, by_day[day]) for day in expected])
    return weeks


def infer_weekly_group(
    observations: Sequence[DailyGain], weekday_shift: int = 0
) -> WeeklyEvidence:
    """Infer early/weekend/late preference from unambiguous weekly maxima."""

    votes: Counter[str] = Counter()
    for week in complete_maple_weeks(observations, weekday_shift):
        maximum = max(gain for _, gain in week)
        if maximum <= 0:
            continue
        peak_days = [day for day, gain in week if gain == maximum]
        if len(peak_days) != 1:
            continue
        votes[GROUP_BY_WEEKDAY[peak_days[0].weekday()]] += 1
    vote_count = sum(votes.values())
    if vote_count == 0:
        return WeeklyEvidence({}, None, 0.0, 0)
    weights = {group: votes[group] / vote_count for group in GROUPS if votes[group]}
    highest = max(votes.values())
    leaders = [group for group in GROUPS if votes[group] == highest]
    primary = leaders[0] if len(leaders) == 1 else None
    return WeeklyEvidence(
        weights=weights,
        primary_group=primary,
        confidence=highest / vote_count,
        valid_week_count=vote_count,
    )


def _normalized_week_profiles(
    observations: Sequence[DailyGain], weekday_shift: int
) -> list[dict[int, float]]:
    profiles: list[dict[int, float]] = []
    for week in complete_maple_weeks(observations, weekday_shift):
        total = sum(gain for _, gain in week)
        if total <= 0:
            continue
        mean = total / 7
        profiles.append({day.weekday(): gain / mean for day, gain in week})
    return profiles


def _normalize_factors(factors: dict[int, float]) -> dict[int, float]:
    mean = statistics.fmean(factors.values())
    if mean <= 0:
        return {weekday: 1.0 for weekday in WEEKDAYS}
    return {weekday: max(0.0, factors[weekday] / mean) for weekday in WEEKDAYS}


def fit_weekly_profiles(
    histories_by_character: dict[str, Sequence[DailyGain]],
    *,
    cutoff: date,
    weekday_shift: int = 0,
    shrinkage: float = 20.0,
) -> WeeklyProfiles:
    """Fit normalized weekday shapes using train histories available by cutoff."""

    if shrinkage < 0:
        raise ValueError("Shrinkage cannot be negative")
    global_sum = {weekday: 0.0 for weekday in WEEKDAYS}
    global_weight = {weekday: 0.0 for weekday in WEEKDAYS}
    group_sum = {
        group: {weekday: 0.0 for weekday in WEEKDAYS} for group in GROUPS
    }
    group_weight = {
        group: {weekday: 0.0 for weekday in WEEKDAYS} for group in GROUPS
    }
    training_group_weight = {group: 0.0 for group in GROUPS}
    active_week_count = 0

    for history in histories_by_character.values():
        evidence = infer_weekly_group(history, weekday_shift)
        week_profiles = _normalized_week_profiles(history, weekday_shift)
        active_week_count += len(week_profiles)
        for profile in week_profiles:
            for weekday, factor in profile.items():
                global_sum[weekday] += factor
                global_weight[weekday] += 1
                for group, evidence_weight in evidence.weights.items():
                    group_sum[group][weekday] += factor * evidence_weight
                    group_weight[group][weekday] += evidence_weight
        for group, evidence_weight in evidence.weights.items():
            training_group_weight[group] += evidence_weight

    global_factors = _normalize_factors(
        {
            weekday: (global_sum[weekday] + shrinkage) /
            (global_weight[weekday] + shrinkage)
            if global_weight[weekday] + shrinkage > 0
            else 1.0
            for weekday in WEEKDAYS
        }
    )
    group_factors: dict[str, dict[int, float]] = {}
    for group in GROUPS:
        group_factors[group] = _normalize_factors(
            {
                weekday: (
                    group_sum[group][weekday]
                    + shrinkage * global_factors[weekday]
                )
                / (group_weight[group][weekday] + shrinkage)
                if group_weight[group][weekday] + shrinkage > 0
                else global_factors[weekday]
                for weekday in WEEKDAYS
            }
        )
    return WeeklyProfiles(
        cutoff=cutoff,
        weekday_shift=weekday_shift,
        global_factors=global_factors,
        group_factors=group_factors,
        training_group_weight=training_group_weight,
        active_week_count=active_week_count,
    )


def fit_causal_profiles_by_origin(
    train_series: dict[str, Sequence[DailyGain]],
    *,
    lookback_days: int = 30,
    weekday_shift: int = 0,
    shrinkage: float = 20.0,
) -> dict[date, WeeklyProfiles]:
    """Fit one train-only profile at every available history cutoff."""

    if not train_series:
        raise ValueError("Training series cannot be empty")
    lengths = {len(series) for series in train_series.values()}
    if len(lengths) != 1:
        raise ValueError("Training characters must share one date range")
    series_length = next(iter(lengths))
    possible_histories = series_length - lookback_days + 1
    if possible_histories <= 0:
        raise ValueError("Training series is shorter than the requested history")
    profiles: dict[date, WeeklyProfiles] = {}
    for start in range(possible_histories):
        histories = {
            ocid: observations[start : start + lookback_days]
            for ocid, observations in train_series.items()
        }
        cutoffs = {history[-1].day for history in histories.values()}
        if len(cutoffs) != 1:
            raise ValueError("Training character dates are not aligned")
        cutoff = next(iter(cutoffs))
        profiles[cutoff] = fit_weekly_profiles(
            histories,
            cutoff=cutoff,
            weekday_shift=weekday_shift,
            shrinkage=shrinkage,
        )
    return profiles


@dataclass(frozen=True)
class WeeklyPatternPredictor:
    """Distribute the SMA total using causal train-only weekday profiles."""

    profiles_by_cutoff: dict[date, WeeklyProfiles]
    alpha: float
    use_groups: bool
    weekday_shift: int

    def __call__(
        self, history: Sequence[DailyGain], target_dates: Sequence[date]
    ) -> list[float]:
        if not 0 <= self.alpha <= 1:
            raise ValueError("Alpha must be between zero and one")
        if not history:
            raise ValueError("History cannot be empty")
        profile = self.profiles_by_cutoff[history[-1].day]
        if profile.weekday_shift != self.weekday_shift:
            raise ValueError("Predictor/profile weekday shifts differ")
        factors = profile.global_factors
        if self.use_groups:
            evidence = infer_weekly_group(history, self.weekday_shift)
            if evidence.weights:
                factors = {
                    weekday: sum(
                        weight * profile.group_factors[group][weekday]
                        for group, weight in evidence.weights.items()
                    )
                    for weekday in WEEKDAYS
                }
        scale = statistics.fmean(item.gain for item in history)
        target_factors = []
        for target_date in target_dates:
            weekday = activity_date(target_date, self.weekday_shift).weekday()
            target_factors.append(1 + self.alpha * (factors[weekday] - 1))
        normalization = statistics.fmean(target_factors) if target_factors else 1.0
        return [max(0.0, scale * factor / normalization) for factor in target_factors]


def group_persistence(
    series_by_character: dict[str, Sequence[DailyGain]],
    *,
    lookback_days: int = 30,
    horizon_days: int = 30,
    weekday_shift: int = 0,
    latest_origin_only: bool = False,
) -> dict:
    """Compare the dominant past group with the dominant future group."""

    history_distribution = Counter()
    future_distribution = Counter()
    confusion = {group: Counter() for group in GROUPS}
    comparable = 0
    matches = 0
    comparable_future_distribution = Counter()
    history_soft_routed = 0
    future_soft_routed = 0
    history_confidences: list[float] = []
    future_confidences: list[float] = []
    history_valid_weeks: list[int] = []
    future_valid_weeks: list[int] = []
    window_count = 0
    for observations in series_by_character.values():
        possible_windows = len(observations) - lookback_days - horizon_days + 1
        starts = [possible_windows - 1] if latest_origin_only else range(possible_windows)
        for start in starts:
            history = observations[start : start + lookback_days]
            future = observations[
                start + lookback_days : start + lookback_days + horizon_days
            ]
            past_evidence = infer_weekly_group(history, weekday_shift)
            future_evidence = infer_weekly_group(future, weekday_shift)
            past_group = past_evidence.primary_group
            future_group = future_evidence.primary_group
            history_valid_weeks.append(past_evidence.valid_week_count)
            future_valid_weeks.append(future_evidence.valid_week_count)
            if past_evidence.weights:
                history_soft_routed += 1
                history_confidences.append(past_evidence.confidence)
            if future_evidence.weights:
                future_soft_routed += 1
                future_confidences.append(future_evidence.confidence)
            history_distribution[past_group or "unknown"] += 1
            future_distribution[future_group or "unknown"] += 1
            if past_group is not None and future_group is not None:
                comparable += 1
                confusion[past_group][future_group] += 1
                comparable_future_distribution[future_group] += 1
                matches += past_group == future_group
            window_count += 1
    return {
        "window_count": window_count,
        "history_group_distribution": dict(history_distribution),
        "future_group_distribution": dict(future_distribution),
        "comparable_window_count": comparable,
        "same_group_count": matches,
        "same_group_rate": matches / comparable if comparable else None,
        "majority_future_group_rate": (
            max(comparable_future_distribution.values()) / comparable
            if comparable
            else None
        ),
        "soft_group_routing": {
            "history_routed_count": history_soft_routed,
            "history_routed_rate": history_soft_routed / window_count,
            "history_mean_confidence": (
                statistics.fmean(history_confidences)
                if history_confidences
                else None
            ),
            "history_mean_valid_week_count": statistics.fmean(history_valid_weeks),
            "future_routed_count": future_soft_routed,
            "future_routed_rate": future_soft_routed / window_count,
            "future_mean_confidence": (
                statistics.fmean(future_confidences)
                if future_confidences
                else None
            ),
            "future_mean_valid_week_count": statistics.fmean(future_valid_weeks),
        },
        "confusion_matrix": {
            group: {target: confusion[group][target] for target in GROUPS}
            for group in GROUPS
        },
    }
