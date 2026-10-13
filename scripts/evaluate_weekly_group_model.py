"""Train and evaluate a causal weekly-content group EXP forecasting model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from maplerival.baselines import (
    evaluate_forecaster,
    evaluate_predictor,
    records_to_daily_gains,
    regression_metrics,
    simple_moving_average,
)
from maplerival.weekly_patterns import (
    WEEKDAYS,
    WeeklyPatternPredictor,
    fit_causal_profiles_by_origin,
    group_persistence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
LOOKBACK_DAYS = 30
HORIZON_DAYS = 30
SHRINKAGE = 20.0
ALPHAS = (0.25, 0.5, 0.75, 1.0)
WEEKDAY_SHIFTS = (0, -1)
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "maple_exp_train.json",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "maple_exp_validation.json",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "maple_exp_test.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "baselines" / "weekly_group_metrics.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_series(path: Path, expected_split: str) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_split = payload.get("metadata", {}).get("split")
    if actual_split != expected_split:
        raise ValueError(
            f"Expected the {expected_split} split at {path}, got {actual_split!r}"
        )
    records_by_ocid: dict[str, list[dict]] = defaultdict(list)
    for record in payload["records"]:
        records_by_ocid[record["ocid"]].append(record)
    series = {
        ocid: records_to_daily_gains(records)
        for ocid, records in records_by_ocid.items()
    }
    expected_characters = int(payload["metadata"]["character_count"])
    if len(series) != expected_characters:
        raise ValueError(
            f"Expected {expected_characters} characters, found {len(series)}"
        )
    return series, payload["metadata"]


def pooled_score(
    series: dict,
    predictor,
    *,
    origin_policy: str = "all",
) -> dict:
    actual: list[int] = []
    predicted: list[float] = []
    window_count = 0
    for observations in series.values():
        possible_windows = len(observations) - LOOKBACK_DAYS - HORIZON_DAYS + 1
        if origin_policy == "all":
            starts = range(possible_windows)
        elif origin_policy == "earliest":
            starts = [0]
        elif origin_policy == "latest":
            starts = [possible_windows - 1]
        else:
            raise ValueError(f"Unknown origin policy: {origin_policy}")
        for start in starts:
            history = observations[start : start + LOOKBACK_DAYS]
            target = observations[
                start + LOOKBACK_DAYS : start + LOOKBACK_DAYS + HORIZON_DAYS
            ]
            actual.extend(item.gain for item in target)
            predicted.extend(predictor(history, [item.day for item in target]))
            window_count += 1
    return {
        "window_count": window_count,
        "prediction_count": len(actual),
        **regression_metrics(actual, predicted),
    }


def choose_candidate(
    validation_series: dict,
    profiles_by_shift: dict[int, dict],
    *,
    use_groups: bool,
    origin_policy: str = "all",
) -> tuple[WeeklyPatternPredictor, list[dict]]:
    candidates: list[tuple[WeeklyPatternPredictor, dict]] = []
    for weekday_shift in WEEKDAY_SHIFTS:
        for alpha in ALPHAS:
            predictor = WeeklyPatternPredictor(
                profiles_by_cutoff=profiles_by_shift[weekday_shift],
                alpha=alpha,
                use_groups=use_groups,
                weekday_shift=weekday_shift,
            )
            score = pooled_score(
                validation_series,
                predictor,
                origin_policy=origin_policy,
            )
            candidates.append(
                (
                    predictor,
                    {
                        "weekday_shift": weekday_shift,
                        "alpha": alpha,
                        "use_groups": use_groups,
                        **score,
                    },
                )
            )
    predictor, _ = min(candidates, key=lambda item: item[1]["rmse"])
    return predictor, [score for _, score in candidates]


def model_name(predictor: WeeklyPatternPredictor) -> str:
    return "weekly_group_profile_30" if predictor.use_groups else "global_weekday_profile_30"


def compact(evaluation: dict) -> dict:
    return {
        "model": evaluation["model"],
        "window_count": evaluation["window_count"],
        "prediction_count": evaluation["prediction_count"],
        **evaluation["metrics"],
    }


def comparison(candidate: dict, baseline: dict) -> dict:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    return {
        "mae_improvement_percent": (
            (baseline_metrics["mae"] - candidate_metrics["mae"])
            / baseline_metrics["mae"]
            * 100
        ),
        "rmse_improvement_percent": (
            (baseline_metrics["rmse"] - candidate_metrics["rmse"])
            / baseline_metrics["rmse"]
            * 100
        ),
        "r2_score_change": (
            candidate_metrics["r2_score"] - baseline_metrics["r2_score"]
        ),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_character_analysis(
    series: dict,
    baseline_predictor,
    candidate_predictor,
    *,
    seed: int = 20261001,
    bootstrap_samples: int = 5000,
    latest_origin_only: bool = False,
) -> dict:
    errors: dict[str, tuple[float, float, int]] = {}
    for ocid, observations in series.items():
        baseline_sse = 0.0
        candidate_sse = 0.0
        count = 0
        possible_windows = len(observations) - LOOKBACK_DAYS - HORIZON_DAYS + 1
        starts = [possible_windows - 1] if latest_origin_only else range(possible_windows)
        for start in starts:
            history = observations[start : start + LOOKBACK_DAYS]
            target = observations[
                start + LOOKBACK_DAYS : start + LOOKBACK_DAYS + HORIZON_DAYS
            ]
            target_dates = [item.day for item in target]
            actual = [item.gain for item in target]
            baseline = baseline_predictor(history, target_dates)
            candidate = candidate_predictor(history, target_dates)
            baseline_sse += sum(
                (estimate - observed) ** 2
                for observed, estimate in zip(actual, baseline)
            )
            candidate_sse += sum(
                (estimate - observed) ** 2
                for observed, estimate in zip(actual, candidate)
            )
            count += len(actual)
        errors[ocid] = (baseline_sse, candidate_sse, count)

    improved_characters = sum(
        candidate_sse < baseline_sse
        for baseline_sse, candidate_sse, _ in errors.values()
    )
    identifiers = sorted(errors)
    rng = random.Random(seed)
    improvements: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [rng.choice(identifiers) for _ in identifiers]
        baseline_sse = sum(errors[ocid][0] for ocid in sample)
        candidate_sse = sum(errors[ocid][1] for ocid in sample)
        count = sum(errors[ocid][2] for ocid in sample)
        baseline_rmse = math.sqrt(baseline_sse / count)
        candidate_rmse = math.sqrt(candidate_sse / count)
        improvements.append((baseline_rmse - candidate_rmse) / baseline_rmse * 100)
    baseline_sse = sum(item[0] for item in errors.values())
    candidate_sse = sum(item[1] for item in errors.values())
    count = sum(item[2] for item in errors.values())
    baseline_rmse = math.sqrt(baseline_sse / count)
    candidate_rmse = math.sqrt(candidate_sse / count)
    return {
        "character_count": len(identifiers),
        "improved_character_count": improved_characters,
        "improved_character_rate": improved_characters / len(identifiers),
        "rmse_improvement_percent": (
            (baseline_rmse - candidate_rmse) / baseline_rmse * 100
        ),
        "character_cluster_bootstrap_samples": bootstrap_samples,
        "rmse_improvement_95_percent_ci": [
            percentile(improvements, 0.025),
            percentile(improvements, 0.975),
        ],
    }


def profile_snapshot(predictor: WeeklyPatternPredictor) -> dict:
    cutoff = max(predictor.profiles_by_cutoff)
    profile = predictor.profiles_by_cutoff[cutoff]

    def named(factors: dict[int, float]) -> dict[str, float]:
        return {WEEKDAY_NAMES[weekday]: factors[weekday] for weekday in WEEKDAYS}

    return {
        "cutoff": cutoff.isoformat(),
        "weekday_shift": predictor.weekday_shift,
        "alpha": predictor.alpha,
        "active_training_week_count": profile.active_week_count,
        "training_group_weight": profile.training_group_weight,
        "global_factors": named(profile.global_factors),
        "group_factors": {
            group: named(factors) for group, factors in profile.group_factors.items()
        },
    }


def main() -> None:
    args = parse_args()
    train_series, train_metadata = load_series(args.train, "train")
    validation_series, validation_metadata = load_series(args.validation, "validation")
    test_series, test_metadata = load_series(args.test, "test")
    if (set(train_series) & set(validation_series)) or (
        set(train_series) & set(test_series)
    ) or (set(validation_series) & set(test_series)):
        raise ValueError("Train, validation, and test characters must be disjoint")

    profiles_by_shift = {
        shift: fit_causal_profiles_by_origin(
            train_series,
            lookback_days=LOOKBACK_DAYS,
            weekday_shift=shift,
            shrinkage=SHRINKAGE,
        )
        for shift in WEEKDAY_SHIFTS
    }
    global_predictor, global_candidates = choose_candidate(
        validation_series, profiles_by_shift, use_groups=False
    )
    group_predictor, group_candidates = choose_candidate(
        validation_series, profiles_by_shift, use_groups=True
    )
    temporal_global_predictor, temporal_global_candidates = choose_candidate(
        validation_series,
        profiles_by_shift,
        use_groups=False,
        origin_policy="earliest",
    )
    temporal_group_predictor, temporal_group_candidates = choose_candidate(
        validation_series,
        profiles_by_shift,
        use_groups=True,
        origin_policy="earliest",
    )

    validation = {
        "sma_30": evaluate_forecaster(validation_series, "sma_30"),
        "personal_weekday_sma_30": evaluate_forecaster(
            validation_series, "weekday_sma_30"
        ),
        "global_weekday_profile_30": evaluate_predictor(
            validation_series,
            global_predictor,
            model_name(global_predictor),
        ),
        "weekly_group_profile_30": evaluate_predictor(
            validation_series,
            group_predictor,
            model_name(group_predictor),
        ),
    }
    test = {
        "sma_30": evaluate_forecaster(test_series, "sma_30"),
        "personal_weekday_sma_30": evaluate_forecaster(test_series, "weekday_sma_30"),
        "global_weekday_profile_30": evaluate_predictor(
            test_series,
            global_predictor,
            model_name(global_predictor),
        ),
        "weekly_group_profile_30": evaluate_predictor(
            test_series,
            group_predictor,
            model_name(group_predictor),
        ),
    }
    validation_latest = {
        "sma_30": evaluate_forecaster(
            validation_series, "sma_30", latest_origin_only=True
        ),
        "global_weekday_profile_30": evaluate_predictor(
            validation_series,
            global_predictor,
            model_name(global_predictor),
            latest_origin_only=True,
        ),
        "weekly_group_profile_30": evaluate_predictor(
            validation_series,
            group_predictor,
            model_name(group_predictor),
            latest_origin_only=True,
        ),
    }
    test_latest = {
        "sma_30": evaluate_forecaster(test_series, "sma_30", latest_origin_only=True),
        "global_weekday_profile_30": evaluate_predictor(
            test_series,
            global_predictor,
            model_name(global_predictor),
            latest_origin_only=True,
        ),
        "weekly_group_profile_30": evaluate_predictor(
            test_series,
            group_predictor,
            model_name(group_predictor),
            latest_origin_only=True,
        ),
    }

    validation_baseline = validation["sma_30"]
    test_baseline = test["sma_30"]
    group_beats_baseline = (
        test["weekly_group_profile_30"]["metrics"]["rmse"]
        < test_baseline["metrics"]["rmse"]
    )
    group_beats_global = (
        test["weekly_group_profile_30"]["metrics"]["rmse"]
        < test["global_weekday_profile_30"]["metrics"]["rmse"]
    )
    temporal_validation = {
        "sma_30": pooled_score(
            validation_series,
            simple_moving_average,
            origin_policy="earliest",
        ),
        "global_weekday_profile_30": pooled_score(
            validation_series,
            temporal_global_predictor,
            origin_policy="earliest",
        ),
        "weekly_group_profile_30": pooled_score(
            validation_series,
            temporal_group_predictor,
            origin_policy="earliest",
        ),
    }
    temporal_test = {
        "sma_30": pooled_score(
            test_series,
            simple_moving_average,
            origin_policy="latest",
        ),
        "global_weekday_profile_30": pooled_score(
            test_series,
            temporal_global_predictor,
            origin_policy="latest",
        ),
        "weekly_group_profile_30": pooled_score(
            test_series,
            temporal_group_predictor,
            origin_policy="latest",
        ),
    }
    sample_validation_series = next(iter(validation_series.values()))
    sample_test_series = next(iter(test_series.values()))
    first_validation_target = sample_validation_series[
        LOOKBACK_DAYS : LOOKBACK_DAYS + HORIZON_DAYS
    ]
    last_test_target = sample_test_series[-HORIZON_DAYS:]
    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": "daily_experience_gain",
            "features": ["previous_30_daily_gains", "weekday"],
            "lookback_days": LOOKBACK_DAYS,
            "forecast_horizon_days": HORIZON_DAYS,
            "weekly_cycle": "Thursday through Wednesday",
            "groups": {
                "early": ["Thursday", "Friday"],
                "weekend": ["Saturday", "Sunday"],
                "late": ["Monday", "Tuesday", "Wednesday"],
            },
            "profile_training": (
                "walk-forward and causal: only train-character observations at or before "
                "each forecast cutoff are used"
            ),
            "learned_profile_total_forecast_constraint": (
                "the global and group profile models preserve the SMA-30 total over "
                "the 30-day horizon; only their daily allocation changes"
            ),
            "selection_metric": "validation rolling-origin RMSE",
            "shrinkage": SHRINKAGE,
            "alpha_grid": list(ALPHAS),
            "weekday_shift_grid": list(WEEKDAY_SHIFTS),
            "weekday_shift_meaning": (
                "0 uses the gain record date; -1 maps a gain to the previous activity date"
            ),
            "sources": {
                "train": {"path": str(args.train.resolve()), "sha256": sha256(args.train)},
                "validation": {
                    "path": str(args.validation.resolve()),
                    "sha256": sha256(args.validation),
                },
                "test": {"path": str(args.test.resolve()), "sha256": sha256(args.test)},
            },
            "date_range": [train_metadata["start_date"], train_metadata["end_date"]],
            "split_unit": "character",
            "evaluation_note": (
                "This is a retrospective character-generalization benchmark. The existing "
                "split was balanced with full-period gain distributions, and rolling windows overlap."
            ),
        },
        "selected_hyperparameters": {
            "global_weekday_profile_30": {
                "alpha": global_predictor.alpha,
                "weekday_shift": global_predictor.weekday_shift,
            },
            "weekly_group_profile_30": {
                "alpha": group_predictor.alpha,
                "weekday_shift": group_predictor.weekday_shift,
                "group_mode": "soft weekly-maximum vote",
            },
        },
        "validation_search": {
            "global_weekday_profile_30": global_candidates,
            "weekly_group_profile_30": group_candidates,
        },
        "temporal_holdout": {
            "protocol": (
                "hyperparameters selected on the earliest validation 30-day target, "
                "then evaluated on the latest non-overlapping test 30-day target"
            ),
            "validation_target_date_range": [
                first_validation_target[0].day.isoformat(),
                first_validation_target[-1].day.isoformat(),
            ],
            "test_target_date_range": [
                last_test_target[0].day.isoformat(),
                last_test_target[-1].day.isoformat(),
            ],
            "selected_hyperparameters": {
                "global_weekday_profile_30": {
                    "alpha": temporal_global_predictor.alpha,
                    "weekday_shift": temporal_global_predictor.weekday_shift,
                },
                "weekly_group_profile_30": {
                    "alpha": temporal_group_predictor.alpha,
                    "weekday_shift": temporal_group_predictor.weekday_shift,
                    "group_mode": "soft weekly-maximum vote",
                },
            },
            "validation_search": {
                "global_weekday_profile_30": temporal_global_candidates,
                "weekly_group_profile_30": temporal_group_candidates,
            },
            "validation": temporal_validation,
            "test": temporal_test,
            "test_comparison_to_sma_30": {
                name: comparison(
                    {"metrics": metrics},
                    {"metrics": temporal_test["sma_30"]},
                )
                for name, metrics in temporal_test.items()
                if name != "sma_30"
            },
            "hypothesis_result": {
                "group_model_beats_sma_on_test_rmse": (
                    temporal_test["weekly_group_profile_30"]["rmse"]
                    < temporal_test["sma_30"]["rmse"]
                ),
                "group_model_beats_global_weekday_model_on_test_rmse": (
                    temporal_test["weekly_group_profile_30"]["rmse"]
                    < temporal_test["global_weekday_profile_30"]["rmse"]
                ),
            },
            "test_paired_character_analysis": {
                "global_weekday_vs_sma": paired_character_analysis(
                    test_series,
                    simple_moving_average,
                    temporal_global_predictor,
                    latest_origin_only=True,
                ),
                "weekly_group_vs_sma": paired_character_analysis(
                    test_series,
                    simple_moving_average,
                    temporal_group_predictor,
                    latest_origin_only=True,
                ),
                "weekly_group_vs_global_weekday": paired_character_analysis(
                    test_series,
                    temporal_global_predictor,
                    temporal_group_predictor,
                    latest_origin_only=True,
                ),
            },
        },
        "validation": {name: compact(value) for name, value in validation.items()},
        "validation_latest_origin": {
            name: compact(value) for name, value in validation_latest.items()
        },
        "test": {name: compact(value) for name, value in test.items()},
        "test_latest_origin": {
            name: compact(value) for name, value in test_latest.items()
        },
        "comparison_to_sma_30": {
            "validation": {
                name: comparison(value, validation_baseline)
                for name, value in validation.items()
                if name != "sma_30"
            },
            "test": {
                name: comparison(value, test_baseline)
                for name, value in test.items()
                if name != "sma_30"
            },
        },
        "hypothesis_result": {
            "group_model_beats_sma_on_test_rmse": group_beats_baseline,
            "group_model_beats_global_weekday_model_on_test_rmse": group_beats_global,
            "interpretation": (
                "personal weekly-content grouping is supported"
                if group_beats_baseline and group_beats_global
                else "weekday allocation may help, but the proposed personal grouping is not supported"
                if group_beats_baseline
                else "the proposed grouping is not supported by this benchmark"
            ),
        },
        "group_diagnostics": {
            "validation": group_persistence(
                validation_series, weekday_shift=group_predictor.weekday_shift
            ),
            "test": group_persistence(
                test_series, weekday_shift=group_predictor.weekday_shift
            ),
            "validation_latest_origin": group_persistence(
                validation_series,
                weekday_shift=group_predictor.weekday_shift,
                latest_origin_only=True,
            ),
            "test_latest_origin": group_persistence(
                test_series,
                weekday_shift=group_predictor.weekday_shift,
                latest_origin_only=True,
            ),
        },
        "paired_character_analysis": {
            "validation": {
                "global_weekday_vs_sma": paired_character_analysis(
                    validation_series,
                    simple_moving_average,
                    global_predictor,
                ),
                "weekly_group_vs_sma": paired_character_analysis(
                    validation_series,
                    simple_moving_average,
                    group_predictor,
                ),
                "weekly_group_vs_global_weekday": paired_character_analysis(
                    validation_series,
                    global_predictor,
                    group_predictor,
                ),
            },
            "test": {
                "global_weekday_vs_sma": paired_character_analysis(
                    test_series,
                    simple_moving_average,
                    global_predictor,
                ),
                "weekly_group_vs_sma": paired_character_analysis(
                    test_series,
                    simple_moving_average,
                    group_predictor,
                ),
                "weekly_group_vs_global_weekday": paired_character_analysis(
                    test_series,
                    global_predictor,
                    group_predictor,
                ),
            },
        },
        "paired_character_analysis_latest_origin": {
            "validation": {
                "global_weekday_vs_sma": paired_character_analysis(
                    validation_series,
                    simple_moving_average,
                    global_predictor,
                    latest_origin_only=True,
                ),
                "weekly_group_vs_sma": paired_character_analysis(
                    validation_series,
                    simple_moving_average,
                    group_predictor,
                    latest_origin_only=True,
                ),
                "weekly_group_vs_global_weekday": paired_character_analysis(
                    validation_series,
                    global_predictor,
                    group_predictor,
                    latest_origin_only=True,
                ),
            },
            "test": {
                "global_weekday_vs_sma": paired_character_analysis(
                    test_series,
                    simple_moving_average,
                    global_predictor,
                    latest_origin_only=True,
                ),
                "weekly_group_vs_sma": paired_character_analysis(
                    test_series,
                    simple_moving_average,
                    group_predictor,
                    latest_origin_only=True,
                ),
                "weekly_group_vs_global_weekday": paired_character_analysis(
                    test_series,
                    global_predictor,
                    group_predictor,
                    latest_origin_only=True,
                ),
            },
        },
        "learned_profile_snapshot": {
            "global_weekday_profile_30": profile_snapshot(global_predictor),
            "weekly_group_profile_30": profile_snapshot(group_predictor),
        },
        "full_selected_model_metrics": {
            "validation": {
                "global_weekday_profile_30": validation["global_weekday_profile_30"],
                "weekly_group_profile_30": validation["weekly_group_profile_30"],
            },
            "test": {
                "global_weekday_profile_30": test["global_weekday_profile_30"],
                "weekly_group_profile_30": test["weekly_group_profile_30"],
            },
        },
    }
    atomic_write_json(args.output, result)
    summary = {
        "output": str(args.output.resolve()),
        "selected_hyperparameters": result["selected_hyperparameters"],
        "validation": result["validation"],
        "test": result["test"],
        "comparison_to_sma_30": result["comparison_to_sma_30"],
        "hypothesis_result": result["hypothesis_result"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
