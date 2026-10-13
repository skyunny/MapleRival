"""Evaluate 30-day moving-average baselines on character-disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from maplerival.baselines import evaluate_forecaster, records_to_daily_gains


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_CANDIDATES = ("sma_30", "weekday_sma_30")
LOOKBACK_DAYS = 30
HORIZON_DAYS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        default=PROJECT_ROOT / "data" / "baselines" / "moving_average_metrics.json",
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


def level_transition_audit(records_by_ocid: dict[str, list[dict]]) -> dict:
    multi_level_transitions = 0
    affected_characters: set[str] = set()
    maximum_level_change = 0
    for ocid, records in records_by_ocid.items():
        ordered = sorted(records, key=lambda item: item["date"])
        for previous, current in zip(ordered, ordered[1:]):
            level_change = int(current["level"]) - int(previous["level"])
            maximum_level_change = max(maximum_level_change, level_change)
            if level_change > 1:
                multi_level_transitions += 1
                affected_characters.add(ocid)
    return {
        "multi_level_transition_count": multi_level_transitions,
        "affected_character_count": len(affected_characters),
        "maximum_daily_level_change": maximum_level_change,
    }


def load_series(path: Path, expected_split: str) -> tuple[dict, dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_split = payload.get("metadata", {}).get("split")
    if actual_split != expected_split:
        raise ValueError(
            f"Expected the {expected_split} split at {path}, got {actual_split!r}"
        )
    records_by_ocid: dict[str, list[dict]] = defaultdict(list)
    for record in payload["records"]:
        records_by_ocid[record["ocid"]].append(record)
    expected_characters = int(payload["metadata"]["character_count"])
    if len(records_by_ocid) != expected_characters:
        raise ValueError(
            f"Expected {expected_characters} characters, found {len(records_by_ocid)}"
        )
    return (
        {
            ocid: records_to_daily_gains(records)
            for ocid, records in records_by_ocid.items()
        },
        payload["metadata"],
        level_transition_audit(records_by_ocid),
    )


def compact(evaluation: dict) -> dict:
    return {
        "model": evaluation["model"],
        "window_count": evaluation["window_count"],
        "prediction_count": evaluation["prediction_count"],
        **evaluation["metrics"],
    }


def main() -> None:
    args = parse_args()
    validation_series, validation_metadata, validation_audit = load_series(
        args.validation, "validation"
    )
    test_series, test_metadata, test_audit = load_series(args.test, "test")
    overlap = set(validation_series) & set(test_series)
    if overlap:
        raise ValueError(f"Validation/test character overlap: {len(overlap)}")

    validation_rolling = {
        model: evaluate_forecaster(
            validation_series,
            model,
            lookback_days=LOOKBACK_DAYS,
            horizon_days=HORIZON_DAYS,
        )
        for model in MODEL_CANDIDATES
    }
    validation_latest = {
        model: evaluate_forecaster(
            validation_series,
            model,
            lookback_days=LOOKBACK_DAYS,
            horizon_days=HORIZON_DAYS,
            latest_origin_only=True,
        )
        for model in MODEL_CANDIDATES
    }
    selected_model = min(
        MODEL_CANDIDATES,
        key=lambda model: validation_rolling[model]["metrics"]["rmse"],
    )
    test_rolling = evaluate_forecaster(
        test_series,
        selected_model,
        lookback_days=LOOKBACK_DAYS,
        horizon_days=HORIZON_DAYS,
    )
    test_latest = evaluate_forecaster(
        test_series,
        selected_model,
        lookback_days=LOOKBACK_DAYS,
        horizon_days=HORIZON_DAYS,
        latest_origin_only=True,
    )

    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": "daily_experience_gain",
            "metric_scope": (
                "pooled raw EXP over character, forecast origin, and horizon day"
            ),
            "metric_units": {
                "mae": "EXP per day",
                "rmse": "EXP per day",
                "r2_score": "dimensionless",
            },
            "features": ["weekday", "previous_daily_experience_gains"],
            "lookback_days": LOOKBACK_DAYS,
            "forecast_horizon_days": HORIZON_DAYS,
            "candidate_definitions": {
                "sma_30": "mean of the previous 30 daily gains, repeated for 30 future days",
                "weekday_sma_30": "mean of same-weekday gains within the previous 30 days",
            },
            "selection_metric": "validation rolling-origin RMSE",
            "selected_model": selected_model,
            "train_split_usage": (
                "not required; moving averages have no fitted parameters"
            ),
            "validation_source": str(args.validation.resolve()),
            "validation_sha256": sha256(args.validation),
            "validation_date_range": [
                validation_metadata["start_date"],
                validation_metadata["end_date"],
            ],
            "test_source": str(args.test.resolve()),
            "test_sha256": sha256(args.test),
            "test_date_range": [
                test_metadata["start_date"],
                test_metadata["end_date"],
            ],
            "split_unit": "character",
            "data_quality": {
                "validation": validation_audit,
                "test": test_audit,
            },
            "evaluation_notes": [
                (
                    "Rolling-origin windows overlap and must not be interpreted "
                    "as independent samples."
                ),
                (
                    "The existing character split was balanced using full-period gain "
                    "distributions; this is suitable for the requested matched splits "
                    "but is weaker than a future-period temporal test."
                ),
                (
                    "Level is used only to reconstruct cumulative EXP across level-ups, "
                    "not as a model feature."
                ),
                "A multi-level daily transition is converted to equivalent cumulative EXP. "
                "If it came from a Burning level bonus, literal earned EXP is not identifiable from daily snapshots.",
            ],
        },
        "validation": {
            "rolling_origin": validation_rolling,
            "latest_origin": validation_latest,
        },
        "selection": {
            "candidates": list(MODEL_CANDIDATES),
            "selected_model": selected_model,
            "rule": "minimum pooled RMSE on all validation rolling origins",
        },
        "test": {
            "rolling_origin": test_rolling,
            "latest_origin": test_latest,
        },
    }
    atomic_write_json(args.output, result)
    summary = {
        "output": str(args.output.resolve()),
        "selected_model": selected_model,
        "validation_rolling": {
            model: compact(evaluation)
            for model, evaluation in validation_rolling.items()
        },
        "validation_latest": {
            model: compact(evaluation)
            for model, evaluation in validation_latest.items()
        },
        "test_rolling": compact(test_rolling),
        "test_latest": compact(test_latest),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
