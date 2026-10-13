"""Train and evaluate a rival EXP-crossover forecaster.

For every pair of characters and every origin day, predict (a) whether the
lagging character overtakes the leader in cumulative (level-aware) EXP within
the next 30 days and (b) how many days that takes.  Characters are split
6:2:2 first and pairs are only formed inside a split, so no character appears
in more than one split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from maplerival.baselines import level_start_experience
from maplerival.crossover import (
    FEATURE_NAMES,
    HORIZON_DAYS,
    LOOKBACK_DAYS,
    build_samples,
    neighbor_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "maple_exp_balanced_8000_2026-06-26_2026-09-26.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "crossover"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "src" / "maplerival" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=800)
    return parser.parse_args()


def load_arrays(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    characters = payload["characters"]
    index = {item["ocid"]: position for position, item in enumerate(characters)}
    dates = sorted({record["date"] for record in payload["records"]})
    day_index = {day: position for position, day in enumerate(dates)}
    totals = np.zeros((len(characters), len(dates)), dtype=np.int64)
    levels = np.zeros((len(characters), len(dates)), dtype=np.int16)
    filled = np.zeros_like(levels, dtype=bool)
    for record in payload["records"]:
        row, column = index[record["ocid"]], day_index[record["date"]]
        level = int(record["level"])
        totals[row, column] = level_start_experience(level) + int(record["experience"])
        levels[row, column] = level
        filled[row, column] = True
    if not filled.all():
        raise ValueError("Dataset has missing character-days")
    weekdays = np.array([date.fromisoformat(day).weekday() for day in dates])
    return payload, characters, dates, totals, levels, weekdays


def split_characters(characters: list[dict], seed: int):
    """6:2:2 split stratified by level band."""

    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}
    bands: dict[str, list[int]] = {}
    for position, item in enumerate(characters):
        bands.setdefault(item["band"], []).append(position)
    for members in bands.values():
        members = np.array(members)
        rng.shuffle(members)
        n_train, n_val = int(len(members) * 0.6), int(len(members) * 0.2)
        splits["train"].extend(members[:n_train].tolist())
        splits["validation"].extend(members[n_train : n_train + n_val].tolist())
        splits["test"].extend(members[n_train + n_val :].tolist())
    return {name: np.array(sorted(ids)) for name, ids in splits.items()}


def to_matrix(samples):
    x = np.array([s.features for s in samples], dtype=np.float64)
    crossed = np.array([s.crossed for s in samples])
    days = np.array([s.days_to_cross if s.days_to_cross else np.nan for s in samples], dtype=float)
    pair = np.array([s.pair_index for s in samples])
    return x, crossed, days, pair


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def classification_metrics(y, probability, threshold=0.5):
    predicted = (probability >= threshold).astype(int)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": safe_auc(y, probability),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, clipped)),
        "log_loss": float(log_loss(y, clipped)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def bootstrap_auc_difference(y, score_a, score_b, pair, repetitions, rng):
    """Pair-level bootstrap of AUC(a) - AUC(b)."""

    unique_pairs = np.unique(pair)
    rows_by_pair = {p: np.nonzero(pair == p)[0] for p in unique_pairs}
    differences = []
    for _ in range(repetitions):
        chosen = rng.choice(unique_pairs, size=len(unique_pairs), replace=True)
        rows = np.concatenate([rows_by_pair[p] for p in chosen])
        if 0 < y[rows].sum() < len(rows):
            differences.append(
                roc_auc_score(y[rows], score_a[rows]) - roc_auc_score(y[rows], score_b[rows])
            )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"mean": float(np.mean(differences)), "ci95": [float(low), float(high)]}


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    _, characters, dates, totals, levels, weekdays = load_arrays(args.input)
    day_count = totals.shape[1]
    origins = range(LOOKBACK_DAYS, day_count - HORIZON_DAYS)
    anchor = origins.start
    splits = split_characters(characters, args.seed)

    data = {}
    for name, ids in splits.items():
        pairs = neighbor_pairs(totals[:, anchor], ids, args.neighbors)
        samples = build_samples(totals, levels, weekdays, pairs, origins)
        data[name] = {"samples": samples, "matrix": to_matrix(samples), "pairs": len(pairs)}
        print(f"{name}: characters={len(ids)} pairs={len(pairs)} samples={len(samples)}", flush=True)

    x_train, y_train, d_train, _ = data["train"]["matrix"]
    x_val, y_val, d_val, _ = data["validation"]["matrix"]
    x_test, y_test, d_test, pair_test = data["test"]["matrix"]

    classifier = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        od_type="Iter",
        od_wait=60,
        verbose=0,
    )
    classifier.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

    cross_train, cross_val = y_train == 1, y_val == 1
    regressor = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=args.seed,
        od_type="Iter",
        od_wait=60,
        verbose=0,
    )
    regressor.fit(
        x_train[cross_train],
        d_train[cross_train],
        eval_set=(x_val[cross_val], d_val[cross_val]),
        use_best_model=True,
    )

    probability = classifier.predict_proba(x_test)[:, 1]
    eta_index = FEATURE_NAMES.index("linear_eta_days")
    linear_eta = x_test[:, eta_index]
    linear_score = -np.minimum(linear_eta, 1e4)
    linear_probability = (linear_eta <= HORIZON_DAYS).astype(float)

    cross_test = y_test == 1
    model_eta = np.clip(regressor.predict(x_test[cross_test]), 1, HORIZON_DAYS)
    linear_eta_crossers = np.where(
        linear_eta[cross_test] <= HORIZON_DAYS, np.maximum(linear_eta[cross_test], 1), HORIZON_DAYS
    )
    actual_eta = d_test[cross_test]
    constant_eta = np.full_like(actual_eta, np.median(d_train[cross_train]))

    def eta_metrics(predicted):
        errors = predicted - actual_eta
        return {
            "mae_days": float(np.abs(errors).mean()),
            "median_abs_error_days": float(np.median(np.abs(errors))),
            "within_3_days": float((np.abs(errors) <= 3).mean()),
            "within_7_days": float((np.abs(errors) <= 7).mean()),
        }

    # Same metrics by days-to-cross bucket, to show where the model helps.
    buckets = {}
    for low, high in ((1, 7), (8, 14), (15, 30)):
        mask = (actual_eta >= low) & (actual_eta <= high)
        if mask.any():
            buckets[f"{low}-{high}"] = {
                "count": int(mask.sum()),
                "catboost_mae": float(np.abs(model_eta[mask] - actual_eta[mask]).mean()),
                "linear_mae": float(np.abs(linear_eta_crossers[mask] - actual_eta[mask]).mean()),
            }

    importance = sorted(
        zip(FEATURE_NAMES, classifier.get_feature_importance()), key=lambda item: -item[1]
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.relative_to(PROJECT_ROOT)) if args.input.is_relative_to(PROJECT_ROOT) else str(args.input),
        "design": {
            "lookback_days": LOOKBACK_DAYS,
            "horizon_days": HORIZON_DAYS,
            "origins": [origins.start, origins.stop - 1],
            "neighbors_per_character": args.neighbors,
            "pair_anchor_day": anchor,
            "seed": args.seed,
            "split": {name: len(ids) for name, ids in splits.items()},
            "pairs": {name: info["pairs"] for name, info in data.items()},
            "samples": {name: len(info["samples"]) for name, info in data.items()},
            "prevalence": {
                name: float(info["matrix"][1].mean()) for name, info in data.items()
            },
        },
        "classification_test": {
            "catboost": classification_metrics(y_test, probability),
            "linear_extrapolation": {
                **classification_metrics(y_test, linear_probability),
                "roc_auc": safe_auc(y_test, linear_score),
            },
            "base_rate": classification_metrics(
                y_test, np.full(len(y_test), data["train"]["matrix"][1].mean())
            ),
            "auc_difference_catboost_minus_linear": bootstrap_auc_difference(
                y_test, probability, linear_score, pair_test, args.bootstrap, rng
            ),
        },
        "eta_test_crossers_only": {
            "count": int(cross_test.sum()),
            "catboost": eta_metrics(model_eta),
            "linear_extrapolation": eta_metrics(linear_eta_crossers),
            "train_median_constant": eta_metrics(constant_eta),
            "by_actual_days_bucket": buckets,
        },
        "feature_importance_classifier": [
            {"feature": name, "importance": float(value)} for name, value in importance
        ],
        "best_iterations": {
            "classifier": int(classifier.get_best_iteration()),
            "regressor": int(regressor.get_best_iteration()),
        },
        "notes": [
            "Characters are split before pairing; no character is shared across splits.",
            "Samples from the same pair are correlated; bootstrap resamples whole pairs.",
            "Pairs whose members both gained no EXP in the lookback window are excluded.",
            "ETA metrics use only samples that really cross within the horizon.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    temporary = metrics_path.with_name(metrics_path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, metrics_path)
    classifier.save_model(str(args.model_dir / "crossover_classifier.cbm"))
    regressor.save_model(str(args.model_dir / "crossover_eta_regressor.cbm"))
    print(json.dumps({k: result[k] for k in ("classification_test", "eta_test_crossers_only")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
