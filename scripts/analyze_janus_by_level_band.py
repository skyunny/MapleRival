"""Compare CatBoost with/without Sol Janus inside five-level bands."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analyze_category_effects import make_model_rows


SEEDS = (20261006, 20261007, 20261008, 20261009, 20261010)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def stratified_ids(frame: pd.DataFrame, seed: int) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": [], "test": []}
    for _, group in frame.groupby("sol_janus_category"):
        ids = group.ocid.to_numpy().copy()
        rng.shuffle(ids)
        if len(ids) < 3:
            result["train"].extend(ids)
            continue
        train_count = max(1, round(len(ids) * 0.6))
        validation_count = max(1, round(len(ids) * 0.2))
        if train_count + validation_count >= len(ids):
            train_count = len(ids) - 2
            validation_count = 1
        result["train"].extend(ids[:train_count])
        result["validation"].extend(ids[train_count : train_count + validation_count])
        result["test"].extend(ids[train_count + validation_count :])
    return set(result["train"]), set(result["validation"]), set(result["test"])


def fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, int]:
    categorical = [index for index, name in enumerate(features) if name in {"target_weekday", "sol_janus_category"}]
    model = CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    scale = 1_000_000_000_000.0
    model.fit(
        train[features],
        train.target_gain / scale,
        cat_features=categorical,
        eval_set=(validation[features], validation.target_gain / scale),
        early_stopping_rounds=40,
        verbose=False,
    )
    prediction = np.maximum(0.0, model.predict(test[features]) * scale)
    return prediction, int(model.get_best_iteration())


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    characters = pd.DataFrame(payload["characters"])
    characters["level_band"] = characters.selection_level.map(
        lambda level: f"{int(level) // 5 * 5}-{int(level) // 5 * 5 + 4}"
    )
    model_rows = make_model_rows(payload).merge(
        characters[["ocid", "selection_level", "level_band"]],
        on="ocid",
        validate="many_to_one",
    )
    lag_features = [f"lag_{index}_log" for index in range(1, 31)]
    common = lag_features + [
        "history_mean_log",
        "history_active_rate",
        "current_level",
        "horizon_day",
        "target_weekday",
    ]
    output: dict[str, object] = {
        "design": {
            "level_definition": "selection_level at 2026-06-26",
            "models": {"A": "history+weekday+level", "B": "A+sol_janus_category"},
            "petite_luna_used": False,
            "character_split": "within-band, Janus-stratified, approximately 60:20:20",
            "rolling_origins": [30, 37, 44, 51, 58, 62],
            "forecast_horizon_days": 30,
            "seeds": list(SEEDS),
        },
        "bands": {},
    }

    for start in range(260, 300, 5):
        band = f"{start}-{start + 4}"
        chars = characters[characters.level_band == band].copy()
        counts = chars.sol_janus_category.value_counts().sort_index().to_dict()
        band_result: dict[str, object] = {
            "character_count": int(len(chars)),
            "janus_counts": {str(key): int(value) for key, value in counts.items()},
            "runs": [],
        }
        if chars.sol_janus_category.nunique() < 2 or sum(value >= 3 for value in counts.values()) < 2:
            band_result["status"] = "not_estimable"
            band_result["reason"] = "fewer than two Sol Janus categories have at least three characters"
            output["bands"][band] = band_result
            continue

        rows = model_rows[model_rows.level_band == band]
        for seed in SEEDS:
            train_ids, validation_ids, test_ids = stratified_ids(chars, seed)
            train = rows[rows.ocid.isin(train_ids)]
            validation = rows[rows.ocid.isin(validation_ids)]
            test = rows[rows.ocid.isin(test_ids)].copy()
            prediction_a, iteration_a = fit_predict(
                train, validation, test, common, seed, args.iterations
            )
            prediction_b, iteration_b = fit_predict(
                train,
                validation,
                test,
                common + ["sol_janus_category"],
                seed,
                args.iterations,
            )
            actual = test.target_gain.to_numpy()
            metric_a, metric_b = metrics(actual, prediction_a), metrics(actual, prediction_b)
            test["error_a"] = np.abs(actual - prediction_a)
            test["error_b"] = np.abs(actual - prediction_b)
            paired = test.groupby("ocid")[["error_a", "error_b"]].mean()
            difference = (paired.error_b - paired.error_a).to_numpy()
            rng = np.random.default_rng(seed)
            bootstrap = np.array(
                [np.mean(rng.choice(difference, len(difference), replace=True)) for _ in range(args.bootstrap)]
            )
            band_result["runs"].append(
                {
                    "seed": seed,
                    "character_counts": {
                        "train": len(train_ids),
                        "validation": len(validation_ids),
                        "test": len(test_ids),
                    },
                    "test_janus_counts": {
                        str(key): int(value)
                        for key, value in chars[chars.ocid.isin(test_ids)]
                        .sol_janus_category.value_counts()
                        .sort_index()
                        .to_dict()
                        .items()
                    },
                    "A": {**metric_a, "best_iteration": iteration_a},
                    "B": {**metric_b, "best_iteration": iteration_b},
                    "B_minus_A": {
                        "mae": metric_b["mae"] - metric_a["mae"],
                        "rmse": metric_b["rmse"] - metric_a["rmse"],
                        "r2": metric_b["r2"] - metric_a["r2"],
                        "character_mean_mae": float(np.mean(difference)),
                        "character_mean_mae_bootstrap_95_ci": [
                            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
                        ],
                    },
                }
            )

        runs = band_result["runs"]
        band_result["status"] = "estimated"
        band_result["aggregate"] = {}
        for model_name in ("A", "B"):
            band_result["aggregate"][model_name] = {
                metric: {
                    "mean": float(np.mean([run[model_name][metric] for run in runs])),
                    "std": float(np.std([run[model_name][metric] for run in runs], ddof=1)),
                }
                for metric in ("mae", "rmse", "r2")
            }
        band_result["improvement"] = {
            "mae_percent": float(
                100
                * (
                    band_result["aggregate"]["A"]["mae"]["mean"]
                    - band_result["aggregate"]["B"]["mae"]["mean"]
                )
                / band_result["aggregate"]["A"]["mae"]["mean"]
            ),
            "rmse_percent": float(
                100
                * (
                    band_result["aggregate"]["A"]["rmse"]["mean"]
                    - band_result["aggregate"]["B"]["rmse"]["mean"]
                )
                / band_result["aggregate"]["A"]["rmse"]["mean"]
            ),
            "r2_points": float(
                band_result["aggregate"]["B"]["r2"]["mean"]
                - band_result["aggregate"]["A"]["r2"]["mean"]
            ),
            "mae_improved_run_count": int(
                sum(run["B"]["mae"] < run["A"]["mae"] for run in runs)
            ),
            "rmse_improved_run_count": int(
                sum(run["B"]["rmse"] < run["A"]["rmse"] for run in runs)
            ),
        }
        output["bands"][band] = band_result

    atomic_write(args.output, output)
    compact = {
        band: {
            "n": result["character_count"],
            "janus": result["janus_counts"],
            "status": result["status"],
            "improvement": result.get("improvement"),
        }
        for band, result in output["bands"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
