"""Analyze EXP differences by stable Sol Janus and Petite Luna categories."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from catboost import CatBoostRegressor
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from maplerival.experience import required_exp


METRICS = (
    "mean_daily_gain",
    "median_daily_gain",
    "active_day_rate",
    "p90_daily_gain",
    "latest_30d_gain",
)
JANUS_ORDER = ("0누스", "1누스", "2누스", "3누스")
PET_ORDER = ("0석펫", "1석펫", "3석펫")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=350)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def level_starts(maximum_level: int) -> dict[int, int]:
    starts = {200: 0}
    for level in range(201, maximum_level + 1):
        starts[level] = starts[level - 1] + required_exp(level - 1)
    return starts


def build_frames(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    characters = pd.DataFrame(payload["characters"])
    records = pd.DataFrame(payload["records"])
    starts = level_starts(int(records["level"].max()))
    records["total_exp"] = [
        starts[int(level)] + int(exp)
        for level, exp in zip(records["level"], records["experience"])
    ]
    records = records.sort_values(["ocid", "date"]).reset_index(drop=True)
    records["daily_exp_gain"] = records.groupby("ocid")["total_exp"].diff()
    daily = records.dropna(subset=["daily_exp_gain"]).copy()
    if (daily["daily_exp_gain"] < 0).any():
        raise ValueError("Negative daily EXP gain detected")
    daily["daily_exp_gain"] = daily["daily_exp_gain"].astype(np.float64)
    daily["log1p_daily_exp_gain"] = np.log1p(daily["daily_exp_gain"])
    daily = daily.merge(
        characters[["ocid", "sol_janus_category", "petite_luna_pet_category"]],
        on="ocid",
        how="left",
        validate="many_to_one",
    )

    grouped = daily.groupby("ocid")["daily_exp_gain"]
    summary = grouped.agg(
        mean_daily_gain="mean",
        median_daily_gain="median",
        active_day_rate=lambda x: float((x > 0).mean()),
        p90_daily_gain=lambda x: float(x.quantile(0.9)),
    ).reset_index()
    latest_30 = (
        daily.sort_values(["ocid", "date"])
        .groupby("ocid")
        .tail(30)
        .groupby("ocid")["daily_exp_gain"]
        .sum()
        .rename("latest_30d_gain")
        .reset_index()
    )
    summary = summary.merge(latest_30, on="ocid", validate="one_to_one")
    summary = summary.merge(
        characters[
            ["ocid", "nickname", "selection_level", "sol_janus_category", "petite_luna_pet_category"]
        ],
        on="ocid",
        validate="one_to_one",
    )
    summary["joint_category"] = (
        summary["sol_janus_category"] + "__" + summary["petite_luna_pet_category"]
    )
    return daily, summary


def bootstrap_ci(values: np.ndarray, statistic, rng: np.random.Generator, n: int) -> list[float]:
    estimates = np.empty(n)
    for index in range(n):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[index] = statistic(sample)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def group_summaries(summary: pd.DataFrame, factor: str, bootstrap: int) -> list[dict]:
    rng = np.random.default_rng(20261006)
    rows: list[dict] = []
    for group, frame in summary.groupby(factor, observed=True):
        row: dict[str, object] = {"group": str(group), "character_count": len(frame)}
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            row[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "median_bootstrap_95_ci": bootstrap_ci(values, np.median, rng, bootstrap),
            }
        rows.append(row)
    return rows


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def statistical_tests(summary: pd.DataFrame, factor: str, bootstrap: int) -> dict:
    rng = np.random.default_rng(20261007)
    result: dict[str, object] = {"kruskal_wallis": {}, "pairwise": {}}
    groups = sorted(summary[factor].unique().tolist())
    for metric in METRICS:
        arrays = [summary.loc[summary[factor] == group, metric].to_numpy(float) for group in groups]
        statistic, pvalue = kruskal(*arrays)
        result["kruskal_wallis"][metric] = {
            "statistic": float(statistic),
            "p_value": float(pvalue),
            "groups": groups,
        }
        pair_rows: list[dict] = []
        raw_pvalues: list[float] = []
        for left, right in itertools.combinations(groups, 2):
            x = summary.loc[summary[factor] == left, metric].to_numpy(float)
            y = summary.loc[summary[factor] == right, metric].to_numpy(float)
            test = mannwhitneyu(x, y, alternative="two-sided")
            delta = 2 * float(test.statistic) / (len(x) * len(y)) - 1
            differences = np.empty(bootstrap)
            for index in range(bootstrap):
                differences[index] = np.median(rng.choice(x, len(x), replace=True)) - np.median(
                    rng.choice(y, len(y), replace=True)
                )
            left_mean, right_mean = float(np.mean(x)), float(np.mean(y))
            row = {
                "left": left,
                "right": right,
                "median_difference_left_minus_right": float(np.median(x) - np.median(y)),
                "median_difference_bootstrap_95_ci": [
                    float(v) for v in np.quantile(differences, [0.025, 0.975])
                ],
                "mean_ratio_left_over_right": (
                    left_mean / right_mean if right_mean != 0 else None
                ),
                "cliffs_delta": delta,
                "p_value": float(test.pvalue),
            }
            pair_rows.append(row)
            raw_pvalues.append(float(test.pvalue))
        for row, adjusted in zip(pair_rows, holm_adjust(raw_pvalues)):
            row["holm_adjusted_p_value"] = adjusted
        result["pairwise"][metric] = pair_rows
    return result


def controlled_regression(daily: pd.DataFrame) -> dict:
    frame = daily.copy()
    frame["level_centered"] = frame["level"] - frame["level"].mean()
    base = smf.ols(
        "log1p_daily_exp_gain ~ level_centered + C(weekday)", data=frame
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["ocid"]})
    full = smf.ols(
        "log1p_daily_exp_gain ~ level_centered + C(weekday) + "
        "C(sol_janus_category) * C(petite_luna_pet_category)",
        data=frame,
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["ocid"]})
    coefficients = [
        {
            "term": term,
            "coefficient": float(value),
            "cluster_robust_p_value": float(full.pvalues[term]),
            "ci_95": [float(x) for x in full.conf_int().loc[term].tolist()],
        }
        for term, value in full.params.items()
        if "sol_janus_category" in term or "petite_luna_pet_category" in term
    ]
    return {
        "outcome": "log1p_daily_exp_gain",
        "observations": int(len(frame)),
        "character_clusters": int(frame["ocid"].nunique()),
        "base_r2": float(base.rsquared),
        "full_r2": float(full.rsquared),
        "incremental_r2": float(full.rsquared - base.rsquared),
        "aic_base": float(base.aic),
        "aic_full": float(full.aic),
        "coefficients": coefficients,
        "note": "OLS with character-clustered robust standard errors; association, not causation.",
    }


def make_model_rows(payload: dict) -> pd.DataFrame:
    chars = {c["ocid"]: c for c in payload["characters"]}
    by_ocid: dict[str, list[dict]] = defaultdict(list)
    for record in payload["records"]:
        by_ocid[record["ocid"]].append(record)
    maximum_level = max(int(r["level"]) for r in payload["records"])
    starts = level_starts(maximum_level)
    rows: list[dict] = []
    origins = (30, 37, 44, 51, 58, 62)
    for ocid, records in by_ocid.items():
        ordered = sorted(records, key=lambda item: item["date"])
        gains = np.array(
            [
                starts[int(current["level"])] + int(current["experience"])
                - starts[int(previous["level"])] - int(previous["experience"])
                for previous, current in zip(ordered, ordered[1:])
            ],
            dtype=float,
        )
        character = chars[ocid]
        for origin in origins:
            history = gains[origin - 30 : origin]
            for horizon in range(1, 31):
                target_index = origin + horizon - 1
                target_record = ordered[target_index + 1]
                row: dict[str, object] = {
                    "ocid": ocid,
                    "origin": origin,
                    "horizon_day": horizon,
                    "target_weekday": target_record["weekday"],
                    "current_level": int(ordered[origin]["level"]),
                    "history_mean_log": float(np.log1p(history).mean()),
                    "history_active_rate": float((history > 0).mean()),
                    "target_gain": float(gains[target_index]),
                    "sol_janus_category": character["sol_janus_category"],
                    "petite_luna_pet_category": character["petite_luna_pet_category"],
                }
                for lag, value in enumerate(history[::-1], start=1):
                    row[f"lag_{lag}_log"] = float(np.log1p(value))
                rows.append(row)
    return pd.DataFrame(rows)


def split_ocids(characters: pd.DataFrame, seed: int) -> tuple[set[str], set[str], set[str]]:
    frame = characters.copy()
    frame["joint"] = (
        frame["sol_janus_category"] + "__" + frame["petite_luna_pet_category"]
    )
    rng = np.random.default_rng(seed)
    split_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for _, group in frame.groupby("joint"):
        ids = group["ocid"].to_numpy().copy()
        rng.shuffle(ids)
        if len(ids) < 3:
            raise ValueError("Every joint category needs at least three characters")
        train_count = max(1, round(len(ids) * 0.6))
        validation_count = max(1, round(len(ids) * 0.2))
        if train_count + validation_count >= len(ids):
            train_count = len(ids) - 2
            validation_count = 1
        split_ids["train"].extend(ids[:train_count])
        split_ids["validation"].extend(
            ids[train_count : train_count + validation_count]
        )
        split_ids["test"].extend(ids[train_count + validation_count :])
    return (
        set(split_ids["train"]),
        set(split_ids["validation"]),
        set(split_ids["test"]),
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def catboost_comparison(payload: dict, iterations: int, bootstrap: int) -> dict:
    model_rows = make_model_rows(payload)
    characters = pd.DataFrame(payload["characters"])
    lag_features = [f"lag_{index}_log" for index in range(1, 31)]
    base_features = lag_features + [
        "history_mean_log", "history_active_rate", "current_level", "horizon_day", "target_weekday"
    ]
    feature_sets = {
        "A": base_features,
        "B": base_features + ["sol_janus_category"],
        "C": base_features + ["petite_luna_pet_category"],
        "D": base_features + ["sol_janus_category", "petite_luna_pet_category"],
    }
    categorical = {"target_weekday", "sol_janus_category", "petite_luna_pet_category"}
    runs: list[dict] = []
    for seed in (20261006, 20261007, 20261008):
        train_ids, validation_ids, test_ids = split_ocids(characters, seed)
        train = model_rows[model_rows.ocid.isin(train_ids)]
        validation = model_rows[model_rows.ocid.isin(validation_ids)]
        test = model_rows[model_rows.ocid.isin(test_ids)].copy()
        predictions: dict[str, np.ndarray] = {}
        run = {
            "seed": seed,
            "character_counts": {
                "train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)
            },
            "models": {},
        }
        for name, features in feature_sets.items():
            cat_indices = [index for index, feature in enumerate(features) if feature in categorical]
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
                train["target_gain"] / scale,
                cat_features=cat_indices,
                eval_set=(validation[features], validation["target_gain"] / scale),
                early_stopping_rounds=40,
                verbose=False,
            )
            predicted = np.maximum(0.0, model.predict(test[features]) * scale)
            predictions[name] = predicted
            raw = regression_metrics(test["target_gain"].to_numpy(), predicted)
            log_metrics = regression_metrics(
                np.log1p(test["target_gain"].to_numpy()), np.log1p(predicted)
            )
            run["models"][name] = {
                **raw,
                "log1p_mae": log_metrics["mae"],
                "log1p_rmse": log_metrics["rmse"],
                "log1p_r2": log_metrics["r2"],
                "best_iteration": int(model.get_best_iteration()),
            }
        test["abs_error_A"] = np.abs(test.target_gain.to_numpy() - predictions["A"])
        test["abs_error_D"] = np.abs(test.target_gain.to_numpy() - predictions["D"])
        per_character = test.groupby("ocid")[["abs_error_A", "abs_error_D"]].mean()
        differences = (per_character.abs_error_D - per_character.abs_error_A).to_numpy()
        rng = np.random.default_rng(seed)
        boot = np.array(
            [np.mean(rng.choice(differences, len(differences), replace=True)) for _ in range(bootstrap)]
        )
        run["D_minus_A_character_mean_mae"] = {
            "estimate": float(np.mean(differences)),
            "bootstrap_95_ci": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        }
        runs.append(run)

    aggregate: dict[str, dict] = {}
    for name in feature_sets:
        aggregate[name] = {}
        for metric in ("mae", "rmse", "r2", "log1p_mae", "log1p_rmse", "log1p_r2"):
            values = [run["models"][name][metric] for run in runs]
            aggregate[name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "values": values,
            }
    return {
        "design": {
            "split": "character-level stratified 60:20:20",
            "rolling_origins": [30, 37, 44, 51, 58, 62],
            "forecast_horizon_days": 30,
            "seeds": [20261006, 20261007, 20261008],
            "common_controls": ["current_level", "target_weekday", "horizon_day"],
        },
        "aggregate": aggregate,
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    daily, summary = build_frames(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "daily_exp_gain.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "character_summary.csv", index=False, encoding="utf-8-sig")

    factors = (
        "sol_janus_category",
        "petite_luna_pet_category",
        "joint_category",
    )
    exploratory = {
        "definitions": {
            "daily_exp_gain": "level-aware cumulative EXP difference between consecutive dates",
            "latest_30d_gain": "sum of the final 30 daily gains per character",
            "unit_of_inference": "character",
        },
        "character_count": int(summary.ocid.nunique()),
        "daily_observation_count": int(len(daily)),
        "group_summaries": {
            factor: group_summaries(summary, factor, args.bootstrap) for factor in factors
        },
        "tests": {
            factor: statistical_tests(summary, factor, args.bootstrap) for factor in factors
        },
        "controlled_regression": controlled_regression(daily),
    }
    write_json(args.output_dir / "statistical_analysis.json", exploratory)
    models = catboost_comparison(payload, args.iterations, args.bootstrap)
    write_json(args.output_dir / "catboost_comparison.json", models)
    print(
        json.dumps(
            {
                "characters": exploratory["character_count"],
                "daily_observations": exploratory["daily_observation_count"],
                "controlled_incremental_r2": exploratory["controlled_regression"]["incremental_r2"],
                "catboost": models["aggregate"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
