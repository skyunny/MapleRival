"""Rival EXP crossover forecast served to the dashboard.

Uses the CatBoost models trained by ``scripts/train_crossover_model.py``.  The
models were fitted on characters of level 260-299 with 30 days of history and
a 30-day horizon, so anything outside that is reported as unavailable instead
of extrapolated.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from .baselines import level_start_experience
from .crossover import HORIZON_DAYS, LOOKBACK_DAYS, daily_gains, pair_features
from .database import recent_snapshots

MODEL_DIR = Path(__file__).parent / "models"
MIN_LEVEL = 260
SNAPSHOT_COUNT = LOOKBACK_DAYS + 1


class ForecastModels:
    def __init__(self, classifier, regressor) -> None:
        self.classifier = classifier
        self.regressor = regressor

    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        row = np.array([features], dtype=np.float64)
        probability = float(self.classifier.predict_proba(row)[0][1])
        eta = float(self.regressor.predict(row)[0])
        return probability, eta


@lru_cache(maxsize=1)
def load_models() -> ForecastModels | None:
    classifier_path = MODEL_DIR / "crossover_classifier.cbm"
    regressor_path = MODEL_DIR / "crossover_eta_regressor.cbm"
    if not (classifier_path.exists() and regressor_path.exists()):
        return None
    from catboost import CatBoostClassifier, CatBoostRegressor

    classifier = CatBoostClassifier()
    classifier.load_model(str(classifier_path))
    regressor = CatBoostRegressor()
    regressor.load_model(str(regressor_path))
    return ForecastModels(classifier, regressor)


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _aligned_window(first: list[dict], second: list[dict]):
    """Last ``SNAPSHOT_COUNT`` consecutive days present for both characters."""

    by_date = [{item["date"]: item for item in series} for series in (first, second)]
    common = set(by_date[0]) & set(by_date[1])
    if not common:
        return None
    end = date.fromisoformat(max(common))
    days = [(end - timedelta(days=offset)).isoformat() for offset in reversed(range(SNAPSHOT_COUNT))]
    if not all(day in common for day in days):
        return None
    return days, [[source[day] for day in days] for source in by_date]


def forecast_pair(
    first_name: str,
    first_series: list[dict],
    second_name: str,
    second_series: list[dict],
    models: ForecastModels | None,
) -> dict:
    """Forecast whether the trailing character overtakes the leader within 30 days."""

    if models is None:
        return _unavailable("model")
    window = _aligned_window(first_series, second_series)
    if window is None:
        return _unavailable("history")
    days, (first_rows, second_rows) = window
    levels = np.array([[row["level"] for row in rows] for rows in (first_rows, second_rows)])
    if levels.min() < MIN_LEVEL:
        return _unavailable("level")
    totals = np.array(
        [
            [level_start_experience(row["level"]) + int(row["exp"]) for row in rows]
            for rows in (first_rows, second_rows)
        ],
        dtype=np.int64,
    )
    origin = SNAPSHOT_COUNT - 1
    if totals[0, origin] == totals[1, origin]:
        return _unavailable("tie")
    names = (first_name, second_name)
    lead = 0 if totals[0, origin] > totals[1, origin] else 1
    lag = 1 - lead
    if daily_gains(totals[lead]).sum() == 0 and daily_gains(totals[lag]).sum() == 0:
        return _unavailable("inactive")

    weekday = date.fromisoformat(days[-1]).weekday()
    features = pair_features(totals[lead], totals[lag], levels[lead], levels[lag], origin, weekday)
    probability, eta = models.predict(features)
    eta_days = int(round(min(max(eta, 1.0), float(HORIZON_DAYS))))
    as_of = date.fromisoformat(days[-1])
    return {
        "available": True,
        "leader": names[lead],
        "lagger": names[lag],
        "gap": str(int(totals[lead, origin] - totals[lag, origin])),
        "probability": probability,
        "etaDays": eta_days,
        "expectedDate": (as_of + timedelta(days=eta_days)).isoformat(),
        "horizonDays": HORIZON_DAYS,
        "asOf": days[-1],
    }


def build_forecasts(database_path: Path, owner: str, rivals: list[str]) -> list[dict]:
    models = load_models()
    owner_series = recent_snapshots(database_path, owner, SNAPSHOT_COUNT + 5)
    forecasts = []
    for rival in rivals:
        rival_series = recent_snapshots(database_path, rival, SNAPSHOT_COUNT + 5)
        result = forecast_pair(owner, owner_series, rival, rival_series, models)
        forecasts.append({"rival": rival, **result})
    return forecasts
