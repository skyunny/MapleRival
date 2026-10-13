import unittest
from datetime import date, timedelta

from maplerival.forecast import SNAPSHOT_COUNT, ForecastModels, forecast_pair, load_models


class StubClassifier:
    def predict_proba(self, rows):
        return [[0.25, 0.75] for _ in rows]


class StubRegressor:
    def predict(self, rows):
        return [11.4 for _ in rows]


MODELS = ForecastModels(StubClassifier(), StubRegressor())
END = date(2026, 10, 13)


def series(exp_start, daily, level=270, days=SNAPSHOT_COUNT, end=END):
    return [
        {
            "date": (end - timedelta(days=days - 1 - index)).isoformat(),
            "level": level,
            "exp": exp_start + daily * index,
        }
        for index in range(days)
    ]


class ForecastTests(unittest.TestCase):
    def test_leader_gap_and_eta(self):
        mine = series(1_000_000, 100_000)
        rival = series(500_000, 300_000)
        result = forecast_pair("나", mine, "라이벌", rival, MODELS)
        self.assertTrue(result["available"])
        self.assertEqual((result["leader"], result["lagger"]), ("라이벌", "나"))
        self.assertEqual(result["probability"], 0.75)
        self.assertEqual(result["etaDays"], 11)
        self.assertEqual(result["expectedDate"], (END + timedelta(days=11)).isoformat())
        expected_gap = (500_000 + 300_000 * 30) - (1_000_000 + 100_000 * 30)
        self.assertEqual(result["gap"], str(expected_gap))

    def test_missing_model(self):
        mine = series(1, 10)
        self.assertEqual(forecast_pair("a", mine, "b", mine, None)["reason"], "model")

    def test_short_history(self):
        result = forecast_pair("a", series(1, 10, days=20), "b", series(2, 10), MODELS)
        self.assertEqual(result["reason"], "history")

    def test_gap_in_dates_is_rejected(self):
        broken = series(2, 10)
        del broken[10]
        result = forecast_pair("a", series(1, 10), "b", broken, MODELS)
        self.assertEqual(result["reason"], "history")

    def test_level_below_training_range(self):
        result = forecast_pair("a", series(1, 10, level=250), "b", series(2, 10), MODELS)
        self.assertEqual(result["reason"], "level")

    def test_inactive_pair(self):
        result = forecast_pair("a", series(100, 0), "b", series(200, 0), MODELS)
        self.assertEqual(result["reason"], "inactive")

    def test_tie(self):
        # 1000 + 5*30 == 940 + 7*30 on the last day
        result = forecast_pair("a", series(1000, 5), "b", series(940, 7), MODELS)
        self.assertEqual(result["reason"], "tie")

    def test_real_models_run(self):
        models = load_models()
        if models is None:
            self.skipTest("trained models not present")
        result = forecast_pair("나", series(10**11, 10**9), "라이벌", series(10**11 + 5 * 10**9, 8 * 10**8), models)
        self.assertTrue(result["available"])
        self.assertTrue(0.0 <= result["probability"] <= 1.0)
        self.assertTrue(1 <= result["etaDays"] <= 30)


if __name__ == "__main__":
    unittest.main()
