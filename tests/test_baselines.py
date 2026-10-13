import math
import unittest
from datetime import date, timedelta

from maplerival.baselines import (
    DailyGain,
    evaluate_forecaster,
    records_to_daily_gains,
    regression_metrics,
    simple_moving_average,
    weekday_moving_average,
)
from maplerival.experience import required_exp


class MovingAverageBaselineTest(unittest.TestCase):
    def test_daily_gain_accounts_for_level_up(self):
        records = [
            {
                "date": "2026-06-26",
                "weekday": "Friday",
                "level": 260,
                "experience": required_exp(260) - 10,
            },
            {
                "date": "2026-06-27",
                "weekday": "Saturday",
                "level": 261,
                "experience": 20,
            },
        ]
        self.assertEqual(records_to_daily_gains(records)[0].gain, 30)

    def test_multi_level_jump_uses_equivalent_cumulative_exp(self):
        records = [
            {
                "date": "2026-06-26",
                "weekday": "Friday",
                "level": 260,
                "experience": required_exp(260) - 10,
            },
            {
                "date": "2026-06-27",
                "weekday": "Saturday",
                "level": 262,
                "experience": 20,
            },
        ]
        self.assertEqual(
            records_to_daily_gains(records)[0].gain,
            required_exp(261) + 30,
        )

    def test_simple_moving_average_repeats_history_mean(self):
        history = [
            DailyGain(date(2026, 1, 1), 10),
            DailyGain(date(2026, 1, 2), 20),
        ]
        self.assertEqual(
            simple_moving_average(history, [date(2026, 1, 3), date(2026, 1, 4)]),
            [15.0, 15.0],
        )

    def test_weekday_moving_average_uses_matching_weekday_only(self):
        monday = date(2026, 1, 5)
        history = [
            DailyGain(monday + timedelta(days=offset), 100 if offset % 7 == 0 else 0)
            for offset in range(14)
        ]
        predictions = weekday_moving_average(
            history,
            [monday + timedelta(days=14), monday + timedelta(days=15)],
        )
        self.assertEqual(predictions, [100.0, 0.0])

    def test_regression_metrics_match_hand_calculation(self):
        metrics = regression_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(1 / 3))
        self.assertAlmostEqual(metrics["r2_score"], 0.5)

    def test_constant_target_r2_is_finite(self):
        self.assertEqual(regression_metrics([5, 5], [5, 5])["r2_score"], 1.0)
        self.assertEqual(regression_metrics([5, 5], [4, 4])["r2_score"], 0.0)

    def test_invalid_snapshot_sequence_is_rejected(self):
        duplicate = [
            {"date": "2026-06-26", "level": 260, "experience": 10},
            {"date": "2026-06-26", "level": 260, "experience": 20},
        ]
        decreasing = [
            {"date": "2026-06-26", "level": 260, "experience": 20},
            {"date": "2026-06-27", "level": 260, "experience": 10},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            records_to_daily_gains(duplicate)
        with self.assertRaisesRegex(ValueError, "Negative"):
            records_to_daily_gains(decreasing)

    def test_rolling_and_latest_origin_window_counts(self):
        start = date(2026, 1, 1)
        series = {
            "character": [
                DailyGain(start + timedelta(days=index), index) for index in range(65)
            ]
        }
        rolling = evaluate_forecaster(
            series, "sma_30", lookback_days=30, horizon_days=30
        )
        latest = evaluate_forecaster(
            series,
            "sma_30",
            lookback_days=30,
            horizon_days=30,
            latest_origin_only=True,
        )
        self.assertEqual(rolling["window_count"], 6)
        self.assertEqual(rolling["prediction_count"], 180)
        self.assertEqual(latest["window_count"], 1)
        self.assertEqual(latest["prediction_count"], 30)


if __name__ == "__main__":
    unittest.main()
