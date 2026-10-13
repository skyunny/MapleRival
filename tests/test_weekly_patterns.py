import statistics
import unittest
from datetime import date, timedelta

from maplerival.baselines import DailyGain
from maplerival.weekly_patterns import (
    WeeklyPatternPredictor,
    complete_maple_weeks,
    fit_causal_profiles_by_origin,
    infer_weekly_group,
)


def daily_series(start: date, length: int, peak_weekday: int | None) -> list[DailyGain]:
    values = []
    for offset in range(length):
        day = start + timedelta(days=offset)
        gain = 100 if peak_weekday is not None and day.weekday() == peak_weekday else 1
        values.append(DailyGain(day, gain))
    return values


class WeeklyPatternModelTest(unittest.TestCase):
    def test_complete_weeks_use_thursday_to_wednesday(self):
        observations = daily_series(date(2026, 1, 1), 15, 4)
        weeks = complete_maple_weeks(observations)
        self.assertEqual(len(weeks), 2)
        self.assertEqual(weeks[0][0][0].weekday(), 3)
        self.assertEqual(weeks[0][-1][0].weekday(), 2)

    def test_group_is_inferred_from_weekly_peak(self):
        start = date(2026, 1, 1)
        early = infer_weekly_group(daily_series(start, 30, 4))
        weekend = infer_weekly_group(daily_series(start, 30, 6))
        late = infer_weekly_group(daily_series(start, 30, 1))
        self.assertEqual(early.primary_group, "early")
        self.assertEqual(weekend.primary_group, "weekend")
        self.assertEqual(late.primary_group, "late")

    def test_zero_and_tied_weeks_do_not_create_false_group(self):
        start = date(2026, 1, 1)
        zeros = [DailyGain(start + timedelta(days=index), 0) for index in range(30)]
        tied = [
            DailyGain(
                start + timedelta(days=index),
                100 if (start + timedelta(days=index)).weekday() in (3, 4) else 1,
            )
            for index in range(30)
        ]
        self.assertIsNone(infer_weekly_group(zeros).primary_group)
        self.assertIsNone(infer_weekly_group(tied).primary_group)

    def test_prediction_preserves_sma_horizon_total(self):
        start = date(2026, 1, 1)
        train = {"a": daily_series(start, 60, 4)}
        profiles = fit_causal_profiles_by_origin(train)
        history = train["a"][:30]
        target_dates = [start + timedelta(days=index) for index in range(30, 60)]
        predictor = WeeklyPatternPredictor(profiles, 1.0, True, 0)
        predictions = predictor(history, target_dates)
        expected_total = statistics.fmean(item.gain for item in history) * 30
        self.assertAlmostEqual(sum(predictions), expected_total)

    def test_earliest_profile_does_not_use_later_observations(self):
        start = date(2026, 1, 1)
        original = daily_series(start, 61, 4)
        changed = list(original)
        for index in range(30, len(changed)):
            changed[index] = DailyGain(changed[index].day, 10_000_000)
        original_profiles = fit_causal_profiles_by_origin({"a": original})
        changed_profiles = fit_causal_profiles_by_origin({"a": changed})
        earliest = min(original_profiles)
        self.assertEqual(
            original_profiles[earliest].group_factors,
            changed_profiles[earliest].group_factors,
        )

    def test_profiles_are_available_through_latest_history_cutoff(self):
        start = date(2026, 1, 1)
        observations = daily_series(start, 60, 4)
        profiles = fit_causal_profiles_by_origin({"a": observations})
        self.assertIn(observations[-1].day, profiles)
        history = observations[-30:]
        future_dates = [
            observations[-1].day + timedelta(days=offset) for offset in range(1, 31)
        ]
        predictions = WeeklyPatternPredictor(profiles, 1.0, True, 0)(
            history, future_dates
        )
        self.assertEqual(len(predictions), 30)


if __name__ == "__main__":
    unittest.main()
