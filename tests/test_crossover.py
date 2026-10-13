import math
import unittest

import numpy as np

from maplerival.crossover import (
    HORIZON_DAYS,
    LOOKBACK_DAYS,
    build_samples,
    days_to_crossover,
    linear_eta,
    neighbor_pairs,
    pair_features,
)


def series(start, daily, days=80):
    return np.array([start + daily * day for day in range(days)], dtype=np.int64)


class CrossoverTests(unittest.TestCase):
    def test_days_to_crossover_finds_first_day(self):
        lead = series(1000, 10)
        lag = series(900, 20)  # gap 100 closes at 10/day -> day 10
        self.assertEqual(days_to_crossover(lead, lag, 0, 30), 10)

    def test_days_to_crossover_none_when_not_caught(self):
        lead = series(1000, 20)
        lag = series(900, 10)
        self.assertIsNone(days_to_crossover(lead, lag, 0, 30))

    def test_days_to_crossover_requires_full_horizon(self):
        lead = series(1000, 10, days=40)
        with self.assertRaises(ValueError):
            days_to_crossover(lead, lead, 20, 30)

    def test_linear_eta(self):
        self.assertEqual(linear_eta(100, 10, 20), 10)
        self.assertEqual(linear_eta(100, 20, 10), math.inf)

    def test_features_use_only_past(self):
        lead = series(10_000, 10)
        lag = series(9_000, 20)
        levels = np.full(80, 260)
        origin = 40
        base = pair_features(lead, lag, levels, levels, origin, 0)
        changed_future_lag = lag.copy()
        changed_future_lag[origin + 1 :] += 10**9
        self.assertEqual(
            base, pair_features(lead, changed_future_lag, levels, levels, origin, 0)
        )

    def test_build_samples_labels_and_roles(self):
        totals = np.vstack([series(10_000, 10), series(9_000, 50)])
        levels = np.full(totals.shape, 260)
        weekdays = np.arange(80) % 7
        samples = build_samples(
            totals, levels, weekdays, [(0, 1)], range(LOOKBACK_DAYS, 80 - HORIZON_DAYS)
        )
        self.assertTrue(samples)
        first = samples[0]
        # at origin 30: 10_300 vs 10_500, so character 1 already leads
        self.assertEqual((first.leader, first.lagger), (1, 0))
        self.assertIsNone(first.days_to_cross)
        for sample in samples:
            expected = days_to_crossover(
                totals[sample.leader], totals[sample.lagger], sample.origin, HORIZON_DAYS
            )
            self.assertEqual(sample.days_to_cross, expected)

    def test_build_samples_skips_inactive_pairs(self):
        totals = np.vstack([np.full(80, 5000), np.full(80, 4000)])
        levels = np.full(totals.shape, 260)
        samples = build_samples(
            totals, levels, np.arange(80) % 7, [(0, 1)], range(30, 50)
        )
        self.assertEqual(samples, [])

    def test_neighbor_pairs_sorted_by_total(self):
        anchor = np.array([50, 10, 30, 20])
        pairs = neighbor_pairs(anchor, np.array([0, 1, 2, 3]), 1)
        self.assertEqual(pairs, [(1, 3), (3, 2), (2, 0)])


if __name__ == "__main__":
    unittest.main()
