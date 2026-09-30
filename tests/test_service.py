import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from maplerival.database import add_rival, initialize, list_rivals, remove_rival
from maplerival.service import build_dashboard


class DashboardServiceTest(unittest.TestCase):
    def test_daily_gain_and_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.db"
            initialize(path)
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.executemany(
                    "INSERT INTO characters(id, name, ocid) VALUES (?, ?, ?)",
                    [(1, "A", "ocid-a"), (2, "B", "ocid-b")],
                    )
                    connection.executemany(
                    "INSERT INTO exp_snapshots(character_id, snapshot_date, level, exp, ranking) VALUES (?, ?, ?, ?, ?)",
                    [
                        (1, "2026-01-01", 295, 100, 2),
                        (1, "2026-01-02", 295, 130, 2),
                        (2, "2026-01-01", 295, 150, 1),
                        (2, "2026-01-02", 295, 170, 1),
                    ],
                    )
            result = build_dashboard(path, 15)
            self.assertEqual(result["leader"], "B")
            self.assertEqual(result["gap"], "40")
            first = next(item for item in result["characters"] if item["name"] == "A")
            self.assertEqual(first["history"][1]["dailyGain"], "30")
            self.assertEqual(result["experienceScale"]["minLevel"], 295)
            self.assertEqual(result["experienceScale"]["maxLevel"], 295)
            self.assertEqual(result["experienceScale"]["ticks"][-1]["label"], "Lv.296")
            self.assertAlmostEqual(first["history"][0]["progressRate"], 100 / 870403132500696 * 100)

    def test_rivals_are_limited_and_removable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.db"
            initialize(path)
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.executemany(
                        "INSERT INTO characters(id, name, ocid) VALUES (?, ?, ?)",
                        [(1, "ME", "me"), (2, "A", "a"), (3, "B", "b"), (4, "C", "c"), (5, "D", "d")],
                    )
            for rival in ("A", "B", "C"):
                add_rival(path, "ME", rival)
            self.assertEqual(list_rivals(path, "ME"), ["A", "B", "C"])
            with self.assertRaisesRegex(ValueError, "최대 3명"):
                add_rival(path, "ME", "D")
            self.assertTrue(remove_rival(path, "ME", "B"))
            self.assertEqual(list_rivals(path, "ME"), ["A", "C"])


if __name__ == "__main__":
    unittest.main()

