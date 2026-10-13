"""Combine datasets, remove fully dormant characters, and validate EXP panels."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from maplerival.experience import required_exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def level_starts(maximum_level: int) -> dict[int, int]:
    starts = {200: 0}
    for level in range(201, maximum_level + 1):
        starts[level] = starts[level - 1] + required_exp(level - 1)
    return starts


def main() -> None:
    args = parse_args()
    characters_by_ocid: dict[str, dict] = {}
    records_by_ocid: dict[str, list[dict]] = defaultdict(list)
    source_counts: dict[str, int] = {}
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_counts[str(path.resolve())] = len(payload["characters"])
        for character in payload["characters"]:
            ocid = character["ocid"]
            if ocid in characters_by_ocid:
                raise SystemExit(f"Duplicate OCID across inputs: {ocid}")
            characters_by_ocid[ocid] = character
        for record in payload["records"]:
            records_by_ocid[record["ocid"]].append(record)

    if set(characters_by_ocid) != set(records_by_ocid):
        raise SystemExit("Character and record OCID sets do not match")
    dormant = {
        ocid
        for ocid, records in records_by_ocid.items()
        if len({(int(item["level"]), int(item["experience"])) for item in records}) == 1
    }
    active_ocids = set(characters_by_ocid) - dormant
    if not active_ocids:
        raise SystemExit("No active characters remain")

    all_active_records = [
        record
        for ocid in active_ocids
        for record in records_by_ocid[ocid]
    ]
    starts = level_starts(max(int(record["level"]) for record in all_active_records))
    expected_count: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    for ocid in active_ocids:
        records = sorted(records_by_ocid[ocid], key=lambda item: item["date"])
        dates = [date.fromisoformat(item["date"]) for item in records]
        if len(dates) != len(set(dates)):
            raise SystemExit(f"Duplicate date for {ocid}")
        if any(current - previous != timedelta(days=1) for previous, current in zip(dates, dates[1:])):
            raise SystemExit(f"Non-consecutive dates for {ocid}")
        if expected_count is None:
            expected_count = len(records)
            start_date, end_date = dates[0], dates[-1]
        if len(records) != expected_count or dates[0] != start_date or dates[-1] != end_date:
            raise SystemExit(f"Panel range mismatch for {ocid}")
        for previous, current in zip(records, records[1:]):
            previous_total = starts[int(previous["level"])] + int(previous["experience"])
            current_total = starts[int(current["level"])] + int(current["experience"])
            if current_total < previous_total:
                raise SystemExit(f"Negative daily EXP gain for {ocid} on {current['date']}")

    characters = sorted(
        (characters_by_ocid[ocid] for ocid in active_ocids),
        key=lambda item: item["nickname"],
    )
    records = sorted(all_active_records, key=lambda item: (item["nickname"], item["date"]))
    band_counts = Counter(
        f"{(int(item['selection_level']) // 10) * 10}-{(int(item['selection_level']) // 10) * 10 + 9}"
        for item in characters
    )
    payload = {
        "metadata": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "character_count": len(characters),
            "record_count": len(records),
            "records_per_character": expected_count,
            "selection_method": "combined active-only character cohorts",
            "level_bands": [[260, 269], [270, 279], [280, 289], [290, 299]],
            "level_band_counts": dict(sorted(band_counts.items())),
            "source_character_counts": source_counts,
            "dormant_definition": "level and experience unchanged for the complete observed period",
            "dormant_removed_character_count": len(dormant),
            "missing_record_count": 0,
        },
        "characters": characters,
        "records": records,
        "missing_records": [],
    }
    atomic_write(args.output, payload)
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
