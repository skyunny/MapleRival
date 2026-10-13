"""Remove fully dormant characters from existing character-level split files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from maplerival.experience import required_exp


SPLIT_NAMES = ("train", "validation", "test")
LEVEL_BANDS = ((260, 269), (270, 279), (280, 289), (290, 299))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def gain_statistics(gains: list[int]) -> dict:
    if not gains:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "zero_rate": 0.0,
        }
    return {
        "count": len(gains),
        "mean": round(statistics.fmean(gains), 3),
        "median": round(percentile(gains, 0.5), 3),
        "p75": round(percentile(gains, 0.75), 3),
        "p90": round(percentile(gains, 0.9), 3),
        "p95": round(percentile(gains, 0.95), 3),
        "zero_rate": round(sum(gain == 0 for gain in gains) / len(gains), 6),
    }


def level_starts(maximum_level: int) -> dict[int, int]:
    starts = {200: 0}
    for level in range(201, maximum_level + 1):
        starts[level] = starts[level - 1] + required_exp(level - 1)
    return starts


def daily_gains(records: list[dict], starts: dict[int, int]) -> list[int]:
    ordered = sorted(records, key=lambda item: item["date"])
    gains: list[int] = []
    for previous, current in zip(ordered, ordered[1:]):
        if (
            date.fromisoformat(current["date"])
            - date.fromisoformat(previous["date"])
        ).days != 1:
            raise ValueError(f"Non-consecutive dates for {current['ocid']}")
        previous_total = starts[int(previous["level"])] + int(previous["experience"])
        current_total = starts[int(current["level"])] + int(current["experience"])
        gain = current_total - previous_total
        if gain < 0:
            raise ValueError(
                f"Negative cumulative EXP gain for {current['ocid']} on {current['date']}"
            )
        gains.append(gain)
    return gains


def band_name(level: int) -> str:
    for lower, upper in LEVEL_BANDS:
        if lower <= level <= upper:
            return f"{lower}-{upper}"
    return "outside-260-299"


def is_fully_dormant(records: list[dict]) -> bool:
    states = {(int(item["level"]), int(item["experience"])) for item in records}
    return len(states) == 1


def filter_file(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records_by_ocid: dict[str, list[dict]] = defaultdict(list)
    for record in payload["records"]:
        records_by_ocid[record["ocid"]].append(record)

    dormant_ocids = {
        ocid for ocid, records in records_by_ocid.items() if is_fully_dormant(records)
    }
    characters = [
        character
        for character in payload["characters"]
        if character["ocid"] not in dormant_ocids
    ]
    records = [
        record for record in payload["records"] if record["ocid"] not in dormant_ocids
    ]
    active_ocids = {character["ocid"] for character in characters}
    if active_ocids != {record["ocid"] for record in records}:
        raise RuntimeError(f"Character/record mismatch after filtering {path}")

    starts = level_starts(max(int(record["level"]) for record in records))
    active_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        active_records[record["ocid"]].append(record)
    gains = [
        gain
        for character_records in active_records.values()
        for gain in daily_gains(character_records, starts)
    ]
    band_counts = Counter(
        band_name(int(character["selection_level"])) for character in characters
    )

    metadata = payload["metadata"]
    metadata["character_count"] = len(characters)
    metadata["record_count"] = len(records)
    metadata["level_band_counts"] = dict(sorted(band_counts.items()))
    metadata["daily_gain_statistics"] = gain_statistics(gains)
    metadata["dormant_filter"] = {
        "definition": (
            "level and experience never changed during the complete observed period; "
            "this is a conservative subset of the 30-day zero-change rule"
        ),
        "removed_character_count": len(dormant_ocids),
    }
    payload["characters"] = characters
    payload["records"] = records
    atomic_write_json(path, payload)

    return payload, {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "character_count": len(characters),
        "record_count": len(records),
        "removed_character_count": len(dormant_ocids),
        "daily_gain_statistics": metadata["daily_gain_statistics"],
        "level_band_counts": metadata["level_band_counts"],
    }


def main() -> None:
    args = parse_args()
    summaries: dict[str, dict] = {}
    removed_total = 0
    for split_name in SPLIT_NAMES:
        path = args.split_dir / f"maple_exp_{split_name}.json"
        _, summary = filter_file(path)
        summaries[split_name] = summary
        removed_total += summary["removed_character_count"]

    manifest_path = args.split_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = summaries
    manifest["active_only_filter"] = {
        "definition": (
            "level and experience never changed during the complete observed period; "
            "this is a conservative subset of the 30-day zero-change rule"
        ),
        "removed_character_count": removed_total,
        "remaining_character_count": sum(
            summary["character_count"] for summary in summaries.values()
        ),
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest["active_only_filter"] | {"files": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
