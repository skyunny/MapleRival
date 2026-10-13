"""Split the collected EXP panel by character while matching gain distributions."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from maplerival.experience import required_exp


SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}
LEVEL_BANDS = ((260, 269), (270, 279), (280, 289), (290, 299))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--trials", type=int, default=2000)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band_for(level: int) -> tuple[int, int]:
    for band in LEVEL_BANDS:
        if band[0] <= level <= band[1]:
            return band
    raise ValueError(f"Selection level {level} is outside the expected bands")


def build_level_starts(maximum_level: int) -> dict[int, int]:
    starts = {200: 0}
    for level in range(201, maximum_level + 1):
        starts[level] = starts[level - 1] + required_exp(level - 1)
    return starts


def character_gains(records: list[dict], level_starts: dict[int, int]) -> list[int]:
    ordered = sorted(records, key=lambda item: item["date"])
    gains: list[int] = []
    for previous, current in zip(ordered, ordered[1:]):
        previous_date = date.fromisoformat(previous["date"])
        current_date = date.fromisoformat(current["date"])
        if (current_date - previous_date).days != 1:
            raise ValueError(f"Non-consecutive dates for {current['ocid']}")
        previous_total = level_starts[int(previous["level"])] + int(previous["experience"])
        current_total = level_starts[int(current["level"])] + int(current["experience"])
        gain = current_total - previous_total
        if gain < 0:
            raise ValueError(f"Negative cumulative EXP gain for {current['ocid']} on {current['date']}")
        gains.append(gain)
    return gains


def gain_statistics(gains: list[int]) -> dict:
    return {
        "count": len(gains),
        "mean": round(statistics.fmean(gains), 3),
        "median": round(percentile(gains, 0.5), 3),
        "p75": round(percentile(gains, 0.75), 3),
        "p90": round(percentile(gains, 0.9), 3),
        "p95": round(percentile(gains, 0.95), 3),
        "zero_rate": round(sum(gain == 0 for gain in gains) / len(gains), 6),
    }


def choose_split(
    characters: list[dict],
    gains_by_ocid: dict[str, list[int]],
    seed: int,
    trials: int,
) -> tuple[dict[str, list[dict]], float, list[float]]:
    character_count = len(characters)
    split_counts = {
        "train": round(character_count * SPLIT_RATIOS["train"]),
        "validation": round(character_count * SPLIT_RATIOS["validation"]),
    }
    split_counts["test"] = character_count - sum(split_counts.values())
    positive_logs = sorted(
        math.log1p(gain)
        for gains in gains_by_ocid.values()
        for gain in gains
        if gain > 0
    )
    boundaries = [percentile(positive_logs, index / 10) for index in range(1, 10)]
    histogram_size = 11
    histogram_by_ocid: dict[str, list[int]] = {}
    activity_by_ocid: dict[str, float] = {}
    mean_gain_by_ocid: dict[str, float] = {}
    p90_gain_by_ocid: dict[str, float] = {}
    p95_gain_by_ocid: dict[str, float] = {}
    global_histogram = [0] * histogram_size
    for character in characters:
        ocid = character["ocid"]
        gains = gains_by_ocid[ocid]
        histogram = [0] * histogram_size
        for gain in gains:
            bucket = 0 if gain == 0 else 1 + bisect.bisect_right(boundaries, math.log1p(gain))
            histogram[bucket] += 1
            global_histogram[bucket] += 1
        histogram_by_ocid[ocid] = histogram
        activity_by_ocid[ocid] = statistics.fmean(math.log1p(gain) for gain in gains)
        mean_gain_by_ocid[ocid] = statistics.fmean(gains)
        p90_gain_by_ocid[ocid] = percentile(gains, 0.9)
        p95_gain_by_ocid[ocid] = percentile(gains, 0.95)

    global_total = sum(global_histogram)
    global_distribution = [count / global_total for count in global_histogram]
    global_activity = statistics.fmean(activity_by_ocid.values())
    activity_scale = statistics.pstdev(activity_by_ocid.values()) or 1.0
    balance_features = (mean_gain_by_ocid, p90_gain_by_ocid, p95_gain_by_ocid)
    balance_targets = [statistics.fmean(feature.values()) for feature in balance_features]
    balance_scales = [statistics.pstdev(feature.values()) or 1.0 for feature in balance_features]

    by_band: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for character in characters:
        by_band[band_for(int(character["selection_level"]))].append(character)
    for items in by_band.values():
        items.sort(key=lambda item: activity_by_ocid[item["ocid"]])

    best_score = float("inf")
    best: dict[str, list[dict]] | None = None
    for trial in range(trials):
        rng = random.Random(seed + trial)
        candidate = {name: [] for name in SPLIT_RATIOS}
        leftovers: list[dict] = []
        for band in LEVEL_BANDS:
            items = by_band[band]
            for start in range(0, len(items), 5):
                block = items[start : start + 5].copy()
                rng.shuffle(block)
                if len(block) == 5:
                    candidate["train"].extend(block[:3])
                    candidate["validation"].append(block[3])
                    candidate["test"].append(block[4])
                else:
                    leftovers.extend(block)
        rng.shuffle(leftovers)
        for character in leftovers:
            available = [
                name
                for name, target in split_counts.items()
                if len(candidate[name]) < target
            ]
            if not available:
                raise RuntimeError("No split has capacity for remaining characters")
            name = max(
                available,
                key=lambda item: split_counts[item] - len(candidate[item]),
            )
            candidate[name].append(character)

        if any(len(candidate[name]) != count for name, count in split_counts.items()):
            raise RuntimeError("Failed to produce exact 6:2:2 split counts")

        score = 0.0
        for name, split_characters in candidate.items():
            aggregate = [0] * histogram_size
            for character in split_characters:
                for index, count in enumerate(histogram_by_ocid[character["ocid"]]):
                    aggregate[index] += count
            total = sum(aggregate)
            distribution = [count / total for count in aggregate]
            score += sum(
                ((actual - expected) ** 2) / max(expected, 1e-9)
                for actual, expected in zip(distribution, global_distribution)
            )
            split_activity = statistics.fmean(
                activity_by_ocid[character["ocid"]] for character in split_characters
            )
            score += ((split_activity - global_activity) / activity_scale) ** 2
            for feature, target, scale in zip(
                balance_features, balance_targets, balance_scales
            ):
                split_value = statistics.fmean(
                    feature[character["ocid"]] for character in split_characters
                )
                score += 4 * ((split_value - target) / scale) ** 2
        if score < best_score:
            best_score = score
            best = candidate

    if best is None:
        raise RuntimeError("No split candidate was generated")
    return best, best_score, boundaries


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    characters = source["characters"]
    records = source["records"]
    if not characters or not records:
        raise SystemExit("The input dataset is empty")

    records_by_ocid: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_ocid[record["ocid"]].append(record)
    if any(len(items) != 93 for items in records_by_ocid.values()):
        raise SystemExit("Every character must have exactly 93 daily records")

    maximum_level = max(int(record["level"]) for record in records)
    level_starts = build_level_starts(maximum_level)
    gains_by_ocid = {
        ocid: character_gains(items, level_starts)
        for ocid, items in records_by_ocid.items()
    }
    splits, objective, boundaries = choose_split(
        characters, gains_by_ocid, args.seed, args.trials
    )

    output_files: dict[str, dict] = {}
    all_split_ocids: set[str] = set()
    for split_name, split_characters in splits.items():
        split_ocids = {character["ocid"] for character in split_characters}
        if all_split_ocids & split_ocids:
            raise RuntimeError("A character was assigned to more than one split")
        all_split_ocids.update(split_ocids)
        split_records = [record for record in records if record["ocid"] in split_ocids]
        split_records.sort(key=lambda item: (item["nickname"], item["date"]))
        split_characters.sort(key=lambda item: item["nickname"])
        split_gains = [gain for ocid in split_ocids for gain in gains_by_ocid[ocid]]
        band_counts = Counter(
            f"{band_for(int(character['selection_level']))[0]}-"
            f"{band_for(int(character['selection_level']))[1]}"
            for character in split_characters
        )
        payload = {
            "metadata": {
                **source["metadata"],
                "split": split_name,
                "split_unit": "character",
                "split_ratio": "6:2:2",
                "character_count": len(split_characters),
                "record_count": len(split_records),
                "random_seed": args.seed,
                "stratification": "level band plus optimized log daily EXP gain histogram",
                "level_band_counts": dict(sorted(band_counts.items())),
                "daily_gain_statistics": gain_statistics(split_gains),
            },
            "characters": split_characters,
            "records": split_records,
            "missing_records": [],
        }
        path = args.output_dir / f"maple_exp_{split_name}.json"
        atomic_write_json(path, payload)
        output_files[split_name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "character_count": len(split_characters),
            "record_count": len(split_records),
            "daily_gain_statistics": payload["metadata"]["daily_gain_statistics"],
            "level_band_counts": payload["metadata"]["level_band_counts"],
        }

    if len(all_split_ocids) != len(characters):
        raise RuntimeError("The split files do not cover all input characters")

    manifest = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "split_ratio": "6:2:2",
        "split_unit": "character",
        "random_seed": args.seed,
        "optimization_trials": args.trials,
        "optimization_objective": objective,
        "positive_log_gain_decile_boundaries": boundaries,
        "files": output_files,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
