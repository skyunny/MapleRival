"""Build an active-only 1,000-character cohort for each five-level band.

EXP snapshots are collected daily. Luna Petite pet count and Sol Janus level are
collected only at the first and last dates. Progress is committed to SQLite so
an interrupted run resumes without repeating completed characters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import dotenv_values


BASE_URL = "https://open.api.nexon.com/maplestory/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEVEL_BANDS = tuple((level, level + 4) for level in range(260, 300, 5))
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--target-per-band", type=int, default=1000)
    parser.add_argument("--candidate-pool-per-band", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--requests-per-second", type=float, default=350)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def band_name(level: int) -> str | None:
    for lower, upper in LEVEL_BANDS:
        if lower <= level <= upper:
            return f"{lower}-{upper}"
    return None


def janus_category(level: int) -> str:
    if level == 0:
        return "0누스"
    if level < 10:
        return "1누스"
    if level < 20:
        return "2누스"
    return "3누스"


def pet_category(count: int) -> str:
    if count == 0:
        return "0석펫"
    if count <= 2:
        return "1석펫"
    return "3석펫"


class RateGate:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            await asyncio.sleep(delay)


class Client:
    def __init__(self, key: str, concurrency: int, requests_per_second: float) -> None:
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-nxopen-api-key": key},
            timeout=40,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate_gate = RateGate(requests_per_second)

    async def close(self) -> None:
        await self.http.aclose()

    async def get(self, path: str, params: dict[str, object]) -> dict:
        async with self.semaphore:
            for attempt in range(10):
                await self.rate_gate.wait()
                try:
                    response = await self.http.get(path, params=params)
                    if response.status_code < 400:
                        return response.json()
                    # Invalid OCIDs/dates are permanent errors. Retrying them made
                    # a rejected candidate take more than a minute.
                    if response.status_code != 429 and response.status_code < 500:
                        response.raise_for_status()
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 429 and exc.response.status_code < 500:
                        raise
                    if attempt == 9:
                        raise
                    await asyncio.sleep(min(30, 1.5 * (attempt + 1)))
                except httpx.RequestError:
                    if attempt == 9:
                        raise
                    await asyncio.sleep(min(30, 1.5 * (attempt + 1)))
        raise RuntimeError("NEXON API retry loop ended unexpectedly")

    async def ranking_page(self, day: date, page: int) -> list[dict]:
        data = await self.get("/ranking/overall", {"date": day.isoformat(), "page": page})
        return data.get("ranking") or []

    async def ranking(self, ocid: str, day: date) -> dict | None:
        # Basic character history contains the same level/EXP fields and is an
        # OCID-keyed lookup. Overall-ranking searches become the main bottleneck
        # when hundreds of thousands of dated snapshots are requested.
        return await self.get(
            "/character/basic", {"ocid": ocid, "date": day.isoformat()}
        )

    async def ocid(self, nickname: str) -> str:
        data = await self.get("/id", {"character_name": nickname})
        return data["ocid"]

    async def dated(self, endpoint: str, ocid: str, day: date) -> dict:
        return await self.get(endpoint, {"ocid": ocid, "date": day.isoformat()})


def open_checkpoint(path: Path, args: argparse.Namespace) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts (
          nickname TEXT PRIMARY KEY, ocid TEXT, band TEXT NOT NULL,
          status TEXT NOT NULL, detail TEXT
        );
        CREATE TABLE IF NOT EXISTS characters (
          ocid TEXT PRIMARY KEY, nickname TEXT NOT NULL, selection_level INTEGER NOT NULL,
          selection_date TEXT NOT NULL, band TEXT NOT NULL, features TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS records (
          ocid TEXT NOT NULL, day TEXT NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY (ocid, day)
        );
        """
    )
    expected = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "target_per_band": str(args.target_per_band),
        "input": str(args.input.resolve()),
        "seed": str(args.seed),
    }
    saved = dict(connection.execute("SELECT key, value FROM settings"))
    if saved and any(saved.get(key) != value for key, value in expected.items()):
        raise SystemExit("Checkpoint parameters do not match this collection request")
    connection.executemany(
        "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", expected.items()
    )
    connection.commit()
    return connection


async def find_last_page(client: Client, day: date) -> int:
    low, high = 1, 2
    while await client.ranking_page(day, high):
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if await client.ranking_page(day, middle):
            low = middle
        else:
            high = middle
    return low


async def page_range(client: Client, day: date, last: int, lower: int, upper: int) -> tuple[int, int]:
    cache: dict[int, tuple[int, int] | None] = {}

    async def levels(page: int) -> tuple[int, int] | None:
        if page not in cache:
            rows = await client.ranking_page(day, page)
            values = [int(row["character_level"]) for row in rows]
            cache[page] = (min(values), max(values)) if values else None
        return cache[page]

    lo, hi = 1, last
    while lo < hi:
        middle = (lo + hi) // 2
        value = await levels(middle)
        if value and value[0] > upper:
            lo = middle + 1
        else:
            hi = middle
    first = lo
    lo, hi = first, last
    while lo < hi:
        middle = (lo + hi + 1) // 2
        value = await levels(middle)
        if value and value[1] >= lower:
            lo = middle
        else:
            hi = middle - 1
    return first, lo


async def candidate_pool(
    client: Client,
    day: date,
    excluded_names: set[str],
    attempted_names: set[str],
    pool_per_band: int,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    last_page = await find_last_page(client, day)
    result: dict[str, list[dict]] = {}
    print(f"ranking_last_page={last_page}", flush=True)
    for lower, upper in LEVEL_BANDS:
        name = f"{lower}-{upper}"
        first, last = await page_range(client, day, last_page, lower, upper)
        pages = list(range(first, last + 1))
        rng.shuffle(pages)
        candidates: list[dict] = []
        seen: set[str] = set()
        for page in pages:
            rows = await client.ranking_page(day, page)
            rng.shuffle(rows)
            for row in rows:
                nickname = row["character_name"]
                level = int(row["character_level"])
                if (
                    lower <= level <= upper
                    and nickname not in excluded_names
                    and nickname not in attempted_names
                    and nickname not in seen
                ):
                    candidates.append(
                        {
                            "nickname": nickname,
                            "selection_level": level,
                            "selection_exp": int(row["character_exp"]),
                            "band": name,
                        }
                    )
                    seen.add(nickname)
            if len(candidates) >= pool_per_band:
                break
        result[name] = candidates
        print(f"candidate_pool band={name} count={len(candidates)} pages={first}-{last}", flush=True)
    return result


def record_from_row(row: dict, ocid: str, fallback_name: str, day: date) -> dict:
    return {
        "nickname": row.get("character_name") or fallback_name,
        "ocid": ocid,
        "level": int(row["character_level"]),
        "experience": int(row["character_exp"]),
        "date": day.isoformat(),
        "weekday": WEEKDAYS[day.weekday()],
    }


def parse_features(pet: dict, hexa: dict, suffix: str) -> dict:
    count = sum(pet.get(f"pet_{index}_pet_type") == "루나 쁘띠" for index in range(1, 4))
    levels = [
        int(core.get("hexa_core_level") or 0)
        for core in hexa.get("character_hexa_core_equipment") or []
        if core.get("hexa_core_name") == "솔 야누스"
    ]
    level = max(levels, default=0)
    return {
        f"petite_luna_pet_count_{suffix}": count,
        f"petite_luna_pet_category_{suffix}": pet_category(count),
        f"sol_janus_level_{suffix}": level,
        f"sol_janus_category_{suffix}": janus_category(level),
    }


async def process_candidate(
    client: Client, candidate: dict, days: list[date], excluded_ocids: set[str]
) -> tuple[str | None, dict | None, list[dict], str]:
    nickname = candidate["nickname"]
    try:
        ocid = await client.ocid(nickname)
        if ocid in excluded_ocids:
            return ocid, None, [], "duplicate"
        start_row, end_row = await asyncio.gather(
            client.ranking(ocid, days[0]), client.ranking(ocid, days[-1])
        )
        if start_row is None or end_row is None:
            return ocid, None, [], "endpoint_ranking_not_found"
        start_state = (int(start_row["character_level"]), int(start_row["character_exp"]))
        end_state = (int(end_row["character_level"]), int(end_row["character_exp"]))
        if start_state == end_state:
            return ocid, None, [], "dormant"

        middle_days = days[1:-1]
        middle_rows = await asyncio.gather(*(client.ranking(ocid, day) for day in middle_days))
        if any(row is None for row in middle_rows):
            return ocid, None, [], "incomplete_ranking_history"
        rows = [start_row, *middle_rows, end_row]
        records = [
            record_from_row(row, ocid, nickname, day)
            for row, day in zip(rows, days, strict=True)
        ]
        pet_start, hexa_start, pet_end, hexa_end = await asyncio.gather(
            client.dated("/character/pet-equipment", ocid, days[0]),
            client.dated("/character/hexamatrix", ocid, days[0]),
            client.dated("/character/pet-equipment", ocid, days[-1]),
            client.dated("/character/hexamatrix", ocid, days[-1]),
        )
        start_suffix = days[0].isoformat().replace("-", "_")
        end_suffix = days[-1].isoformat().replace("-", "_")
        features = {
            **parse_features(pet_start, hexa_start, start_suffix),
            **parse_features(pet_end, hexa_end, end_suffix),
        }
        features["feature_date"] = days[-1].isoformat()
        features["petite_luna_pet_category"] = features[f"petite_luna_pet_category_{end_suffix}"]
        features["sol_janus_category"] = features[f"sol_janus_category_{end_suffix}"]
        character = {
            "nickname": records[0]["nickname"],
            "selection_level": candidate["selection_level"],
            "selection_date": days[0].isoformat(),
            "ocid": ocid,
            "band": candidate["band"],
            **features,
        }
        return ocid, character, records, "active"
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            return None, None, [], f"HTTP_{exc.response.status_code}"
        return None, None, [], type(exc).__name__


def commit_result(
    db: sqlite3.Connection,
    candidate: dict,
    ocid: str | None,
    character: dict | None,
    records: list[dict],
    status: str,
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO attempts(nickname, ocid, band, status, detail) VALUES (?, ?, ?, ?, ?)",
        (candidate["nickname"], ocid, candidate["band"], status, None),
    )
    if character is not None and ocid is not None:
        db.execute(
            "INSERT OR REPLACE INTO characters VALUES (?, ?, ?, ?, ?, ?)",
            (
                ocid,
                character["nickname"],
                character["selection_level"],
                character["selection_date"],
                character["band"],
                json.dumps(character, ensure_ascii=False),
            ),
        )
        db.executemany(
            "INSERT OR REPLACE INTO records VALUES (?, ?, ?)",
            ((ocid, row["date"], json.dumps(row, ensure_ascii=False)) for row in records),
        )


async def enrich_existing(
    client: Client, source: dict, db: sqlite3.Connection, days: list[date], batch_size: int
) -> None:
    already = {row[0] for row in db.execute("SELECT ocid FROM characters WHERE band = 'existing'")}
    pending = [character for character in source["characters"] if character["ocid"] not in already]
    start_suffix = days[0].isoformat().replace("-", "_")
    end_suffix = days[-1].isoformat().replace("-", "_")
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]

        async def fetch(character: dict) -> tuple[dict, dict]:
            ocid = character["ocid"]
            pet_start, hexa_start, pet_end, hexa_end = await asyncio.gather(
                client.dated("/character/pet-equipment", ocid, days[0]),
                client.dated("/character/hexamatrix", ocid, days[0]),
                client.dated("/character/pet-equipment", ocid, days[-1]),
                client.dated("/character/hexamatrix", ocid, days[-1]),
            )
            features = {
                **parse_features(pet_start, hexa_start, start_suffix),
                **parse_features(pet_end, hexa_end, end_suffix),
            }
            updated = dict(character)
            updated.update(features)
            updated["feature_date"] = days[-1].isoformat()
            updated["band"] = band_name(int(updated["selection_level"]))
            updated["petite_luna_pet_category"] = features[f"petite_luna_pet_category_{end_suffix}"]
            updated["sol_janus_category"] = features[f"sol_janus_category_{end_suffix}"]
            return character, updated

        results = await asyncio.gather(*(fetch(character) for character in batch))
        for original, updated in results:
            db.execute(
                "INSERT OR REPLACE INTO characters VALUES (?, ?, ?, ?, ?, ?)",
                (
                    updated["ocid"], updated["nickname"], updated["selection_level"],
                    updated["selection_date"], "existing",
                    json.dumps(updated, ensure_ascii=False),
                ),
            )
        db.commit()
        print(f"existing_features={min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def build_output(args: argparse.Namespace, source: dict, db: sqlite3.Connection) -> None:
    existing_ocids = {character["ocid"] for character in source["characters"]}
    characters_by_ocid: dict[str, dict] = {}
    for ocid, payload in db.execute("SELECT ocid, features FROM characters"):
        character = json.loads(payload)
        actual_band = band_name(int(character["selection_level"]))
        if actual_band:
            character["band"] = actual_band
            characters_by_ocid[ocid] = character

    by_band: dict[str, list[dict]] = {f"{lo}-{hi}": [] for lo, hi in LEVEL_BANDS}
    for character in characters_by_ocid.values():
        by_band[character["band"]].append(character)
    selected: list[dict] = []
    for name, characters in by_band.items():
        existing = sorted(
            (item for item in characters if item["ocid"] in existing_ocids),
            key=lambda item: item["nickname"],
        )
        new = sorted(
            (item for item in characters if item["ocid"] not in existing_ocids),
            key=lambda item: item["nickname"],
        )
        chosen = (existing + new)[: args.target_per_band]
        if len(chosen) != args.target_per_band:
            raise RuntimeError(f"Band {name} has only {len(chosen)} active characters")
        selected.extend(chosen)
    selected_ocids = {character["ocid"] for character in selected}

    source_records = [record for record in source["records"] if record["ocid"] in selected_ocids]
    new_records = [
        json.loads(payload)
        for ocid, payload in db.execute("SELECT ocid, payload FROM records")
        if ocid in selected_ocids and ocid not in existing_ocids
    ]
    records = source_records + new_records
    records.sort(key=lambda item: (item["ocid"], item["date"]))
    counts = Counter(record["ocid"] for record in records)
    expected_days = (args.end - args.start).days + 1
    if len(selected_ocids) != len(selected) or any(counts[ocid] != expected_days for ocid in selected_ocids):
        raise RuntimeError("Duplicate characters or incomplete daily records in final dataset")
    if len({(record["ocid"], record["date"]) for record in records}) != len(records):
        raise RuntimeError("Duplicate OCID/date record in final dataset")

    suffixes = [day.isoformat().replace("-", "_") for day in (args.start, args.end)]
    feature_names = [
        f"{base}_{suffix}"
        for suffix in suffixes
        for base in (
            "petite_luna_pet_count", "petite_luna_pet_category",
            "sol_janus_level", "sol_janus_category",
        )
    ] + ["petite_luna_pet_category", "sol_janus_category"]
    payload = {
        "metadata": {
            "start_date": args.start.isoformat(),
            "end_date": args.end.isoformat(),
            "character_count": len(selected),
            "record_count": len(records),
            "records_per_character": expected_days,
            "active_definition": "level or experience changed between start and end",
            "selection_method": "1,000 active characters per five-level start-date band",
            "daily_data_source": "NEXON character/basic and previously collected ranking/overall snapshots",
            "level_bands": [list(band) for band in LEVEL_BANDS],
            "level_band_counts": {name: len(items[: args.target_per_band]) for name, items in by_band.items()},
            "character_features": feature_names,
            "feature_dates": [args.start.isoformat(), args.end.isoformat()],
            "random_seed": args.seed,
            "checkpoint": str(args.checkpoint.resolve()),
        },
        "characters": sorted(selected, key=lambda item: (item["selection_level"], item["nickname"])),
        "records": records,
        "missing_records": [],
    }
    atomic_write_json(args.output, payload)


async def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise SystemExit("--end must not be before --start")
    key = (dotenv_values(PROJECT_ROOT / ".env").get("NEXON_API_KEY") or "").strip()
    if not key:
        raise SystemExit("NEXON_API_KEY is not configured")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    db = open_checkpoint(args.checkpoint, args)
    client = Client(key, args.concurrency, args.requests_per_second)
    days = [args.start + timedelta(days=offset) for offset in range((args.end - args.start).days + 1)]
    try:
        if args.probe:
            rows = await client.ranking_page(args.start, 1)
            print(json.dumps({"row_count": len(rows), "date": args.start.isoformat()}))
            return

        await enrich_existing(client, source, db, days, args.batch_size)
        existing_names = {character["nickname"] for character in source["characters"]}
        excluded_ocids = {character["ocid"] for character in source["characters"]}
        attempted_names = {row[0] for row in db.execute("SELECT nickname FROM attempts")}
        pools = await candidate_pool(
            client, args.start, existing_names, attempted_names,
            args.candidate_pool_per_band, args.seed,
        )
        existing_counts = Counter(
            band_name(int(character["selection_level"])) for character in source["characters"]
        )
        accepted_counts = Counter(
            row[0] for row in db.execute("SELECT band FROM characters WHERE band != 'existing'")
        )

        for lower, upper in LEVEL_BANDS:
            name = f"{lower}-{upper}"
            needed = args.target_per_band - existing_counts[name] - accepted_counts[name]
            candidates = pools[name]
            cursor = 0
            while needed > 0:
                batch = candidates[cursor : cursor + args.batch_size]
                if not batch:
                    raise RuntimeError(
                        f"Candidate pool exhausted for {name}; rerun with a larger --candidate-pool-per-band"
                    )
                cursor += len(batch)
                results = await asyncio.gather(
                    *(process_candidate(client, candidate, days, excluded_ocids) for candidate in batch)
                )
                for candidate, (ocid, character, records, status) in zip(batch, results, strict=True):
                    if character is not None and needed > 0:
                        commit_result(db, candidate, ocid, character, records, status)
                        excluded_ocids.add(character["ocid"])
                        needed -= 1
                        accepted_counts[name] += 1
                    else:
                        commit_result(db, candidate, ocid, None, [], status if character is None else "surplus")
                db.commit()
                status_counts = dict(
                    db.execute("SELECT status, COUNT(*) FROM attempts GROUP BY status")
                )
                print(
                    f"checkpoint band={name} accepted={accepted_counts[name]} "
                    f"remaining={needed} attempts={sum(status_counts.values())} statuses={status_counts}",
                    flush=True,
                )

        build_output(args, source, db)
        print(f"output={args.output.resolve()}", flush=True)
    finally:
        await client.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
