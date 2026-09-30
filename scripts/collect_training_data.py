"""Collect a reproducible MapleStory EXP training cohort from NEXON Open API."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import dotenv_values


BASE_URL = "https://open.api.nexon.com/maplestory/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEVEL_BANDS = ((260, 269), (270, 279), (280, 289), (290, 299))
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class ApiClient:
    def __init__(self, api_key: str, concurrency: int) -> None:
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-nxopen-api-key": api_key},
            timeout=30,
        )
        self.limit = asyncio.Semaphore(concurrency)

    async def close(self) -> None:
        await self.http.aclose()

    async def get(self, path: str, params: dict[str, object]) -> dict:
        async with self.limit:
            for attempt in range(8):
                response = await self.http.get(path, params=params)
                if response.status_code != 429:
                    break
                await asyncio.sleep(min(20, 1.5 * (attempt + 1)))
            response.raise_for_status()
            return response.json()

    async def ranking_page(self, day: date, page: int) -> list[dict]:
        payload = await self.get(
            "/ranking/overall", {"date": day.isoformat(), "page": page}
        )
        return payload.get("ranking") or []

    async def basic(self, ocid: str, day: date) -> dict:
        return await self.get(
            "/character/basic", {"ocid": ocid, "date": day.isoformat()}
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--characters", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def band_for(level: int) -> tuple[int, int] | None:
    return next((band for band in LEVEL_BANDS if band[0] <= level <= band[1]), None)


async def find_last_page(client: ApiClient, day: date) -> int:
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


async def page_level_range(client: ApiClient, day: date, page: int) -> tuple[int, int] | None:
    rows = await client.ranking_page(day, page)
    levels = [int(row["character_level"]) for row in rows]
    return (min(levels), max(levels)) if levels else None


async def page_bounds_for_band(
    client: ApiClient, day: date, last_page: int, lower: int, upper: int
) -> tuple[int, int]:
    cache: dict[int, tuple[int, int] | None] = {}

    async def levels(page: int) -> tuple[int, int] | None:
        if page not in cache:
            cache[page] = await page_level_range(client, day, page)
        return cache[page]

    # First page whose minimum level is no greater than the band's upper edge.
    lo, hi = 1, last_page
    while lo < hi:
        mid = (lo + hi) // 2
        value = await levels(mid)
        if value and value[0] > upper:
            lo = mid + 1
        else:
            hi = mid
    first = lo

    # Last page whose maximum level is no lower than the band's lower edge.
    lo, hi = first, last_page
    while lo < hi:
        mid = (lo + hi + 1) // 2
        value = await levels(mid)
        if value and value[1] >= lower:
            lo = mid
        else:
            hi = mid - 1
    return first, lo


async def select_cohort(
    client: ApiClient, day: date, size: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    last_page = await find_last_page(client, day)
    base, remainder = divmod(size, len(LEVEL_BANDS))
    quotas = {band: base + (index < remainder) for index, band in enumerate(LEVEL_BANDS)}
    selected: list[dict] = []
    seen: set[str] = set()

    print(f"ranking_last_page={last_page}", flush=True)
    for band in LEVEL_BANDS:
        first, last = await page_bounds_for_band(client, day, last_page, *band)
        pages = list(range(first, last + 1))
        rng.shuffle(pages)
        candidates: list[dict] = []
        for page in pages:
            for row in await client.ranking_page(day, page):
                level = int(row["character_level"])
                ocid = row.get("ocid")
                if band_for(level) == band and ocid and ocid not in seen:
                    candidates.append(
                        {
                            "nickname": row["character_name"],
                            "ocid": ocid,
                            "selection_level": level,
                            "selection_date": day.isoformat(),
                        }
                    )
            if len(candidates) >= quotas[band] * 4:
                break
        rng.shuffle(candidates)
        chosen = candidates[: quotas[band]]
        if len(chosen) != quotas[band]:
            raise RuntimeError(f"Not enough candidates for levels {band}: {len(chosen)}")
        selected.extend(chosen)
        seen.update(item["ocid"] for item in chosen)
        print(f"selected_{band[0]}_{band[1]}={len(chosen)} pages={first}-{last}", flush=True)
    rng.shuffle(selected)
    return selected


async def collect_character(
    client: ApiClient, character: dict, days: list[date]
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []

    async def fetch(day: date) -> None:
        try:
            row = await client.basic(character["ocid"], day)
            records.append(
                {
                    "nickname": row.get("character_name") or character["nickname"],
                    "ocid": character["ocid"],
                    "level": int(row["character_level"]),
                    "experience": int(row["character_exp"]),
                    "date": day.isoformat(),
                    "weekday": WEEKDAYS[day.weekday()],
                }
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            errors.append({"ocid": character["ocid"], "date": day.isoformat(), "status": status})

    await asyncio.gather(*(fetch(day) for day in days))
    records.sort(key=lambda item: item["date"])
    errors.sort(key=lambda item: item["date"])
    return records, errors


async def main() -> None:
    args = parse_args()
    if args.end < args.start:
        raise SystemExit("--end must not be before --start")
    api_key = (dotenv_values(PROJECT_ROOT / ".env").get("NEXON_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("NEXON_API_KEY is not configured")

    client = ApiClient(api_key, args.concurrency)
    try:
        if args.probe:
            rows = await client.ranking_page(args.start, 1)
            levels = [int(row["character_level"]) for row in rows]
            print(json.dumps({"row_count": len(rows), "min_level": min(levels), "max_level": max(levels)}))
            return

        cohort = await select_cohort(client, args.start, args.characters, args.seed)
        days = [
            args.start + timedelta(days=offset)
            for offset in range((args.end - args.start).days + 1)
        ]
        all_records: list[dict] = []
        all_errors: list[dict] = []
        for index, character in enumerate(cohort, start=1):
            records, errors = await collect_character(client, character, days)
            all_records.extend(records)
            all_errors.extend(errors)
            if index % 10 == 0 or index == len(cohort):
                print(f"characters={index}/{len(cohort)} records={len(all_records)} errors={len(all_errors)}", flush=True)

        all_records.sort(key=lambda item: (item["nickname"], item["date"]))
        payload = {
            "metadata": {
                "start_date": args.start.isoformat(),
                "end_date": args.end.isoformat(),
                "character_count": len(cohort),
                "record_count": len(all_records),
                "selection_method": "equal stratified random sample by 10-level band from start-date overall ranking",
                "level_bands": [list(band) for band in LEVEL_BANDS],
                "random_seed": args.seed,
                "missing_record_count": len(all_errors),
            },
            "characters": cohort,
            "records": all_records,
            "missing_records": all_errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output={args.output.resolve()}", flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
