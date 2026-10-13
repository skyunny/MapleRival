"""Collect a reproducible MapleStory EXP training cohort from NEXON Open API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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
                try:
                    response = await self.http.get(path, params=params)
                    if response.status_code != 429 and response.status_code < 500:
                        response.raise_for_status()
                        return response.json()
                    response.raise_for_status()
                except (httpx.RequestError, httpx.HTTPStatusError):
                    if attempt == 7:
                        raise
                    await asyncio.sleep(min(20, 1.5 * (attempt + 1)))
            raise RuntimeError("NEXON API retry loop ended unexpectedly")

    async def ranking_page(self, day: date, page: int) -> list[dict]:
        payload = await self.get(
            "/ranking/overall", {"date": day.isoformat(), "page": page}
        )
        return payload.get("ranking") or []

    async def basic(self, ocid: str, day: date) -> dict:
        return await self.get(
            "/character/basic", {"ocid": ocid, "date": day.isoformat()}
        )

    async def ranking(self, ocid: str, day: date) -> dict | None:
        payload = await self.get(
            "/ranking/overall",
            {"ocid": ocid, "date": day.isoformat(), "page": 1},
        )
        rows = payload.get("ranking") or []
        return rows[0] if rows else None

    async def ocid(self, nickname: str) -> str:
        payload = await self.get("/id", {"character_name": nickname})
        return payload["ocid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--characters", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reserve-per-band", type=int, default=25)
    parser.add_argument(
        "--exclude-input",
        type=Path,
        action="append",
        default=[],
        help="JSON dataset whose nicknames and OCIDs must not be selected (repeatable)",
    )
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def collection_parameters(args: argparse.Namespace) -> dict:
    return {
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "character_count": args.characters,
        "random_seed": args.seed,
        "daily_data_source": "ranking-overall",
        "reserve_per_level_band": args.reserve_per_band,
        "exclude_inputs": [str(path.resolve()) for path in args.exclude_input],
    }


def checkpoint_path(args: argparse.Namespace) -> Path:
    return args.checkpoint or args.output.with_name(
        f"{args.output.stem}.checkpoint.json"
    )


def save_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    cohort: list[dict],
    completed_ocids: set[str],
    records: list[dict],
    errors: list[dict],
    status: str,
) -> None:
    atomic_write_json(
        path,
        {
            "version": 1,
            "status": status,
            "parameters": collection_parameters(args),
            "cohort": cohort,
            "completed_ocids": sorted(completed_ocids),
            "records": records,
            "missing_records": errors,
        },
    )


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
    client: ApiClient,
    day: date,
    size: int,
    seed: int,
    excluded_names: set[str] | None = None,
    excluded_ocids: set[str] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    excluded_names = excluded_names or set()
    excluded_ocids = excluded_ocids or set()
    last_page = await find_last_page(client, day)
    base, remainder = divmod(size, len(LEVEL_BANDS))
    quotas = {band: base + (index < remainder) for index, band in enumerate(LEVEL_BANDS)}
    selected: list[dict] = []
    seen_names: set[str] = set()

    print(f"ranking_last_page={last_page}", flush=True)
    for band in LEVEL_BANDS:
        first, last = await page_bounds_for_band(client, day, last_page, *band)
        print(f"band_{band[0]}_{band[1]}_pages={first}-{last}", flush=True)
        pages = list(range(first, last + 1))
        rng.shuffle(pages)
        candidates: list[dict] = []
        for page in pages:
            for row in await client.ranking_page(day, page):
                level = int(row["character_level"])
                nickname = row["character_name"]
                if (
                    band_for(level) == band
                    and nickname not in seen_names
                    and nickname not in excluded_names
                ):
                    candidates.append(
                        {
                            "nickname": nickname,
                            "selection_level": level,
                            "selection_date": day.isoformat(),
                        }
                    )
            if len(candidates) >= quotas[band] * 4:
                break
        rng.shuffle(candidates)
        async def resolve_candidate(character: dict) -> dict | None:
            try:
                character["ocid"] = await client.ocid(character["nickname"])
                return character
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    return None
                raise

        resolved = await asyncio.gather(*(resolve_candidate(item) for item in candidates))
        chosen = [
            item
            for item in resolved
            if item is not None and item["ocid"] not in excluded_ocids
        ][: quotas[band]]
        if len(chosen) != quotas[band]:
            raise RuntimeError(
                f"Not enough resolvable candidates for levels {band}: {len(chosen)}"
            )
        selected.extend(chosen)
        seen_names.update(item["nickname"] for item in chosen)
        print(f"selected_{band[0]}_{band[1]}={len(chosen)} pages={first}-{last}", flush=True)
    print(f"resolved_ocids={len(selected)}", flush=True)
    rng.shuffle(selected)
    return selected


async def collect_character(
    client: ApiClient, character: dict, days: list[date]
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []

    async def fetch(day: date) -> None:
        try:
            row = await client.ranking(character["ocid"], day)
            if row is None:
                errors.append(
                    {
                        "ocid": character["ocid"],
                        "date": day.isoformat(),
                        "status": None,
                        "reason": "ranking_not_found",
                    }
                )
                return
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

        checkpoint = checkpoint_path(args)
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("version") != 1:
                raise SystemExit(f"Unsupported checkpoint version: {saved.get('version')}")
            if saved.get("parameters") != collection_parameters(args):
                raise SystemExit("Checkpoint parameters do not match this collection request")
            cohort = saved["cohort"]
            completed_ocids = set(saved.get("completed_ocids", []))
            all_records = saved.get("records", [])
            all_errors = saved.get("missing_records", [])
            print(
                f"resume checkpoint={checkpoint.resolve()} "
                f"completed={len(completed_ocids)}/{len(cohort)}",
                flush=True,
            )
        else:
            excluded_names: set[str] = set()
            excluded_ocids: set[str] = set()
            for path in args.exclude_input:
                excluded = json.loads(path.read_text(encoding="utf-8"))
                for character in excluded.get("characters", []):
                    if character.get("nickname"):
                        excluded_names.add(character["nickname"])
                    if character.get("ocid"):
                        excluded_ocids.add(character["ocid"])
            candidate_count = args.characters + args.reserve_per_band * len(LEVEL_BANDS)
            cohort = await select_cohort(
                client,
                args.start,
                candidate_count,
                args.seed,
                excluded_names,
                excluded_ocids,
            )
            completed_ocids: set[str] = set()
            all_records: list[dict] = []
            all_errors: list[dict] = []
            save_checkpoint(
                checkpoint,
                args=args,
                cohort=cohort,
                completed_ocids=completed_ocids,
                records=all_records,
                errors=all_errors,
                status="collecting",
            )
            print(f"checkpoint initialized={checkpoint.resolve()}", flush=True)

        days = [
            args.start + timedelta(days=offset)
            for offset in range((args.end - args.start).days + 1)
        ]
        newly_completed = 0
        for index, character in enumerate(cohort, start=1):
            if character["ocid"] in completed_ocids:
                continue
            records, errors = await collect_character(client, character, days)
            all_records.extend(records)
            all_errors.extend(errors)
            completed_ocids.add(character["ocid"])
            newly_completed += 1
            if newly_completed % args.checkpoint_every == 0 or len(completed_ocids) == len(cohort):
                save_checkpoint(
                    checkpoint,
                    args=args,
                    cohort=cohort,
                    completed_ocids=completed_ocids,
                    records=all_records,
                    errors=all_errors,
                    status="collecting",
                )
                print(
                    f"checkpoint characters={len(completed_ocids)}/{len(cohort)} "
                    f"records={len(all_records)} errors={len(all_errors)}",
                    flush=True,
                )

        records_per_ocid: dict[str, int] = {}
        for record in all_records:
            records_per_ocid[record["ocid"]] = records_per_ocid.get(record["ocid"], 0) + 1
        ocids_with_errors = {error["ocid"] for error in all_errors}
        base, remainder = divmod(args.characters, len(LEVEL_BANDS))
        final_cohort: list[dict] = []
        complete_by_band: dict[tuple[int, int], list[dict]] = {}
        for band_index, band in enumerate(LEVEL_BANDS):
            quota = base + (band_index < remainder)
            valid = [
                character
                for character in cohort
                if band_for(character["selection_level"]) == band
                and character["ocid"] not in ocids_with_errors
                and records_per_ocid.get(character["ocid"]) == len(days)
            ]
            complete_by_band[band] = valid
            final_cohort.extend(valid[:quota])

        # Ranking history can be unavailable disproportionately in one level band.
        # Preserve the per-band quotas where possible, then fill any shortfall from
        # complete reserve characters in the other bands rather than discarding an
        # otherwise sufficient checkpoint.
        if len(final_cohort) < args.characters:
            selected_ocids = {character["ocid"] for character in final_cohort}
            reserves = [
                character
                for character in cohort
                if character["ocid"] not in selected_ocids
                and character["ocid"] not in ocids_with_errors
                and records_per_ocid.get(character["ocid"]) == len(days)
            ]
            needed = args.characters - len(final_cohort)
            if len(reserves) < needed:
                raise RuntimeError(
                    f"Only {len(final_cohort) + len(reserves)} complete characters; "
                    f"need {args.characters}. Collect more reserve candidates."
                )
            final_cohort.extend(reserves[:needed])

        final_ocids = {character["ocid"] for character in final_cohort}
        final_records = [record for record in all_records if record["ocid"] in final_ocids]
        final_errors = [error for error in all_errors if error["ocid"] in final_ocids]
        final_records.sort(key=lambda item: (item["nickname"], item["date"]))
        payload = {
            "metadata": {
                "start_date": args.start.isoformat(),
                "end_date": args.end.isoformat(),
                "character_count": len(final_cohort),
                "candidate_count": len(cohort),
                "record_count": len(final_records),
                "selection_method": "equal stratified random sample by 10-level band from start-date overall ranking",
                "level_bands": [list(band) for band in LEVEL_BANDS],
                "random_seed": args.seed,
                "missing_record_count": len(final_errors),
            },
            "characters": final_cohort,
            "records": final_records,
            "missing_records": final_errors,
        }
        atomic_write_json(args.output, payload)
        save_checkpoint(
            checkpoint,
            args=args,
            cohort=cohort,
            completed_ocids=completed_ocids,
            records=all_records,
            errors=all_errors,
            status="complete",
        )
        print(f"output={args.output.resolve()}", flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
