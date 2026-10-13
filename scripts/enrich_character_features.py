"""Add pet and HEXA character-level features from NEXON Open API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values


BASE_URL = "https://open.api.nexon.com/maplestory/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument(
        "--field-suffix",
        default="",
        help="Suffix added to feature names, for example _2026_06_26",
    )
    return parser.parse_args()


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Client:
    def __init__(self, key: str, concurrency: int) -> None:
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-nxopen-api-key": key},
            timeout=30,
        )
        self.limit = asyncio.Semaphore(concurrency)

    async def close(self) -> None:
        await self.http.aclose()

    async def get(self, endpoint: str, ocid: str, day: str) -> dict:
        async with self.limit:
            for attempt in range(8):
                try:
                    response = await self.http.get(
                        endpoint, params={"ocid": ocid, "date": day}
                    )
                    if response.status_code != 429 and response.status_code < 500:
                        response.raise_for_status()
                        return response.json()
                    response.raise_for_status()
                except (httpx.RequestError, httpx.HTTPStatusError):
                    if attempt == 7:
                        raise
                    await asyncio.sleep(min(20, 1.5 * (attempt + 1)))
        raise RuntimeError("NEXON API retry loop ended unexpectedly")


def parse_features(pet: dict, hexa: dict, day: str, field_suffix: str) -> dict:
    petite_count = sum(
        pet.get(f"pet_{index}_pet_type") == "루나 쁘띠" for index in range(1, 4)
    )
    sol_janus = [
        int(core.get("hexa_core_level") or 0)
        for core in (hexa.get("character_hexa_core_equipment") or [])
        if core.get("hexa_core_name") == "솔 야누스"
    ]
    return {
        f"petite_luna_pet_count{field_suffix}": petite_count,
        f"sol_janus_level{field_suffix}": max(sol_janus, default=0),
    }


async def main() -> None:
    args = parse_args()
    key = (dotenv_values(PROJECT_ROOT / ".env").get("NEXON_API_KEY") or "").strip()
    if not key:
        raise SystemExit("NEXON_API_KEY is not configured")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or args.output.with_name(f"{args.output.stem}.checkpoint.json")
    completed: dict[str, dict] = {}
    errors: dict[str, str] = {}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            saved.get("input") != str(args.input.resolve())
            or saved.get("date") != args.date
            or saved.get("field_suffix", "") != args.field_suffix
        ):
            raise SystemExit("Checkpoint parameters do not match")
        completed = saved.get("features", {})
        errors = saved.get("errors", {})
        print(f"resume completed={len(completed)}/{len(source['characters'])}", flush=True)

    client = Client(key, args.concurrency)
    try:
        pending = [c for c in source["characters"] if c["ocid"] not in completed]
        for start in range(0, len(pending), args.checkpoint_every):
            batch = pending[start : start + args.checkpoint_every]

            async def fetch(character: dict) -> tuple[str, dict | None, str | None]:
                ocid = character["ocid"]
                try:
                    pet, hexa = await asyncio.gather(
                        client.get("/character/pet-equipment", ocid, args.date),
                        client.get("/character/hexamatrix", ocid, args.date),
                    )
                    return ocid, parse_features(
                        pet, hexa, args.date, args.field_suffix
                    ), None
                except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                    return ocid, None, type(exc).__name__

            for ocid, features, error in await asyncio.gather(*(fetch(c) for c in batch)):
                if features is not None:
                    completed[ocid] = features
                    errors.pop(ocid, None)
                else:
                    errors[ocid] = error or "unknown"
            atomic_write(
                checkpoint,
                {
                    "input": str(args.input.resolve()),
                    "date": args.date,
                    "field_suffix": args.field_suffix,
                    "features": completed,
                    "errors": errors,
                },
            )
            print(
                f"checkpoint completed={len(completed)}/{len(source['characters'])} "
                f"errors={len(errors)}",
                flush=True,
            )
    finally:
        await client.close()

    if errors or len(completed) != len(source["characters"]):
        raise RuntimeError(f"Feature collection incomplete: {len(errors)} errors")
    for character in source["characters"]:
        character.update(completed[character["ocid"]])
    feature_names = [
        f"petite_luna_pet_count{args.field_suffix}",
        f"sol_janus_level{args.field_suffix}",
    ]
    existing_features = list(source["metadata"].get("character_features", []))
    source["metadata"]["character_features"] = list(
        dict.fromkeys(existing_features + feature_names)
    )
    feature_dates = dict(source["metadata"].get("character_feature_dates", {}))
    feature_dates.update({name: args.date for name in feature_names})
    source["metadata"]["character_feature_dates"] = feature_dates
    atomic_write(args.output, source)
    print(f"output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
