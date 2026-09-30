import asyncio
import ssl
from datetime import date, datetime, timedelta

import httpx
import truststore

from .config import Settings
from .database import upsert_character, upsert_snapshots


BASE_URL = "https://open.api.nexon.com/maplestory/v1"


def system_ssl_context() -> ssl.SSLContext:
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class NexonApiError(RuntimeError):
    pass


class NexonClient:
    def __init__(self, api_key: str) -> None:
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"x-nxopen-api-key": api_key},
            timeout=15,
            verify=system_ssl_context(),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def get(self, path: str, params: dict) -> dict:
        for attempt in range(4):
            response = await self.client.get(path, params=params)
            if response.status_code != 429:
                break
            await asyncio.sleep(1.5 * (attempt + 1))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise NexonApiError(f"NEXON API {response.status_code}: {response.text[:180]}") from exc
        await asyncio.sleep(0.23)
        return response.json()

    async def ocid(self, name: str) -> str:
        return (await self.get("/id", {"character_name": name}))["ocid"]

    async def basic(self, ocid: str, day: str | None = None) -> dict:
        params = {"ocid": ocid}
        if day is not None:
            params["date"] = day
        return await self.get("/character/basic", params)

    async def ranking(self, ocid: str, day: str) -> dict | None:
        result = await self.get("/ranking/overall", {"ocid": ocid, "date": day, "page": 1})
        rows = result.get("ranking") or []
        return rows[0] if rows else None


async def collect_characters(settings: Settings, names: tuple[str, ...] | list[str]) -> dict[str, int]:
    if not settings.nexon_api_key:
        raise NexonApiError(".env에 NEXON_API_KEY가 없습니다.")

    end_day = date.today() - timedelta(days=1)
    # One extra day is required to calculate the first displayed day's gain.
    days = [end_day - timedelta(days=offset) for offset in reversed(range(settings.history_days + 1))]
    client = NexonClient(settings.nexon_api_key)
    collected: dict[str, int] = {}
    try:
        for name in names:
            ocid = await client.ocid(name)
            basic = await client.basic(ocid)
            character_id = upsert_character(
                settings.database_path,
                name,
                ocid,
                basic.get("world_name"),
                basic.get("character_class"),
                int(basic["character_level"]) if basic.get("character_level") is not None else None,
                float(basic["character_exp_rate"]) if basic.get("character_exp_rate") is not None else None,
                int(basic["character_exp"]) if basic.get("character_exp") is not None else None,
                datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            snapshots = []
            for day in days:
                item = await client.ranking(ocid, day.isoformat())
                if item:
                    snapshots.append(
                        {
                            "date": day.isoformat(),
                            "level": int(item["character_level"]),
                            "exp": int(item["character_exp"]),
                            "ranking": item.get("ranking"),
                        }
                    )
            upsert_snapshots(settings.database_path, character_id, snapshots)
            collected[name] = len(snapshots)
    finally:
        await client.close()
    return collected


async def collect_history(settings: Settings) -> dict[str, int]:
    return await collect_characters(settings, settings.character_names)

