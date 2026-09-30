import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import PROJECT_ROOT, get_settings
from .database import (
    add_rival,
    get_alert_state,
    get_character,
    get_notification_channel,
    initialize,
    list_rival_alert_states,
    list_rivals,
    remove_rival,
)
from .alerting import STATE_LABELS, evaluate_alerts
from .multi_alerting import (
    evaluate_owner_alerts,
    hourly_all_rivals_loop,
    initialize_rival_baseline,
    notify_rival_added,
    notify_rival_removed,
    register_owner_webhook,
)
from .nexon import NexonApiError, collect_characters, collect_history
from .service import build_dashboard


STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    initialize(settings.database_path)
    stop_event = asyncio.Event()
    alert_task = asyncio.create_task(hourly_all_rivals_loop(settings, stop_event))
    try:
        yield
    finally:
        stop_event.set()
        await alert_task


app = FastAPI(title="Maple Rival", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/dashboard")
async def dashboard() -> dict:
    settings = get_settings()
    return build_dashboard(settings.database_path, settings.history_days)


class CharacterRequest(BaseModel):
    name: str


class WebhookRequest(BaseModel):
    url: str


def clean_name(value: str) -> str:
    name = value.strip()
    if not 2 <= len(name) <= 12:
        raise HTTPException(status_code=422, detail="캐릭터명은 2~12자로 입력해 주세요.")
    return name


@app.post("/api/characters/select")
async def select_character(request: CharacterRequest) -> dict:
    settings = get_settings()
    name = clean_name(request.name)
    try:
        await collect_characters(settings, [name])
    except (NexonApiError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=404, detail="캐릭터를 찾지 못했거나 데이터를 조회할 수 없습니다.") from exc
    return {"name": name, "dashboardUrl": f"/dashboard?owner={name}"}


@app.get("/api/owners/{owner_name}/dashboard")
async def owner_dashboard(owner_name: str) -> dict:
    settings = get_settings()
    owner_name = clean_name(owner_name)
    if get_character(settings.database_path, owner_name) is None:
        raise HTTPException(status_code=404, detail="먼저 내 캐릭터를 검색해 주세요.")
    rivals = list_rivals(settings.database_path, owner_name)
    return {
        "owner": owner_name,
        "rivals": rivals,
        "maximumRivals": 3,
        "dashboard": build_dashboard(
            settings.database_path, settings.history_days, [owner_name, *rivals]
        ),
    }


@app.post("/api/owners/{owner_name}/rivals")
async def create_rival(owner_name: str, request: CharacterRequest) -> dict:
    settings = get_settings()
    owner_name = clean_name(owner_name)
    rival_name = clean_name(request.name)
    if get_character(settings.database_path, owner_name) is None:
        raise HTTPException(status_code=404, detail="먼저 내 캐릭터를 검색해 주세요.")
    if owner_name == rival_name:
        raise HTTPException(status_code=422, detail="내 캐릭터를 라이벌로 등록할 수 없습니다.")
    try:
        await collect_characters(settings, [rival_name])
        add_rival(settings.database_path, owner_name, rival_name)
        await initialize_rival_baseline(settings, owner_name, rival_name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (NexonApiError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=404, detail="라이벌 캐릭터를 찾지 못했거나 데이터를 조회할 수 없습니다.") from exc
    result = await owner_dashboard(owner_name)
    result["notification"] = await notify_rival_added(settings, owner_name, rival_name)
    return result


@app.delete("/api/owners/{owner_name}/rivals/{rival_name}")
async def delete_rival(owner_name: str, rival_name: str) -> dict:
    settings = get_settings()
    owner_name = clean_name(owner_name)
    rival_name = clean_name(rival_name)
    if not remove_rival(settings.database_path, owner_name, rival_name):
        raise HTTPException(status_code=404, detail="등록된 라이벌을 찾을 수 없습니다.")
    result = await owner_dashboard(owner_name)
    result["notification"] = await notify_rival_removed(settings, owner_name, rival_name)
    return result


@app.post("/api/owners/{owner_name}/refresh")
async def refresh_owner_dashboard(owner_name: str) -> dict:
    settings = get_settings()
    owner_name = clean_name(owner_name)
    if get_character(settings.database_path, owner_name) is None:
        raise HTTPException(status_code=404, detail="먼저 내 캐릭터를 검색해 주세요.")
    names = [owner_name, *list_rivals(settings.database_path, owner_name)]
    try:
        await collect_characters(settings, names)
    except (NexonApiError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail="캐릭터 데이터를 갱신하지 못했습니다.") from exc
    return await owner_dashboard(owner_name)


@app.get("/api/owners/{owner_name}/webhook")
async def owner_webhook_status(owner_name: str) -> dict:
    settings = get_settings()
    owner_name = clean_name(owner_name)
    channel = get_notification_channel(settings.database_path, owner_name)
    states = list_rival_alert_states(settings.database_path, owner_name)
    return {
        "configured": channel is not None,
        "updatedAt": channel["updated_at"] if channel else None,
        "intervalMinutes": 60,
        "rivalStates": [
            {
                "rivalName": state["rival_name"],
                "state": state["state"],
                "stateLabel": STATE_LABELS.get(state["state"], state["state"]),
                "gapPercentagePoints": state["gap_percentage_points"],
                "checkedAt": state["checked_at"],
                "notifiedAt": state["notified_at"],
            }
            for state in states
        ],
    }


@app.put("/api/owners/{owner_name}/webhook")
async def configure_owner_webhook(owner_name: str, request: WebhookRequest) -> dict:
    settings = get_settings()
    owner_name = clean_name(owner_name)
    if get_character(settings.database_path, owner_name) is None:
        raise HTTPException(status_code=404, detail="먼저 내 캐릭터를 검색해 주세요.")
    try:
        return await register_owner_webhook(settings, owner_name, request.url.strip())
    except (ValueError, NexonApiError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/owners/{owner_name}/alerts/check")
async def check_owner_alerts(owner_name: str) -> dict:
    settings = get_settings()
    owner_name = clean_name(owner_name)
    try:
        return await evaluate_owner_alerts(settings, owner_name)
    except (ValueError, NexonApiError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/refresh")
async def refresh() -> dict:
    settings = get_settings()
    try:
        collected = await collect_history(settings)
    except NexonApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"collected": collected, "dashboard": build_dashboard(settings.database_path, settings.history_days)}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "project": PROJECT_ROOT.name}


@app.get("/api/alerts/status")
async def alert_status() -> dict:
    state = get_alert_state(get_settings().database_path)
    if state is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "ownCharacter": state["own_character_name"],
        "rivalCharacter": state["rival_character_name"],
        "state": state["state"],
        "stateLabel": STATE_LABELS.get(state["state"], state["state"]),
        "gapPercentagePoints": state["gap_percentage_points"],
        "initializedAt": state["initialized_at"],
        "checkedAt": state["checked_at"],
        "notifiedAt": state["notified_at"],
        "intervalMinutes": 60,
    }


@app.post("/api/alerts/check")
async def check_alerts() -> dict:
    try:
        return await evaluate_alerts(get_settings())
    except (NexonApiError, ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
