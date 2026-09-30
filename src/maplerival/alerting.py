import asyncio
import hashlib
from datetime import datetime
from urllib.parse import urlparse

import httpx

from .config import Settings
from .database import (
    get_alert_state,
    log_notification,
    save_alert_state,
    upsert_character,
)
from .nexon import NexonApiError, NexonClient, system_ssl_context


OWN_SAFE = "OWN_LEADING_1_PLUS"
OWN_CLOSE = "OWN_LEADING_UNDER_1"
RIVAL_CLOSE = "RIVAL_LEADING_UNDER_1"
RIVAL_SAFE = "RIVAL_LEADING_1_PLUS"

STATE_LABELS = {
    OWN_SAFE: "내 캐릭터가 1%p 이상 앞서는 중",
    OWN_CLOSE: "내 캐릭터가 1%p 미만 앞서는 중",
    RIVAL_CLOSE: "라이벌이 1%p 미만 앞서는 중",
    RIVAL_SAFE: "라이벌이 1%p 이상 앞서는 중",
}


def progress_score(level: int, exp_rate: float) -> float:
    return level * 100 + exp_rate


def classify_gap(gap: float) -> str:
    if gap >= 1:
        return OWN_SAFE
    if gap >= 0:
        return OWN_CLOSE
    if gap > -1:
        return RIVAL_CLOSE
    return RIVAL_SAFE


def transition_message(previous: str, current: str, own: str, rival: str, gap: float) -> str:
    distance = abs(gap)
    if current == OWN_SAFE:
        if previous in (RIVAL_CLOSE, RIVAL_SAFE):
            headline = f"🏁 {own}님, 재역전 성공! 주인공 자리는 다시 내 것"
        else:
            headline = f"💨 {rival}님이 백미러에서 작아지는 중!"
        detail = f"{own}님이 현재 {distance:.3f}%p 앞서고 있어요. 이 흐름 그대로 쭉!"
    elif current == OWN_CLOSE:
        if previous in (RIVAL_CLOSE, RIVAL_SAFE):
            headline = f"⚡ {own}님이 다시 코앞에서 추월했어요!"
        else:
            headline = f"😳 {rival}님 숨소리가 들립니다. 방심 금지!"
        detail = f"{own}님이 {distance:.3f}%p 앞서는 초접전이에요."
    elif current == RIVAL_CLOSE:
        if previous in (OWN_CLOSE, OWN_SAFE):
            headline = f"🚨 앗, {rival}님에게 역전당했어요!"
        else:
            headline = f"🔥 {own}님이 {rival}님 턱밑까지 따라붙었습니다!"
        detail = f"차이는 단 {distance:.3f}%p. 아직 한 판 더 남았어요."
    else:
        headline = f"🏃 {rival}님이 시야 밖으로 도망가는 중!"
        detail = f"현재 {distance:.3f}%p 차이예요. 추격 부스터가 필요합니다."
    return f"{headline}\n{detail}\n현재 상태: {STATE_LABELS[current]}"


def webhook_fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def validate_webhook_url(url: str) -> None:
    parsed = urlparse(url)
    valid_hosts = {"discord.com", "discordapp.com"}
    if parsed.scheme != "https" or parsed.hostname not in valid_hosts or not parsed.path.startswith("/api/webhooks/"):
        raise ValueError("Discord 공식 웹훅 URL만 사용할 수 있습니다.")


async def current_profiles(settings: Settings) -> dict[str, dict]:
    client = NexonClient(settings.nexon_api_key)
    profiles = {}
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        for name in (settings.own_character_name, settings.rival_character_name):
            ocid = await client.ocid(name)
            basic = await client.basic(ocid)
            profile = {
                "name": name,
                "ocid": ocid,
                "world": basic.get("world_name"),
                "class_name": basic.get("character_class"),
                "level": int(basic["character_level"]),
                "exp": int(basic["character_exp"]),
                "exp_rate": float(basic["character_exp_rate"]),
                "checked_at": checked_at,
            }
            profiles[name] = profile
            upsert_character(
                settings.database_path,
                name,
                ocid,
                profile["world"],
                profile["class_name"],
                profile["level"],
                profile["exp_rate"],
                profile["exp"],
                checked_at,
            )
    finally:
        await client.close()
    return profiles


async def send_discord(webhook_url: str, message: str) -> None:
    validate_webhook_url(webhook_url)
    async with httpx.AsyncClient(timeout=15, verify=system_ssl_context()) as client:
        response = await client.post(
            webhook_url,
            json={"content": message, "username": "Maple Rival"},
        )
        response.raise_for_status()


async def evaluate_alerts(settings: Settings, *, notify: bool = True) -> dict:
    if not settings.nexon_api_key:
        raise NexonApiError(".env에 NEXON_API_KEY가 없습니다.")
    if not settings.discord_webhook_url:
        raise ValueError(".env에 DISCORD_WEBHOOK_URL이 없습니다.")
    validate_webhook_url(settings.discord_webhook_url)

    profiles = await current_profiles(settings)
    own = profiles[settings.own_character_name]
    rival = profiles[settings.rival_character_name]
    gap = progress_score(own["level"], own["exp_rate"]) - progress_score(rival["level"], rival["exp_rate"])
    state = classify_gap(gap)
    checked_at = own["checked_at"]
    fingerprint = webhook_fingerprint(settings.discord_webhook_url)
    previous = get_alert_state(settings.database_path)
    should_initialize = (
        previous is None
        or previous["webhook_hash"] != fingerprint
        or previous["own_character_name"] != settings.own_character_name
        or previous["rival_character_name"] != settings.rival_character_name
    )

    if should_initialize:
        save_alert_state(
            settings.database_path,
            own_name=settings.own_character_name,
            rival_name=settings.rival_character_name,
            state=state,
            gap=gap,
            webhook_hash=fingerprint,
            checked_at=checked_at,
            initialize=True,
        )
        return {"action": "initialized", "state": state, "stateLabel": STATE_LABELS[state], "gap": gap}

    if previous["state"] == state:
        save_alert_state(
            settings.database_path,
            own_name=settings.own_character_name,
            rival_name=settings.rival_character_name,
            state=state,
            gap=gap,
            webhook_hash=fingerprint,
            checked_at=checked_at,
        )
        return {"action": "unchanged", "state": state, "stateLabel": STATE_LABELS[state], "gap": gap}

    message = transition_message(
        previous["state"], state, settings.own_character_name, settings.rival_character_name, gap
    )
    if not notify:
        return {"action": "would_notify", "state": state, "stateLabel": STATE_LABELS[state], "gap": gap, "message": message}

    try:
        await send_discord(settings.discord_webhook_url, message)
    except Exception as exc:
        log_notification(
            settings.database_path,
            previous_state=previous["state"], new_state=state, gap=gap,
            message=message, delivery_status="failed", error_message=str(exc)[:500], created_at=checked_at,
        )
        raise

    save_alert_state(
        settings.database_path,
        own_name=settings.own_character_name,
        rival_name=settings.rival_character_name,
        state=state,
        gap=gap,
        webhook_hash=fingerprint,
        checked_at=checked_at,
        notified_at=checked_at,
    )
    log_notification(
        settings.database_path,
        previous_state=previous["state"], new_state=state, gap=gap,
        message=message, delivery_status="sent", error_message=None, created_at=checked_at,
    )
    return {"action": "notified", "state": state, "stateLabel": STATE_LABELS[state], "gap": gap, "message": message}


async def hourly_alert_loop(settings: Settings, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await evaluate_alerts(settings)
        except Exception:
            # Failures are isolated from the web server; the next hourly run retries.
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.alert_interval_seconds)
        except TimeoutError:
            continue
