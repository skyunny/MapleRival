import asyncio
import base64
import hashlib
from datetime import datetime

from cryptography.fernet import Fernet, InvalidToken

from .alerting import (
    STATE_LABELS,
    classify_gap,
    progress_score,
    send_discord,
    transition_message,
    validate_webhook_url,
    webhook_fingerprint,
)
from .config import Settings
from .database import (
    get_notification_channel,
    get_rival_alert_state,
    list_notification_owners,
    list_rivals,
    log_rival_notification,
    log_webhook_event,
    replace_owner_alert_baselines,
    save_notification_channel,
    save_rival_alert_state,
    upsert_character,
)
from .nexon import NexonApiError, NexonClient


def _fernet(secret: str) -> Fernet:
    if not secret:
        raise ValueError("웹훅 암호화에 사용할 서버 비밀값이 없습니다.")
    key = base64.urlsafe_b64encode(hashlib.sha256(f"maple-rival:{secret}".encode()).digest())
    return Fernet(key)


def encrypt_webhook(secret: str, url: str) -> str:
    return _fernet(secret).encrypt(url.encode()).decode()


def decrypt_webhook(secret: str, ciphertext: str) -> str:
    try:
        return _fernet(secret).decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("저장된 웹훅 정보를 복호화할 수 없습니다.") from exc


async def fetch_profiles(settings: Settings, names: list[str]) -> dict[str, dict]:
    client = NexonClient(settings.nexon_api_key)
    profiles = {}
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        for name in dict.fromkeys(names):
            ocid = await client.ocid(name)
            basic = await client.basic(ocid)
            profile = {
                "name": name,
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
                basic.get("world_name"),
                basic.get("character_class"),
                profile["level"],
                profile["exp_rate"],
                profile["exp"],
                checked_at,
            )
    finally:
        await client.close()
    return profiles


def gap_between(own: dict, rival: dict) -> float:
    return progress_score(own["level"], own["exp_rate"]) - progress_score(
        rival["level"], rival["exp_rate"]
    )


def monitored_rivals_text(rivals: list[str]) -> str:
    if rivals:
        return f"현재 모니터링 중인 라이벌 캐릭터 : {', '.join(rivals)}"
    return "모니터링 중인 라이벌 캐릭터가 없습니다. 라이벌 캐릭터를 등록하세요."


def webhook_registered_message(rivals: list[str]) -> str:
    return f"MAPLE RIVAL : 라이벌 경험치 모니터링을 시작합니다.\n{monitored_rivals_text(rivals)}"


def rival_added_message(rival_name: str, rivals: list[str]) -> str:
    return f"{rival_name} 경험치 모니터링을 시작합니다.\n{monitored_rivals_text(rivals)}"


def rival_removed_message(rival_name: str, rivals: list[str]) -> str:
    return f"{rival_name} 경험치 모니터링을 종료합니다.\n{monitored_rivals_text(rivals)}"


def safe_delivery_error(exc: Exception) -> str:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status_code})" if status_code else "")


async def send_owner_event(
    settings: Settings,
    owner_name: str,
    *,
    event_type: str,
    message: str,
    rival_name: str | None = None,
    webhook_url: str | None = None,
) -> dict:
    if webhook_url is None:
        channel = get_notification_channel(settings.database_path, owner_name)
        if channel is None:
            return {"configured": False, "sent": False, "error": None}
        webhook_url = decrypt_webhook(settings.encryption_secret, channel["webhook_ciphertext"])
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        await send_discord(webhook_url, message)
    except Exception as exc:
        error = safe_delivery_error(exc)
        log_webhook_event(
            settings.database_path,
            owner_name=owner_name,
            rival_name=rival_name,
            event_type=event_type,
            message=message,
            delivery_status="failed",
            error_message=error,
            created_at=timestamp,
        )
        return {"configured": True, "sent": False, "error": error}
    log_webhook_event(
        settings.database_path,
        owner_name=owner_name,
        rival_name=rival_name,
        event_type=event_type,
        message=message,
        delivery_status="sent",
        error_message=None,
        created_at=timestamp,
    )
    return {"configured": True, "sent": True, "error": None}


async def register_owner_webhook(settings: Settings, owner_name: str, webhook_url: str) -> dict:
    validate_webhook_url(webhook_url)
    rivals = list_rivals(settings.database_path, owner_name)
    profiles = await fetch_profiles(settings, [owner_name, *rivals])
    timestamp = profiles[owner_name]["checked_at"]
    states = []
    for rival_name in rivals:
        gap = gap_between(profiles[owner_name], profiles[rival_name])
        states.append({"rival_name": rival_name, "gap": gap, "state": classify_gap(gap)})
    save_notification_channel(
        settings.database_path,
        owner_name,
        encrypt_webhook(settings.encryption_secret, webhook_url),
        webhook_fingerprint(webhook_url),
        timestamp,
    )
    replace_owner_alert_baselines(settings.database_path, owner_name, states, timestamp)
    notification = await send_owner_event(
        settings,
        owner_name,
        event_type="webhook_registered",
        message=webhook_registered_message(rivals),
        webhook_url=webhook_url,
    )
    return {
        "configured": True,
        "owner": owner_name,
        "rivalStates": [
            {
                "rivalName": item["rival_name"],
                "state": item["state"],
                "stateLabel": STATE_LABELS[item["state"]],
                "gapPercentagePoints": item["gap"],
            }
            for item in states
        ],
        "initializedAt": timestamp,
        "notification": notification,
    }


async def initialize_rival_baseline(settings: Settings, owner_name: str, rival_name: str) -> dict | None:
    if get_notification_channel(settings.database_path, owner_name) is None:
        return None
    profiles = await fetch_profiles(settings, [owner_name, rival_name])
    gap = gap_between(profiles[owner_name], profiles[rival_name])
    state = classify_gap(gap)
    save_rival_alert_state(
        settings.database_path,
        owner_name=owner_name,
        rival_name=rival_name,
        state=state,
        gap=gap,
        checked_at=profiles[owner_name]["checked_at"],
        initialize=True,
    )
    return {"state": state, "gap": gap}


async def notify_rival_added(settings: Settings, owner_name: str, rival_name: str) -> dict:
    rivals = list_rivals(settings.database_path, owner_name)
    return await send_owner_event(
        settings,
        owner_name,
        event_type="rival_added",
        rival_name=rival_name,
        message=rival_added_message(rival_name, rivals),
    )


async def notify_rival_removed(settings: Settings, owner_name: str, rival_name: str) -> dict:
    rivals = list_rivals(settings.database_path, owner_name)
    return await send_owner_event(
        settings,
        owner_name,
        event_type="rival_removed",
        rival_name=rival_name,
        message=rival_removed_message(rival_name, rivals),
    )


async def evaluate_owner_alerts(settings: Settings, owner_name: str) -> dict:
    channel = get_notification_channel(settings.database_path, owner_name)
    if channel is None:
        return {"owner": owner_name, "action": "not_configured", "results": []}
    webhook_url = decrypt_webhook(settings.encryption_secret, channel["webhook_ciphertext"])
    validate_webhook_url(webhook_url)
    rivals = list_rivals(settings.database_path, owner_name)
    if not rivals:
        return {"owner": owner_name, "action": "no_rivals", "results": []}
    profiles = await fetch_profiles(settings, [owner_name, *rivals])
    timestamp = profiles[owner_name]["checked_at"]
    results = []
    for rival_name in rivals:
        gap = gap_between(profiles[owner_name], profiles[rival_name])
        state = classify_gap(gap)
        previous = get_rival_alert_state(settings.database_path, owner_name, rival_name)
        if previous is None:
            save_rival_alert_state(
                settings.database_path,
                owner_name=owner_name,
                rival_name=rival_name,
                state=state,
                gap=gap,
                checked_at=timestamp,
                initialize=True,
            )
            results.append({"rival": rival_name, "action": "initialized", "state": state, "gap": gap})
            continue
        if previous["state"] == state:
            save_rival_alert_state(
                settings.database_path,
                owner_name=owner_name,
                rival_name=rival_name,
                state=state,
                gap=gap,
                checked_at=timestamp,
            )
            results.append({"rival": rival_name, "action": "unchanged", "state": state, "gap": gap})
            continue

        message = transition_message(previous["state"], state, owner_name, rival_name, gap)
        try:
            await send_discord(webhook_url, message)
        except Exception as exc:
            log_rival_notification(
                settings.database_path,
                owner_name=owner_name,
                rival_name=rival_name,
                previous_state=previous["state"],
                new_state=state,
                gap=gap,
                message=message,
                delivery_status="failed",
                error_message=str(exc)[:500],
                created_at=timestamp,
            )
            results.append({"rival": rival_name, "action": "failed", "state": state, "gap": gap})
            continue

        save_rival_alert_state(
            settings.database_path,
            owner_name=owner_name,
            rival_name=rival_name,
            state=state,
            gap=gap,
            checked_at=timestamp,
            notified_at=timestamp,
        )
        log_rival_notification(
            settings.database_path,
            owner_name=owner_name,
            rival_name=rival_name,
            previous_state=previous["state"],
            new_state=state,
            gap=gap,
            message=message,
            delivery_status="sent",
            error_message=None,
            created_at=timestamp,
        )
        results.append({"rival": rival_name, "action": "notified", "state": state, "gap": gap})
    return {"owner": owner_name, "action": "checked", "results": results}


async def hourly_all_rivals_loop(settings: Settings, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        for owner_name in list_notification_owners(settings.database_path):
            try:
                await evaluate_owner_alerts(settings, owner_name)
            except Exception:
                # One owner's API or webhook failure must not stop other owners.
                continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.alert_interval_seconds)
        except TimeoutError:
            continue
