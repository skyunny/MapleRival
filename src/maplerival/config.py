from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    nexon_api_key: str
    discord_webhook_url: str
    encryption_secret: str
    database_path: Path
    character_names: tuple[str, ...] = ("꽃게쥬", "탕무스")
    own_character_name: str = "탕무스"
    rival_character_name: str = "꽃게쥬"
    history_days: int = 15
    alert_interval_seconds: int = 3600

def get_settings() -> Settings:
    values = {**dotenv_values(PROJECT_ROOT / ".env"), **os.environ}
    api_key = (values.get("NEXON_API_KEY") or "").strip()
    database_path = Path(
        (values.get("DATABASE_PATH") or str(PROJECT_ROOT / "maple_rival.db")).strip()
    ).expanduser()
    return Settings(
        nexon_api_key=api_key,
        discord_webhook_url=(values.get("DISCORD_WEBHOOK_URL") or "").strip(),
        encryption_secret=(values.get("APP_ENCRYPTION_KEY") or api_key).strip(),
        database_path=database_path,
    )

