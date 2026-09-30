import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize(path: Path) -> None:
    with closing(connect(path)) as connection:
        with connection:
            connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                ocid TEXT NOT NULL UNIQUE,
                world_name TEXT,
                character_class TEXT,
                current_level INTEGER,
                exp_rate REAL,
                current_exp INTEGER,
                current_checked_at TEXT
            );

            CREATE TABLE IF NOT EXISTS exp_snapshots (
                character_id INTEGER NOT NULL REFERENCES characters(id),
                snapshot_date TEXT NOT NULL,
                level INTEGER NOT NULL,
                exp INTEGER NOT NULL,
                ranking INTEGER,
                collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (character_id, snapshot_date)
            );

            CREATE TABLE IF NOT EXISTS alert_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                own_character_name TEXT NOT NULL,
                rival_character_name TEXT NOT NULL,
                state TEXT NOT NULL,
                gap_percentage_points REAL NOT NULL,
                webhook_hash TEXT NOT NULL,
                initialized_at TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                notified_at TEXT
            );

            CREATE TABLE IF NOT EXISTS notification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                previous_state TEXT NOT NULL,
                new_state TEXT NOT NULL,
                gap_percentage_points REAL NOT NULL,
                message TEXT NOT NULL,
                delivery_status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rivalries (
                owner_character_id INTEGER NOT NULL REFERENCES characters(id),
                rival_character_id INTEGER NOT NULL REFERENCES characters(id),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (owner_character_id, rival_character_id),
                CHECK (owner_character_id <> rival_character_id)
            );

            CREATE TABLE IF NOT EXISTS notification_channels (
                owner_character_id INTEGER PRIMARY KEY REFERENCES characters(id),
                webhook_ciphertext TEXT NOT NULL,
                webhook_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rival_alert_states (
                owner_character_id INTEGER NOT NULL REFERENCES characters(id),
                rival_character_id INTEGER NOT NULL REFERENCES characters(id),
                state TEXT NOT NULL,
                gap_percentage_points REAL NOT NULL,
                initialized_at TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                notified_at TEXT,
                PRIMARY KEY (owner_character_id, rival_character_id)
            );

            CREATE TABLE IF NOT EXISTS rival_notification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_character_id INTEGER NOT NULL REFERENCES characters(id),
                rival_character_id INTEGER NOT NULL REFERENCES characters(id),
                previous_state TEXT NOT NULL,
                new_state TEXT NOT NULL,
                gap_percentage_points REAL NOT NULL,
                message TEXT NOT NULL,
                delivery_status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS webhook_event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_character_id INTEGER NOT NULL REFERENCES characters(id),
                rival_name TEXT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                delivery_status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );
            """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(characters)")}
            if "current_level" not in columns:
                connection.execute("ALTER TABLE characters ADD COLUMN current_level INTEGER")
            if "exp_rate" not in columns:
                connection.execute("ALTER TABLE characters ADD COLUMN exp_rate REAL")
            if "current_exp" not in columns:
                connection.execute("ALTER TABLE characters ADD COLUMN current_exp INTEGER")
            if "current_checked_at" not in columns:
                connection.execute("ALTER TABLE characters ADD COLUMN current_checked_at TEXT")


def upsert_character(
    path: Path,
    name: str,
    ocid: str,
    world_name: str | None,
    character_class: str | None,
    current_level: int | None = None,
    exp_rate: float | None = None,
    current_exp: int | None = None,
    current_checked_at: str | None = None,
) -> int:
    with closing(connect(path)) as connection:
        with connection:
            connection.execute(
            """
            INSERT INTO characters(
                name, ocid, world_name, character_class, current_level,
                exp_rate, current_exp, current_checked_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                ocid = excluded.ocid,
                world_name = COALESCE(excluded.world_name, characters.world_name),
                character_class = COALESCE(excluded.character_class, characters.character_class),
                current_level = COALESCE(excluded.current_level, characters.current_level),
                exp_rate = COALESCE(excluded.exp_rate, characters.exp_rate),
                current_exp = COALESCE(excluded.current_exp, characters.current_exp),
                current_checked_at = COALESCE(excluded.current_checked_at, characters.current_checked_at)
            """,
            (
                name, ocid, world_name, character_class, current_level,
                exp_rate, current_exp, current_checked_at,
            ),
            )
            row = connection.execute("SELECT id FROM characters WHERE name = ?", (name,)).fetchone()
        return int(row["id"])


def upsert_snapshots(path: Path, character_id: int, snapshots: Iterable[dict]) -> None:
    rows = [
        (character_id, item["date"], item["level"], item["exp"], item.get("ranking"))
        for item in snapshots
    ]
    with closing(connect(path)) as connection:
        with connection:
            connection.executemany(
            """
            INSERT INTO exp_snapshots(character_id, snapshot_date, level, exp, ranking)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(character_id, snapshot_date) DO UPDATE SET
                level = excluded.level,
                exp = excluded.exp,
                ranking = excluded.ranking,
                collected_at = CURRENT_TIMESTAMP
            """,
            rows,
            )


def dashboard_rows(path: Path, limit: int, names: list[str] | None = None) -> list[sqlite3.Row]:
    with closing(connect(path)) as connection:
        name_filter = ""
        parameters: list[object] = []
        if names:
            placeholders = ",".join("?" for _ in names)
            name_filter = f"WHERE c.name IN ({placeholders})"
            parameters.extend(names)
        parameters.extend([limit, *names] if names else [limit])
        return connection.execute(
            f"""
            WITH recent_dates AS (
                SELECT DISTINCT s.snapshot_date
                FROM exp_snapshots s
                JOIN characters c ON c.id = s.character_id
                {name_filter}
                ORDER BY snapshot_date DESC
                LIMIT ?
            )
            SELECT c.name, c.world_name, c.character_class, c.current_level, c.exp_rate,
                   c.current_exp, c.current_checked_at,
                   s.snapshot_date, s.level, s.exp, s.ranking
            FROM exp_snapshots s
            JOIN characters c ON c.id = s.character_id
            WHERE s.snapshot_date IN (SELECT snapshot_date FROM recent_dates)
            {f"AND c.name IN ({','.join('?' for _ in names)})" if names else ''}
            ORDER BY s.snapshot_date, c.id
            """,
            parameters,
        ).fetchall()


def get_character(path: Path, name: str) -> dict | None:
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


def list_rivals(path: Path, owner_name: str) -> list[str]:
    with closing(connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT rival.name
            FROM rivalries r
            JOIN characters owner ON owner.id = r.owner_character_id
            JOIN characters rival ON rival.id = r.rival_character_id
            WHERE owner.name = ?
            ORDER BY r.created_at, rival.id
            """,
            (owner_name,),
        ).fetchall()
        return [row["name"] for row in rows]


def add_rival(path: Path, owner_name: str, rival_name: str, maximum: int = 3) -> None:
    with closing(connect(path)) as connection:
        with connection:
            owner = connection.execute("SELECT id FROM characters WHERE name = ?", (owner_name,)).fetchone()
            rival = connection.execute("SELECT id FROM characters WHERE name = ?", (rival_name,)).fetchone()
            if owner is None or rival is None:
                raise ValueError("등록되지 않은 캐릭터입니다.")
            if owner["id"] == rival["id"]:
                raise ValueError("내 캐릭터를 라이벌로 등록할 수 없습니다.")
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM rivalries WHERE owner_character_id = ?",
                (owner["id"],),
            ).fetchone()["count"]
            exists = connection.execute(
                "SELECT 1 FROM rivalries WHERE owner_character_id = ? AND rival_character_id = ?",
                (owner["id"], rival["id"]),
            ).fetchone()
            if exists:
                raise ValueError("이미 등록된 라이벌입니다.")
            if count >= maximum:
                raise ValueError(f"라이벌은 최대 {maximum}명까지 등록할 수 있습니다.")
            connection.execute(
                "INSERT INTO rivalries(owner_character_id, rival_character_id) VALUES (?, ?)",
                (owner["id"], rival["id"]),
            )


def remove_rival(path: Path, owner_name: str, rival_name: str) -> bool:
    with closing(connect(path)) as connection:
        with connection:
            connection.execute(
                """
                DELETE FROM rival_alert_states
                WHERE owner_character_id = (SELECT id FROM characters WHERE name = ?)
                  AND rival_character_id = (SELECT id FROM characters WHERE name = ?)
                """,
                (owner_name, rival_name),
            )
            cursor = connection.execute(
                """
                DELETE FROM rivalries
                WHERE owner_character_id = (SELECT id FROM characters WHERE name = ?)
                  AND rival_character_id = (SELECT id FROM characters WHERE name = ?)
                """,
                (owner_name, rival_name),
            )
            return cursor.rowcount > 0


def save_notification_channel(
    path: Path,
    owner_name: str,
    webhook_ciphertext: str,
    webhook_hash: str,
    timestamp: str,
) -> None:
    with closing(connect(path)) as connection:
        with connection:
            owner = connection.execute("SELECT id FROM characters WHERE name = ?", (owner_name,)).fetchone()
            if owner is None:
                raise ValueError("먼저 내 캐릭터를 검색해 주세요.")
            connection.execute(
                """
                INSERT INTO notification_channels(
                    owner_character_id, webhook_ciphertext, webhook_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_character_id) DO UPDATE SET
                    webhook_ciphertext = excluded.webhook_ciphertext,
                    webhook_hash = excluded.webhook_hash,
                    updated_at = excluded.updated_at
                """,
                (owner["id"], webhook_ciphertext, webhook_hash, timestamp, timestamp),
            )


def get_notification_channel(path: Path, owner_name: str) -> dict | None:
    with closing(connect(path)) as connection:
        row = connection.execute(
            """
            SELECT n.*, c.name AS owner_name
            FROM notification_channels n
            JOIN characters c ON c.id = n.owner_character_id
            WHERE c.name = ?
            """,
            (owner_name,),
        ).fetchone()
        return dict(row) if row else None


def list_notification_owners(path: Path) -> list[str]:
    with closing(connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT c.name
            FROM notification_channels n
            JOIN characters c ON c.id = n.owner_character_id
            ORDER BY c.id
            """
        ).fetchall()
        return [row["name"] for row in rows]


def get_rival_alert_state(path: Path, owner_name: str, rival_name: str) -> dict | None:
    with closing(connect(path)) as connection:
        row = connection.execute(
            """
            SELECT s.*
            FROM rival_alert_states s
            JOIN characters owner ON owner.id = s.owner_character_id
            JOIN characters rival ON rival.id = s.rival_character_id
            WHERE owner.name = ? AND rival.name = ?
            """,
            (owner_name, rival_name),
        ).fetchone()
        return dict(row) if row else None


def list_rival_alert_states(path: Path, owner_name: str) -> list[dict]:
    with closing(connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT rival.name AS rival_name, s.state, s.gap_percentage_points,
                   s.initialized_at, s.checked_at, s.notified_at
            FROM rival_alert_states s
            JOIN characters owner ON owner.id = s.owner_character_id
            JOIN characters rival ON rival.id = s.rival_character_id
            WHERE owner.name = ?
            ORDER BY rival.name
            """,
            (owner_name,),
        ).fetchall()
        return [dict(row) for row in rows]


def save_rival_alert_state(
    path: Path,
    *,
    owner_name: str,
    rival_name: str,
    state: str,
    gap: float,
    checked_at: str,
    initialize: bool = False,
    notified_at: str | None = None,
) -> None:
    with closing(connect(path)) as connection:
        with connection:
            ids = connection.execute(
                """
                SELECT
                    (SELECT id FROM characters WHERE name = ?) AS owner_id,
                    (SELECT id FROM characters WHERE name = ?) AS rival_id
                """,
                (owner_name, rival_name),
            ).fetchone()
            if ids["owner_id"] is None or ids["rival_id"] is None:
                raise ValueError("등록되지 않은 캐릭터입니다.")
            existing = connection.execute(
                "SELECT initialized_at, notified_at FROM rival_alert_states WHERE owner_character_id = ? AND rival_character_id = ?",
                (ids["owner_id"], ids["rival_id"]),
            ).fetchone()
            initialized_at = checked_at if initialize or existing is None else existing["initialized_at"]
            last_notified = notified_at if notified_at is not None else (existing["notified_at"] if existing else None)
            connection.execute(
                """
                INSERT INTO rival_alert_states(
                    owner_character_id, rival_character_id, state,
                    gap_percentage_points, initialized_at, checked_at, notified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_character_id, rival_character_id) DO UPDATE SET
                    state = excluded.state,
                    gap_percentage_points = excluded.gap_percentage_points,
                    initialized_at = excluded.initialized_at,
                    checked_at = excluded.checked_at,
                    notified_at = excluded.notified_at
                """,
                (
                    ids["owner_id"], ids["rival_id"], state, gap,
                    initialized_at, checked_at, last_notified,
                ),
            )


def replace_owner_alert_baselines(
    path: Path, owner_name: str, states: list[dict], timestamp: str
) -> None:
    with closing(connect(path)) as connection:
        with connection:
            owner = connection.execute("SELECT id FROM characters WHERE name = ?", (owner_name,)).fetchone()
            if owner is None:
                raise ValueError("등록되지 않은 내 캐릭터입니다.")
            connection.execute(
                "DELETE FROM rival_alert_states WHERE owner_character_id = ?", (owner["id"],)
            )
    for item in states:
        save_rival_alert_state(
            path,
            owner_name=owner_name,
            rival_name=item["rival_name"],
            state=item["state"],
            gap=item["gap"],
            checked_at=timestamp,
            initialize=True,
        )


def log_rival_notification(
    path: Path,
    *,
    owner_name: str,
    rival_name: str,
    previous_state: str,
    new_state: str,
    gap: float,
    message: str,
    delivery_status: str,
    error_message: str | None,
    created_at: str,
) -> None:
    with closing(connect(path)) as connection:
        with connection:
            ids = connection.execute(
                """
                SELECT
                    (SELECT id FROM characters WHERE name = ?) AS owner_id,
                    (SELECT id FROM characters WHERE name = ?) AS rival_id
                """,
                (owner_name, rival_name),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO rival_notification_log(
                    owner_character_id, rival_character_id, previous_state, new_state,
                    gap_percentage_points, message, delivery_status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ids["owner_id"], ids["rival_id"], previous_state, new_state,
                    gap, message, delivery_status, error_message, created_at,
                ),
            )


def log_webhook_event(
    path: Path,
    *,
    owner_name: str,
    rival_name: str | None,
    event_type: str,
    message: str,
    delivery_status: str,
    error_message: str | None,
    created_at: str,
) -> None:
    with closing(connect(path)) as connection:
        with connection:
            owner = connection.execute("SELECT id FROM characters WHERE name = ?", (owner_name,)).fetchone()
            if owner is None:
                raise ValueError("등록되지 않은 내 캐릭터입니다.")
            connection.execute(
                """
                INSERT INTO webhook_event_log(
                    owner_character_id, rival_name, event_type, message,
                    delivery_status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner["id"], rival_name, event_type, message,
                    delivery_status, error_message, created_at,
                ),
            )


def get_alert_state(path: Path) -> dict | None:
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT * FROM alert_state WHERE id = 1").fetchone()
        return dict(row) if row else None


def save_alert_state(
    path: Path,
    *,
    own_name: str,
    rival_name: str,
    state: str,
    gap: float,
    webhook_hash: str,
    checked_at: str,
    initialize: bool = False,
    notified_at: str | None = None,
) -> None:
    with closing(connect(path)) as connection:
        with connection:
            existing = connection.execute("SELECT initialized_at FROM alert_state WHERE id = 1").fetchone()
            initialized_at = checked_at if initialize or existing is None else existing["initialized_at"]
            connection.execute(
                """
                INSERT INTO alert_state(
                    id, own_character_name, rival_character_name, state,
                    gap_percentage_points, webhook_hash, initialized_at,
                    checked_at, notified_at
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    own_character_name = excluded.own_character_name,
                    rival_character_name = excluded.rival_character_name,
                    state = excluded.state,
                    gap_percentage_points = excluded.gap_percentage_points,
                    webhook_hash = excluded.webhook_hash,
                    initialized_at = excluded.initialized_at,
                    checked_at = excluded.checked_at,
                    notified_at = COALESCE(excluded.notified_at, alert_state.notified_at)
                """,
                (
                    own_name, rival_name, state, gap, webhook_hash,
                    initialized_at, checked_at, notified_at,
                ),
            )


def log_notification(
    path: Path,
    *,
    previous_state: str,
    new_state: str,
    gap: float,
    message: str,
    delivery_status: str,
    error_message: str | None,
    created_at: str,
) -> None:
    with closing(connect(path)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO notification_log(
                    previous_state, new_state, gap_percentage_points,
                    message, delivery_status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    previous_state, new_state, gap, message,
                    delivery_status, error_message, created_at,
                ),
            )

