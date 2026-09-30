from collections import defaultdict
from pathlib import Path

from .database import dashboard_rows
from .experience import build_experience_scale


def build_dashboard(path: Path, limit: int = 15, names: list[str] | None = None) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    profiles: dict[str, dict] = {}

    # Read one hidden baseline day so the first visible day's gain is accurate.
    for row in dashboard_rows(path, limit + 1, names):
        name = row["name"]
        profiles[name] = {
            "name": name,
            "world": row["world_name"],
            "className": row["character_class"],
            "currentLevel": row["current_level"],
            "expRate": row["exp_rate"],
            "currentExp": row["current_exp"],
            "currentCheckedAt": row["current_checked_at"],
        }
        grouped[name].append(
            {
                "date": row["snapshot_date"],
                "level": row["level"],
                "exp": str(row["exp"]),
                "ranking": row["ranking"],
            }
        )

    characters = []
    for name, history in grouped.items():
        previous = None
        for point in history:
            current = int(point["exp"])
            point["dailyGain"] = str(current - previous) if previous is not None and current >= previous else None
            previous = current
        history = history[-limit:]
        latest = history[-1]
        characters.append(
            {
                **profiles[name],
                "level": profiles[name]["currentLevel"] or latest["level"],
                "expRate": profiles[name]["expRate"],
                "ranking": latest["ranking"],
                "latestExp": str(profiles[name]["currentExp"] or int(latest["exp"])),
                "currentCheckedAt": profiles[name]["currentCheckedAt"],
                "history": history,
            }
        )

    if names:
        order = {name: index for index, name in enumerate(names)}
        characters.sort(key=lambda item: order.get(item["name"], len(order)))
    else:
        characters.sort(key=lambda item: int(item["latestExp"]), reverse=True)
    gap = None
    leader = None
    if len(characters) >= 2:
        ranked = sorted(characters, key=lambda item: (item["level"], item["expRate"] or 0), reverse=True)
        leader = ranked[0]["name"]
        gap = str(abs(int(ranked[0]["latestExp"]) - int(ranked[1]["latestExp"])))

    experience_scale = build_experience_scale(characters) if characters else None
    return {
        "days": limit,
        "latestDate": max((point["date"] for history in grouped.values() for point in history), default=None),
        "currentCheckedAt": max(
            (profile["currentCheckedAt"] for profile in profiles.values() if profile["currentCheckedAt"]),
            default=None,
        ),
        "leader": leader,
        "gap": gap,
        "characters": characters,
        "experienceScale": experience_scale,
    }

