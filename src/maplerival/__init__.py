"""Maple Rival application package."""

from .main import app

__all__ = ["app"]


def main() -> None:
    import uvicorn

    uvicorn.run("maplerival.main:app", host="127.0.0.1", port=8000, reload=True)
