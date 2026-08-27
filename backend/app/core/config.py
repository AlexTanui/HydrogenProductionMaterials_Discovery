from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    app_name: str = "Hydrogen Materials Discovery API"
    database_url: str = f"sqlite:///{BACKEND_DIR}/app.db"
    checkpoint_path: str = str(ROOT_DIR / "experiments" / "checkpoints" / "baseline.pt")
    cors_origins: list[str] = ["http://localhost:5173"]

    class Config:
        env_file = ".env"


settings = Settings()
