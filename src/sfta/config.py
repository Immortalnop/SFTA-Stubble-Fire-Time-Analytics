"""Load config.yaml and .env; expose data paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict
    data_dir: Path
    outputs: Path

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def years(self) -> list[int]:
        a, b = self.raw["years"]
        return list(range(a, b + 1))

    def season_bounds(self, year: int, which: str = "season") -> tuple[date, date]:
        w = self.raw[which]
        return (date.fromisoformat(f"{year}-{w['start']}"), date.fromisoformat(f"{year}-{w['end']}"))

    def raw_dir(self, *parts: str) -> Path:
        p = self.data_dir.joinpath("raw", *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def processed(self, name: str) -> Path:
        p = self.data_dir / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p / name

    def figure(self, name: str) -> Path:
        p = self.outputs / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p / name

    def table(self, name: str) -> Path:
        p = self.outputs / "tables"
        p.mkdir(parents=True, exist_ok=True)
        return p / name


def load_config(path: str | Path | None = None) -> Config:
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    data_dir = Path(os.environ.get("SFTA_DATA_DIR", raw["data_dir"]))
    return Config(raw=raw, data_dir=data_dir, outputs=PROJECT_ROOT / "outputs")


def require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise SystemExit(f"{name} is not set. Add it to .env (see .env.example).")
    return val
