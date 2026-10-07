"""Atomic JSON checkpoint storage for restartable closed-loop campaigns."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "autopf.closed_loop.campaign.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonStateStore:
    """Persist campaign state after every externally visible transition."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).resolve()

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported campaign schema in {self.path}: {payload.get('schema')!r}")
        return payload

    def initialize(self, campaign_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        existing = self.load()
        if existing is not None:
            if existing.get("campaign_id") != campaign_id:
                raise ValueError(
                    f"Checkpoint belongs to campaign {existing.get('campaign_id')!r}, not {campaign_id!r}."
                )
            return existing
        now = utc_now()
        state = {
            "schema": SCHEMA,
            "campaign_id": campaign_id,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "next_iteration": 0,
            "strategy_state": {},
            "iterations": [],
            "metadata": dict(metadata or {}),
        }
        self.save(state)
        return state

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = utc_now()
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(state, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
