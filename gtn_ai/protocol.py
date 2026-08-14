from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


OBSERVATION_SCHEMA_VERSION = 3
ACTION_SCHEMA_VERSION = 3
SERVICE_API_VERSION = 1


@dataclass(frozen=True)
class Action:
    """One atomic decision accepted by the headless environment."""

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ACTION_SCHEMA_VERSION,
            "kind": self.kind,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        if not isinstance(data, dict):
            raise TypeError("action must be an object")
        version = int(data.get("schema_version", ACTION_SCHEMA_VERSION))
        if version != ACTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported action schema version: {version}")
        kind = str(data.get("kind") or "").strip()
        payload = data.get("payload") or {}
        if not kind or not isinstance(payload, dict):
            raise ValueError("action requires kind and object payload")
        return cls(kind=kind, payload=dict(payload))

    @property
    def key(self) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
