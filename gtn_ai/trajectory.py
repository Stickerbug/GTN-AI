from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .protocol import Action


TRAJECTORY_SCHEMA_VERSION = 4


@dataclass
class DecisionRecord:
    step: int
    player: int
    observation: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    action: dict[str, Any]
    forced_fallback: bool = False
    behavior_log_prob: float | None = None
    behavior_value: float | None = None
    behavior_entropy: float | None = None
    behavior_temperature: float | None = None
    teacher: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.teacher is None:
            payload.pop("teacher", None)
        return payload


@dataclass
class Episode:
    seed: int
    official_mods: list[str]
    loadout_fingerprint: str
    ruleset_fingerprint: str
    policies: list[str]
    winner: int
    terminated: bool
    truncated: bool
    steps: int
    rounds: int
    opening_events: list[Any] = field(default_factory=list)
    drafted_decks: list[list[str]] = field(default_factory=list)
    first_player: int = -1
    pregame_steps: int = 0
    combat_steps: int = 0
    loop_recoveries: int = 0
    forced_fallback_actions: int = 0
    policy_fingerprints: list[str] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            **asdict(self),
        }
        payload["decisions"] = [decision.to_dict() for decision in self.decisions]
        return payload


class JsonlTrajectoryWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, episode: Episode) -> None:
        opener = gzip.open if self.path.suffix.lower() == ".gz" else open
        with opener(self.path, "at", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def action_dicts(actions: Iterable[Action]) -> list[dict[str, Any]]:
    return [action.to_dict() for action in actions]
