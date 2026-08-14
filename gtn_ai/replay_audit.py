from __future__ import annotations

import argparse
import json
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


REPLAY_DOWNLOAD_MAGIC = b"GTNRPL1\n"
MAX_REPLAY_BYTES = 64 * 1024 * 1024


def load_downloaded_replay(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    raw = source.read_bytes()
    if len(raw) > MAX_REPLAY_BYTES:
        raise ValueError(f"replay exceeds the {MAX_REPLAY_BYTES}-byte audit limit")
    if not raw.startswith(REPLAY_DOWNLOAD_MAGIC):
        raise ValueError("not a GTN downloaded replay package")
    offset = len(REPLAY_DOWNLOAD_MAGIC)
    if len(raw) < offset + 4:
        raise ValueError("truncated replay header")
    header_size = struct.unpack(">I", raw[offset:offset + 4])[0]
    offset += 4
    if header_size <= 0 or offset + header_size > len(raw):
        raise ValueError("invalid replay header size")
    header = json.loads(raw[offset:offset + header_size].decode("utf-8"))
    if not isinstance(header, dict) or header.get("format") != "gtn-replay":
        raise ValueError("unsupported replay package header")
    if header.get("encoding") != "zlib-json":
        raise ValueError(f"unsupported replay encoding: {header.get('encoding')}")
    compressed = raw[offset + header_size:]
    try:
        inflater = zlib.decompressobj()
        decoded = inflater.decompress(compressed, MAX_REPLAY_BYTES + 1)
        if len(decoded) > MAX_REPLAY_BYTES or inflater.unconsumed_tail:
            raise ValueError("decoded replay exceeds the audit limit")
        decoded += inflater.flush(MAX_REPLAY_BYTES + 1 - len(decoded))
        if len(decoded) > MAX_REPLAY_BYTES or not inflater.eof:
            raise ValueError("decoded replay is oversized or truncated")
    except zlib.error as exc:
        raise ValueError("invalid compressed replay payload") from exc
    replay = json.loads(decoded.decode("utf-8"))
    if not isinstance(replay, dict):
        raise ValueError("replay payload must be an object")
    return header, replay


def audit_replay(path: str | Path) -> dict[str, Any]:
    header, replay = load_downloaded_replay(path)
    meta = replay.get("meta") if isinstance(replay.get("meta"), dict) else {}
    actions = replay.get("actions") if isinstance(replay.get("actions"), list) else []
    keyframes = replay.get("keyframes") if isinstance(replay.get("keyframes"), list) else []
    action_types = Counter(
        str(action.get("type") or "unknown")
        for action in actions
        if isinstance(action, dict)
    )
    state_actions = sum(
        1 for action in actions
        if isinstance(action, dict) and isinstance(action.get("state"), dict)
    )
    ai_snapshots = sum(
        1 for action in actions
        if isinstance(action, dict)
        and isinstance(action.get("ai_decision"), dict)
        and isinstance(action["ai_decision"].get("observation"), dict)
        and isinstance(action["ai_decision"].get("legal_actions"), list)
    )
    player_decisions = sum(
        action_types.get(kind, 0)
        for kind in (
            "play_card",
            "response",
            "resolve_choice",
            "v2_ui_response",
            "use_equipment",
            "end_turn",
        )
    )
    strict_ready = player_decisions > 0 and ai_snapshots >= player_decisions
    limitations = []
    if str(meta.get("mode") or header.get("mode") or "") != "1v1":
        limitations.append("only formal 1v1 is supported by the current AI environment")
    if bool(meta.get("truncated")):
        limitations.append("replay metadata marks the action stream as truncated")
    if ai_snapshots < player_decisions:
        limitations.append(
            "decision-time public observations and complete legal-action sets are missing"
        )
    if state_actions < len(actions):
        limitations.append("some action states were compacted or omitted")
    return {
        "path": str(Path(path).resolve()),
        "replay_id": header.get("replay_id"),
        "mode": meta.get("mode") or header.get("mode"),
        "players": meta.get("players") or header.get("players") or [],
        "winner_index": meta.get("winner_index"),
        "rules": replay.get("rules") or {},
        "actions": len(actions),
        "keyframes": len(keyframes),
        "action_types": dict(sorted(action_types.items())),
        "actions_with_state": state_actions,
        "decision_actions": player_decisions,
        "ai_decision_snapshots": ai_snapshots,
        "strict_behavior_cloning_ready": strict_ready,
        "usable_for": (
            ["strict_behavior_cloning", "outcome_and_action_statistics"]
            if strict_ready
            else ["outcome_and_action_statistics"]
        ),
        "limitations": limitations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit downloaded GTN replays for AI-training suitability"
    )
    parser.add_argument("replay", nargs="+", help="Downloaded .gtnreplay files")
    args = parser.parse_args(argv)
    reports = [audit_replay(path) for path in args.replay]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
