from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, Action
from .replay_audit import load_downloaded_replay


PLAYER_DECISION_DATASET_SCHEMA_VERSION = 1

# Only explicit player/account identity fields are forbidden. Generic fields such
# as card names and reveal tokens are legitimate public game information.
_SENSITIVE_KEYS = frozenset({
    "account",
    "account_id",
    "account_name",
    "chat",
    "chat_history",
    "email",
    "ip",
    "ip_address",
    "messages",
    "nickname",
    "player_name",
    "player_names",
    "room_chat_history",
    "session_id",
    "sid",
    "username",
})
_OPAQUE_GROUP_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_DECISION_ACTION_TYPES = frozenset({
    "confirm_opening_reveal",
    "draft_pick",
    "draft_reroll",
    "end_turn",
    "play_card",
    "resolve_choice",
    "response",
    "select_opening_event",
    "submit_event_sub_choice",
    "use_equipment",
    "v2_ui_response",
})


def extract_strict_player_decisions(path: str | Path) -> dict[str, Any]:
    """Extract anonymized decision-time samples or explain every rejection.

    Old replays remain useful for aggregate statistics, but this function never
    reconstructs missing observations or legal actions from post-action state.
    """

    header, replay = load_downloaded_replay(path)
    meta = replay.get("meta") if isinstance(replay.get("meta"), dict) else {}
    actions = replay.get("actions") if isinstance(replay.get("actions"), list) else []
    mode = str(meta.get("mode") or header.get("mode") or "")
    replay_group = _replay_group(header, replay)
    replay_reasons: list[str] = []
    if mode != "1v1":
        replay_reasons.append("unsupported_mode")
    if bool(meta.get("truncated") or replay.get("truncated")):
        replay_reasons.append("truncated_replay")

    winner = _as_int(meta.get("winner_index"), -1)
    if winner not in (-1, 0, 1):
        replay_reasons.append("invalid_winner")

    samples: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    decision_actions = 0
    snapshot_actions = 0
    rulesets: set[str] = set()
    grouping = Counter()

    for action_index, raw_action in enumerate(actions):
        if not isinstance(raw_action, dict):
            continue
        actor = _as_int(raw_action.get("actor"), -1)
        action_type = str(raw_action.get("type") or "")
        is_decision = actor in (0, 1) and action_type in _DECISION_ACTION_TYPES
        snapshot = raw_action.get("ai_decision")
        if is_decision:
            decision_actions += 1
        if not isinstance(snapshot, dict):
            if is_decision:
                rejection_counts["missing_ai_decision"] += 1
            continue
        snapshot_actions += 1
        if replay_reasons:
            rejection_counts[replay_reasons[0]] += 1
            continue
        sample, reason = _validate_snapshot(
            snapshot,
            actor=actor,
            action_index=action_index,
            action_type=action_type,
            replay_group=replay_group,
            winner=winner,
        )
        if reason:
            rejection_counts[reason] += 1
            continue
        rulesets.add(str(sample["ruleset_fingerprint"]))
        grouping[str(sample["player_group_quality"])] += 1
        samples.append(sample)

    if len(rulesets) > 1:
        rejection_counts["mixed_rulesets_in_replay"] += len(samples)
        samples = []
    if decision_actions and snapshot_actions < decision_actions:
        replay_reasons.append("incomplete_decision_coverage")

    return {
        "source": str(Path(path).resolve()),
        "replay_group": replay_group,
        "mode": mode,
        "decision_actions": decision_actions,
        "snapshot_actions": snapshot_actions,
        "accepted_samples": len(samples),
        "rejected_samples": int(sum(rejection_counts.values())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "limitations": sorted(set(replay_reasons)),
        "ruleset_fingerprints": sorted(rulesets),
        "player_group_quality": dict(sorted(grouping.items())),
        "strict_ready": bool(
            decision_actions > 0
            and len(samples) == decision_actions
            and not replay_reasons
            and len(rulesets) == 1
        ),
        "samples": samples,
    }


def import_player_replays(
    paths: Iterable[str | Path],
    *,
    output: str | Path,
) -> dict[str, Any]:
    reports = [extract_strict_player_decisions(path) for path in paths]
    samples = [sample for report in reports for sample in report.pop("samples")]
    rulesets = sorted({str(sample["ruleset_fingerprint"]) for sample in samples})
    if len(rulesets) > 1:
        raise ValueError("player dataset cannot mix game rulesets")
    _write_jsonl_atomic(output, samples)
    rejection_counts: Counter[str] = Counter()
    for report in reports:
        rejection_counts.update(report.get("rejection_counts") or {})
    return {
        "schema_version": PLAYER_DECISION_DATASET_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "output": str(Path(output).resolve()),
        "replays": len(reports),
        "strict_ready_replays": sum(bool(report["strict_ready"]) for report in reports),
        "samples": len(samples),
        "ruleset_fingerprints": rulesets,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "reports": reports,
    }


def _validate_snapshot(
    snapshot: dict[str, Any],
    *,
    actor: int,
    action_index: int,
    action_type: str,
    replay_group: str,
    winner: int,
) -> tuple[dict[str, Any] | None, str | None]:
    observation = snapshot.get("observation")
    legal_raw = snapshot.get("legal_actions")
    selected_raw = snapshot.get("selected_action")
    if not isinstance(observation, dict):
        return None, "invalid_observation"
    if int(observation.get("schema_version", -1)) != OBSERVATION_SCHEMA_VERSION:
        return None, "observation_schema_mismatch"
    sensitive = _find_sensitive_fields(observation)
    if sensitive:
        return None, "sensitive_observation_fields"
    if not isinstance(legal_raw, list) or not legal_raw:
        return None, "missing_legal_actions"
    if not isinstance(selected_raw, dict):
        return None, "missing_selected_action"
    try:
        legal = [Action.from_dict(value) for value in legal_raw]
        selected = Action.from_dict(selected_raw)
    except (TypeError, ValueError):
        return None, "invalid_action_schema"
    legal_keys = [action.key for action in legal]
    if len(set(legal_keys)) != len(legal_keys):
        return None, "duplicate_legal_actions"
    if selected.key not in set(legal_keys):
        return None, "selected_action_not_legal"
    if actor not in (0, 1):
        return None, "invalid_actor"
    observed_seat = _as_int(observation.get("seat"), actor)
    decision_player = _as_int(observation.get("decision_player"), actor)
    if observed_seat != actor or decision_player != actor:
        return None, "actor_observation_mismatch"
    ruleset = str((observation.get("loadout") or {}).get("ruleset_fingerprint") or "")
    if not ruleset:
        return None, "missing_ruleset_fingerprint"

    supplied_group = snapshot.get("actor_group_hash")
    if isinstance(supplied_group, str) and _OPAQUE_GROUP_RE.fullmatch(supplied_group):
        player_group = supplied_group
        group_quality = "cross_replay"
    else:
        player_group = hashlib.sha256(
            f"{replay_group}:seat:{actor}".encode("utf-8")
        ).hexdigest()[:24]
        group_quality = "replay_only"
    outcome = 0.0 if winner < 0 else (1.0 if winner == actor else -1.0)
    return {
        "schema_version": PLAYER_DECISION_DATASET_SCHEMA_VERSION,
        "replay_group": replay_group,
        "player_group": player_group,
        "player_group_quality": group_quality,
        "ruleset_fingerprint": ruleset,
        "action_index": int(action_index),
        "source_action_type": action_type,
        "actor": actor,
        "outcome": outcome,
        "observation": observation,
        "legal_actions": [action.to_dict() for action in legal],
        "selected_action": selected.to_dict(),
    }, None


def _find_sensitive_fields(value: Any, path: str = "observation") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            child_path = f"{path}.{key}"
            if normalized in _SENSITIVE_KEYS:
                found.append(child_path)
            else:
                found.extend(_find_sensitive_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_sensitive_fields(child, f"{path}[{index}]"))
    return found


def _replay_group(header: dict[str, Any], replay: dict[str, Any]) -> str:
    meta = replay.get("meta") if isinstance(replay.get("meta"), dict) else {}
    material = {
        "replay_id": header.get("replay_id"),
        "created_at": header.get("created_at") or meta.get("created_at"),
        "mode": meta.get("mode") or header.get("mode"),
        "actions": len(replay.get("actions") or []),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"gtn-player-data:{encoded}".encode("utf-8")).hexdigest()[:24]


def _write_jsonl_atomic(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".gz" if target.suffix.lower() == ".gz" else ".jsonl"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        opener = gzip.open if target.suffix.lower() == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strictly validate and anonymize decision-time GTN player data"
    )
    parser.add_argument("replay", nargs="+", help="Downloaded .gtnreplay files")
    parser.add_argument("--output", required=True, help="Output .jsonl or .jsonl.gz")
    parser.add_argument("--report", help="Optional JSON quality report")
    args = parser.parse_args(argv)
    report = import_player_replays(args.replay, output=args.output)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["samples"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
