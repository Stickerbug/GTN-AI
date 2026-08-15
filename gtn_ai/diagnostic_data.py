from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .player_data import PLAYER_DECISION_DATASET_SCHEMA_VERSION, _find_sensitive_fields
from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, Action


@dataclass(frozen=True)
class DiagnosticBundle:
    source: str
    manifest: dict[str, Any]
    decisions: tuple[dict[str, Any], ...]
    markers: tuple[dict[str, Any], ...]


def import_diagnostic_sessions(
    paths: Iterable[str | Path],
    *,
    output: str | Path,
    actor_kinds: Iterable[str] = ("human",),
    include_unfinished: bool = False,
    include_marked: bool = False,
) -> dict[str, Any]:
    allowed_actor_kinds = {str(value) for value in actor_kinds}
    invalid_kinds = allowed_actor_kinds - {"human", "ai", "system"}
    if invalid_kinds or not allowed_actor_kinds:
        raise ValueError(f"invalid actor kinds: {sorted(invalid_kinds)}")

    sources = _discover_sources(paths)
    rows: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    reports = []
    seen_sessions: set[str] = set()
    rulesets: set[str] = set()
    for source in sources:
        try:
            bundle = _read_bundle(source)
        except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            rejections[f"invalid_bundle:{type(exc).__name__}"] += 1
            continue
        session_id = str(bundle.manifest.get("session_id") or "")
        dedupe_key = session_id or hashlib.sha256(bundle.source.encode("utf-8")).hexdigest()
        if dedupe_key in seen_sessions:
            rejections["duplicate_session"] += len(bundle.decisions)
            continue
        seen_sessions.add(dedupe_key)
        accepted, report = _bundle_rows(
            bundle,
            actor_kinds=allowed_actor_kinds,
            include_unfinished=bool(include_unfinished),
            include_marked=bool(include_marked),
        )
        rows.extend(accepted)
        rejections.update(report.pop("rejection_counts"))
        reports.append(report)
        rulesets.update(
            str(row.get("ruleset_fingerprint") or "")
            for row in accepted
            if row.get("ruleset_fingerprint")
        )
    if len(rulesets) > 1:
        raise ValueError(
            "diagnostic sessions contain multiple rulesets; import each ruleset separately"
        )
    _write_jsonl_atomic(output, rows)
    return {
        "schema_version": PLAYER_DECISION_DATASET_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "output": str(Path(output).resolve()),
        "sources": len(sources),
        "sessions": len(reports),
        "samples": len(rows),
        "actor_kinds": sorted(allowed_actor_kinds),
        "ruleset_fingerprints": sorted(rulesets),
        "rejection_counts": dict(sorted(rejections.items())),
        "reports": reports,
    }


def _bundle_rows(
    bundle: DiagnosticBundle,
    *,
    actor_kinds: set[str],
    include_unfinished: bool,
    include_marked: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = bundle.manifest
    status = str(manifest.get("status") or "")
    rejections: Counter[str] = Counter()
    if status != "finished" and not include_unfinished:
        rejections["unfinished_session"] += len(bundle.decisions)
        return [], {
            "source": bundle.source,
            "session": _opaque_session_id(manifest, bundle.source),
            "decisions": len(bundle.decisions),
            "accepted": 0,
            "rejection_counts": dict(rejections),
        }

    marked_ids = {
        _as_int(marker.get("decision_id"), -1)
        for marker in bundle.markers
        if isinstance(marker, dict)
    }
    winner = _winner_from_manifest(manifest)
    session_group = _opaque_session_id(manifest, bundle.source)
    rows = []
    for decision in bundle.decisions:
        if not isinstance(decision, dict):
            rejections["invalid_decision"] += 1
            continue
        decision_id = _as_int(decision.get("decision_id"), -1)
        actor_kind = str(decision.get("actor_kind") or "")
        if actor_kind not in actor_kinds:
            rejections["actor_kind_filtered"] += 1
            continue
        if decision_id in marked_ids and not include_marked:
            rejections["marked_decision"] += 1
            continue
        row, reason = _decision_row(
            decision,
            session_group=session_group,
            winner=winner,
        )
        if reason:
            rejections[reason] += 1
            continue
        rows.append(row)
    return rows, {
        "source": bundle.source,
        "session": session_group,
        "status": status,
        "winner": winner,
        "decisions": len(bundle.decisions),
        "accepted": len(rows),
        "rejection_counts": dict(rejections),
    }


def _decision_row(
    decision: dict[str, Any],
    *,
    session_group: str,
    winner: int,
) -> tuple[dict[str, Any] | None, str | None]:
    actor = _as_int(decision.get("actor"), -1)
    if actor not in (0, 1):
        return None, "invalid_actor"
    observation = decision.get("observation")
    legal_raw = decision.get("legal_actions")
    selected_raw = decision.get("action")
    if not isinstance(observation, dict):
        return None, "invalid_observation"
    if _as_int(observation.get("schema_version"), -1) != OBSERVATION_SCHEMA_VERSION:
        return None, "observation_schema_mismatch"
    if _find_sensitive_fields(observation):
        return None, "sensitive_observation_fields"
    if not isinstance(legal_raw, list) or not legal_raw:
        return None, "missing_legal_actions"
    try:
        legal = [Action.from_dict(value) for value in legal_raw]
        selected = Action.from_dict(selected_raw)
    except (TypeError, ValueError):
        return None, "invalid_action_schema"
    legal_keys = [action.key for action in legal]
    if len(legal_keys) != len(set(legal_keys)):
        return None, "duplicate_legal_actions"
    if selected.key not in set(legal_keys):
        return None, "selected_action_not_legal"
    if _as_int(observation.get("seat"), actor) != actor:
        return None, "actor_observation_mismatch"
    if _as_int(observation.get("decision_player"), actor) != actor:
        return None, "actor_observation_mismatch"
    ruleset = str((observation.get("loadout") or {}).get("ruleset_fingerprint") or "")
    if not ruleset:
        return None, "missing_ruleset_fingerprint"
    outcome = 0.0 if winner not in (0, 1) else (1.0 if winner == actor else -1.0)
    return {
        "schema_version": PLAYER_DECISION_DATASET_SCHEMA_VERSION,
        "replay_group": session_group,
        "player_group": hashlib.sha256(
            f"{session_group}:seat:{actor}".encode("utf-8")
        ).hexdigest()[:24],
        "player_group_quality": "diagnostic_session",
        "ruleset_fingerprint": ruleset,
        "action_index": _as_int(decision.get("decision_id"), 0),
        "source_action_type": selected.kind,
        "actor": actor,
        "actor_kind": str(decision.get("actor_kind") or ""),
        "outcome": outcome,
        "observation": observation,
        "legal_actions": [action.to_dict() for action in legal],
        "selected_action": selected.to_dict(),
    }, None


def _discover_sources(paths: Iterable[str | Path]) -> list[Path]:
    sources: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_file():
            sources.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        if (path / "manifest.json").is_file() and (path / "decisions.jsonl").is_file():
            sources.append(path)
            continue
        sources.extend(sorted(
            child.parent
            for child in path.rglob("manifest.json")
            if (child.parent / "decisions.jsonl").is_file()
            and "exports" not in child.parent.parts
        ))
        sources.extend(sorted(path.glob("exports/*.gtnai.zip")))
    return sources


def _read_bundle(path: Path) -> DiagnosticBundle:
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        decisions = tuple(_read_jsonl_text((path / "decisions.jsonl").read_text(encoding="utf-8")))
        marker_path = path / "markers.jsonl"
        markers = tuple(_read_jsonl_text(marker_path.read_text(encoding="utf-8"))) if marker_path.is_file() else ()
        return DiagnosticBundle(str(path), manifest, decisions, markers)
    if path.suffix.lower() != ".zip":
        raise ValueError(f"unsupported diagnostic source: {path}")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        decisions = tuple(_read_jsonl_text(archive.read("decisions.jsonl").decode("utf-8")))
        markers = (
            tuple(_read_jsonl_text(archive.read("markers.jsonl").decode("utf-8")))
            if "markers.jsonl" in archive.namelist()
            else ()
        )
    return DiagnosticBundle(str(path), manifest, decisions, markers)


def _read_jsonl_text(text: str) -> Iterator[dict[str, Any]]:
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def _winner_from_manifest(manifest: dict[str, Any]) -> int:
    outcome = manifest.get("outcome") if isinstance(manifest.get("outcome"), dict) else {}
    return _as_int(outcome.get("winner"), -1)


def _opaque_session_id(manifest: dict[str, Any], source: str) -> str:
    session_id = str(manifest.get("session_id") or source)
    return hashlib.sha256(f"gtn-diagnostic:{session_id}".encode("utf-8")).hexdigest()[:24]


def _write_jsonl_atomic(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".gz" if target.suffix.lower() == ".gz" else ".jsonl",
        dir=target.parent,
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
        description="Import local human-vs-AI diagnostics as strict player decisions",
    )
    parser.add_argument("paths", nargs="+", help="Session directories, roots, or .gtnai.zip files")
    parser.add_argument("--output", required=True, help="Output .jsonl or .jsonl.gz")
    parser.add_argument(
        "--actors",
        default="human",
        help="Comma-separated actor kinds: human, ai, system (default: human)",
    )
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--include-marked", action="store_true")
    args = parser.parse_args(argv)
    report = import_diagnostic_sessions(
        args.paths,
        output=args.output,
        actor_kinds=[value.strip() for value in args.actors.split(",") if value.strip()],
        include_unfinished=args.include_unfinished,
        include_marked=args.include_marked,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
