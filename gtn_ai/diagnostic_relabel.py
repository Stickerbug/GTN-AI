from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence

from .diagnostic_data import (
    DiagnosticBundle,
    _as_int,
    _discover_sources,
    _read_bundle,
    _winner_from_manifest,
)
from .environment import Garden1v1Env
from .game_imports import configure_game_imports
from .policies import policy_from_name
from .progress import ProgressReporter
from .protocol import Action
from .trajectory import DecisionRecord, Episode


MAX_COMPRESSED_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024


def default_relabel_policy() -> str:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "structured-v2-search-dagger-v2.epoch-06.pt"
    )
    return (
        f"unsafe-rollout-cpu:{checkpoint};candidates=5;rollouts=4;"
        "max-rollouts=12;confidence=.05;batch=2;horizon=6;"
        "belief=true;annotate=true;exploration=0"
    )


def default_game_root() -> Path:
    return Path(__file__).resolve().parents[2] / "Python联机版"


def relabel_diagnostic_sessions(
    paths: Iterable[str | Path],
    *,
    output: str | Path,
    game_root: str | Path | None = None,
    policy_name: str | None = None,
    actor_kinds: Iterable[str] = ("human", "ai"),
    include_unfinished: bool = False,
    only_marked: bool = False,
    only_candidates: bool = False,
    marker_lookback: int = 2,
    marker_lookahead: int = 0,
    uncertainty_margin: float = 0.05,
    anchor_rate: float = 0.1,
    max_decisions: int = 0,
    seed: int = 0,
    trust_private_snapshots: bool = False,
    overwrite: bool = False,
    progress_interval: float = 10.0,
    show_progress: bool = True,
    policy: Any | None = None,
    policy_factory: Callable[..., Any] = policy_from_name,
) -> dict[str, Any]:
    """Re-evaluate trusted diagnostic snapshots with an offline teacher.

    The resulting JSONL uses the normal trajectory schema, so it can be passed
    directly to ``build_recorded_teacher_cache``.  The recorded browser action
    remains the behavior action; the search result is stored in ``teacher``.
    """

    if not trust_private_snapshots:
        raise ValueError(
            "private snapshots are executable Pickle data; pass "
            "trust_private_snapshots=True only for trusted local diagnostics"
        )
    if only_marked and only_candidates:
        raise ValueError("only_marked and only_candidates are mutually exclusive")
    allowed_kinds = {str(value).strip() for value in actor_kinds if str(value).strip()}
    invalid_kinds = allowed_kinds - {"human", "ai", "system"}
    if invalid_kinds or not allowed_kinds:
        raise ValueError(f"invalid actor kinds: {sorted(invalid_kinds)}")
    lookback = max(0, int(marker_lookback))
    lookahead = max(0, int(marker_lookahead))
    uncertainty_threshold = max(0.0, float(uncertainty_margin))
    stable_anchor_rate = max(0.0, min(1.0, float(anchor_rate)))

    target = Path(output).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_game_root = Path(game_root or default_game_root()).resolve()
    configure_game_imports(resolved_game_root)

    sources = _deduplicated_sources(_discover_sources(paths))
    bundles: list[tuple[Path, DiagnosticBundle]] = []
    rejections: Counter[str] = Counter()
    seen_sessions: set[str] = set()
    for source in sources:
        try:
            bundle = _read_bundle(source)
        except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            rejections[f"invalid_bundle:{type(exc).__name__}"] += 1
            continue
        session_id = str(bundle.manifest.get("session_id") or "").strip()
        dedupe_key = session_id or str(source).casefold()
        if dedupe_key in seen_sessions:
            rejections["duplicate_session"] += len(bundle.decisions)
            continue
        seen_sessions.add(dedupe_key)
        bundles.append((source, bundle))

    total = sum(len(bundle.decisions) for _, bundle in bundles)
    progress = ProgressReporter(
        "relabel-diagnostics",
        total=total,
        interval=progress_interval,
        enabled=show_progress,
    )
    active_policy = policy
    chosen_policy_name = str(policy_name or default_relabel_policy())
    rulesets: set[str] = set()
    reports: list[dict[str, Any]] = []
    processed = 0
    labeled = 0
    episodes = 0
    decision_limit = max(0, int(max_decisions))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".gz" if target.suffix.lower() == ".gz" else ".jsonl",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    opener = gzip.open if target.suffix.lower() == ".gz" else open
    try:
        with opener(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for source, bundle in bundles:
                if decision_limit and labeled >= decision_limit:
                    break
                status = str(bundle.manifest.get("status") or "")
                if status != "finished" and not include_unfinished:
                    count = len(bundle.decisions)
                    rejections["unfinished_session"] += count
                    processed += count
                    progress.update(processed, labeled=labeled)
                    continue

                marker_windows = _marker_windows(
                    bundle.markers,
                    lookback=lookback,
                    lookahead=lookahead,
                )
                marked_ids = {
                    _as_int(marker.get("decision_id"), -1)
                    for marker in bundle.markers
                    if isinstance(marker, dict)
                }
                session_decisions: list[dict[str, Any]] = []
                session_rejections: Counter[str] = Counter()
                session_id = str(bundle.manifest.get("session_id") or source.stem)
                with _SnapshotReader(source) as snapshots:
                    for decision in bundle.decisions:
                        if decision_limit and labeled >= decision_limit:
                            break
                        processed += 1
                        decision_id = _as_int(decision.get("decision_id"), -1)
                        actor_kind = str(decision.get("actor_kind") or "")
                        if actor_kind not in allowed_kinds:
                            reason = "actor_kind_filtered"
                            session_rejections[reason] += 1
                            rejections[reason] += 1
                            progress.update(processed, labeled=labeled)
                            continue
                        marker_context = marker_windows.get(decision_id)
                        if only_marked and marker_context is None:
                            reason = "unmarked_decision"
                            session_rejections[reason] += 1
                            rejections[reason] += 1
                            progress.update(processed, labeled=labeled)
                            continue
                        candidate_reasons = _automatic_candidate_reasons(
                            decision,
                            session_id=session_id,
                            marker_context=marker_context,
                            uncertainty_margin=uncertainty_threshold,
                            anchor_rate=stable_anchor_rate,
                        )
                        if only_candidates and not candidate_reasons:
                            reason = "non_candidate_decision"
                            session_rejections[reason] += 1
                            rejections[reason] += 1
                            progress.update(processed, labeled=labeled)
                            continue
                        snapshot_name = str(decision.get("snapshot") or "")
                        if not snapshot_name:
                            reason = "missing_snapshot"
                            session_rejections[reason] += 1
                            rejections[reason] += 1
                            progress.update(processed, labeled=labeled)
                            continue
                        try:
                            engine = snapshots.load_engine(snapshot_name)
                            env = _environment_for_decision(
                                engine,
                                decision,
                                game_root=resolved_game_root,
                                seed=_stable_seed(seed, session_id, decision_id),
                            )
                            observation, legal, recorded_action = _validate_snapshot_decision(
                                env,
                                decision,
                                manifest=bundle.manifest,
                            )
                            if active_policy is None:
                                active_policy = policy_factory(chosen_policy_name, seed=int(seed))
                            teacher, search = _teacher_label(
                                active_policy,
                                env,
                                observation,
                                legal,
                                _as_int(decision.get("actor"), -1),
                            )
                        except Exception as exc:
                            reason = _rejection_reason(exc)
                            session_rejections[reason] += 1
                            rejections[reason] += 1
                            progress.update(processed, labeled=labeled)
                            continue

                        record = DecisionRecord(
                            step=decision_id,
                            player=_as_int(decision.get("actor"), -1),
                            observation=observation,
                            legal_actions=[action.to_dict() for action in legal],
                            action=recorded_action.to_dict(),
                            teacher=teacher,
                        ).to_dict()
                        record["diagnostic"] = {
                            "decision_id": decision_id,
                            "actor_kind": actor_kind,
                            "marked": decision_id in marked_ids,
                            "marker_window": marker_context is not None,
                            "marker_distance": (
                                marker_context["distance"]
                                if marker_context is not None
                                else None
                            ),
                            "marker_ids": (
                                marker_context["marker_ids"]
                                if marker_context is not None
                                else []
                            ),
                            "marker_labels": (
                                marker_context["labels"]
                                if marker_context is not None
                                else []
                            ),
                            "candidate_reasons": candidate_reasons,
                            "recorded_action_matches_teacher": (
                                recorded_action.key == str(teacher.get("action_key") or "")
                            ),
                            "search_seconds": search.get("seconds"),
                            "search_margin": teacher.get("search_margin"),
                        }
                        session_decisions.append(record)
                        labeled += 1
                        progress.update(processed, labeled=labeled)

                if session_decisions:
                    episode = _diagnostic_episode(
                        bundle,
                        session_decisions,
                        policy_name=chosen_policy_name,
                    )
                    ruleset = str(episode.get("ruleset_fingerprint") or "")
                    if ruleset:
                        rulesets.add(ruleset)
                    handle.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
                    handle.write("\n")
                    episodes += 1
                reports.append({
                    "source": str(source),
                    "session": _opaque_session_id(session_id),
                    "status": status,
                    "decisions": len(bundle.decisions),
                    "labeled": len(session_decisions),
                    "rejection_counts": dict(sorted(session_rejections.items())),
                })

        if len(rulesets) > 1:
            raise ValueError(
                "diagnostic sessions contain multiple rulesets; relabel each ruleset separately"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
        progress.finish(processed, labeled=labeled)

    return {
        "output": str(target),
        "sources": len(sources),
        "sessions": len(reports),
        "episodes": episodes,
        "decisions_seen": processed,
        "decisions_labeled": labeled,
        "actor_kinds": sorted(allowed_kinds),
        "only_marked": bool(only_marked),
        "only_candidates": bool(only_candidates),
        "marker_lookback": lookback,
        "marker_lookahead": lookahead,
        "uncertainty_margin": uncertainty_threshold,
        "anchor_rate": stable_anchor_rate,
        "policy": chosen_policy_name,
        "ruleset_fingerprints": sorted(rulesets),
        "rejection_counts": dict(sorted(rejections.items())),
        "reports": reports,
    }


def _marker_windows(
    markers: Sequence[dict[str, Any]],
    *,
    lookback: int,
    lookahead: int,
) -> dict[int, dict[str, Any]]:
    """Map delayed UI markers to a small, auditable decision window.

    Positive distance means the decision happened before the marker's recorded
    decision. This is the usual case when a human notices a bad AI action one
    or two interactions later. No decision in the window is assumed wrong; the
    offline teacher still decides whether it is a correction or an anchor.
    """

    windows: dict[int, dict[str, Any]] = {}
    for marker in markers:
        if not isinstance(marker, dict):
            continue
        marker_id = _as_int(marker.get("decision_id"), -1)
        if marker_id < 0:
            continue
        label = str(marker.get("label") or "review")[:80]
        for decision_id in range(
            max(0, marker_id - max(0, int(lookback))),
            marker_id + max(0, int(lookahead)) + 1,
        ):
            distance = marker_id - decision_id
            entry = windows.setdefault(decision_id, {
                "distance": distance,
                "marker_ids": [],
                "labels": [],
            })
            if abs(distance) < abs(int(entry["distance"])):
                entry["distance"] = distance
            if marker_id not in entry["marker_ids"]:
                entry["marker_ids"].append(marker_id)
            if label not in entry["labels"]:
                entry["labels"].append(label)
    for entry in windows.values():
        entry["marker_ids"].sort()
        entry["labels"].sort()
    return windows


def _automatic_candidate_reasons(
    decision: dict[str, Any],
    *,
    session_id: str,
    marker_context: dict[str, Any] | None,
    uncertainty_margin: float,
    anchor_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if marker_context is not None:
        reasons.append("marker_window")
    metadata = decision.get("policy_metadata")
    if isinstance(metadata, dict):
        if metadata.get("changed") is True:
            reasons.append("live_search_changed_action")
        try:
            margin = float(metadata.get("score_margin"))
        except (TypeError, ValueError):
            margin = math.inf
        if math.isfinite(margin) and margin <= max(0.0, float(uncertainty_margin)):
            reasons.append("low_live_search_margin")
    decision_id = _as_int(decision.get("decision_id"), -1)
    if anchor_rate > 0 and _stable_fraction(session_id, decision_id) < min(1.0, anchor_rate):
        reasons.append("deterministic_anchor")
    return reasons


def _stable_fraction(session_id: str, decision_id: int) -> float:
    digest = hashlib.sha256(
        f"gtn-diagnostic-anchor:{session_id}:{int(decision_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class _SnapshotReader:
    def __init__(self, source: Path) -> None:
        self.source = source.resolve()
        self.archive: zipfile.ZipFile | None = None

    def __enter__(self) -> _SnapshotReader:
        if self.source.is_file():
            self.archive = zipfile.ZipFile(self.source)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.archive is not None:
            self.archive.close()
            self.archive = None

    def load_engine(self, snapshot_name: str):
        member = _safe_snapshot_member(snapshot_name)
        if self.source.is_dir():
            snapshot = (self.source / Path(*member.parts)).resolve()
            if self.source not in snapshot.parents:
                raise ValueError("snapshot path escapes its diagnostic session")
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
            if snapshot.stat().st_size > MAX_COMPRESSED_SNAPSHOT_BYTES:
                raise ValueError("compressed snapshot is too large")
            with gzip.open(snapshot, "rb") as handle:
                raw = handle.read(MAX_SNAPSHOT_BYTES + 1)
        else:
            if self.archive is None:
                raise RuntimeError("snapshot archive is not open")
            name = member.as_posix()
            info = self.archive.getinfo(name)
            if info.file_size > MAX_COMPRESSED_SNAPSHOT_BYTES:
                raise ValueError("compressed snapshot is too large")
            with self.archive.open(info) as compressed:
                with gzip.GzipFile(fileobj=io.BytesIO(compressed.read())) as handle:
                    raw = handle.read(MAX_SNAPSHOT_BYTES + 1)
        if len(raw) > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot is too large")
        return pickle.loads(raw)


def _safe_snapshot_member(value: str) -> PurePosixPath:
    normalized = str(value or "").replace("\\", "/")
    member = PurePosixPath(normalized)
    if (
        member.is_absolute()
        or not member.parts
        or member.parts[0] != "snapshots"
        or ".." in member.parts
        or member.suffixes[-2:] != [".pkl", ".gz"]
    ):
        raise ValueError("invalid diagnostic snapshot path")
    return member


def _environment_for_decision(
    engine,
    decision: dict[str, Any],
    *,
    game_root: Path,
    seed: int,
) -> Garden1v1Env:
    recorded_observation = decision.get("observation")
    if not isinstance(recorded_observation, dict):
        raise ValueError("invalid_observation")
    loadout = recorded_observation.get("loadout")
    enabled_mods = loadout.get("official_mods") if isinstance(loadout, dict) else None
    return Garden1v1Env.from_engine(
        engine,
        game_root=game_root,
        seed=seed,
        enabled_mods=enabled_mods,
        public_history=recorded_observation.get("public_history") or (),
    )


def _validate_snapshot_decision(
    env: Garden1v1Env,
    decision: dict[str, Any],
    *,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[Action], Action]:
    actor = _as_int(decision.get("actor"), -1)
    if actor not in (0, 1):
        raise ValueError("invalid_actor")
    if env.decision_player() != actor:
        raise ValueError("snapshot_actor_mismatch")
    observation = env.observe(actor)
    if _as_int(observation.get("decision_player"), -1) != actor:
        raise ValueError("snapshot_actor_mismatch")
    legal = env.legal_actions(actor)
    if not legal:
        raise ValueError("snapshot_has_no_legal_actions")
    try:
        recorded_legal = [
            Action.from_dict(value)
            for value in decision.get("legal_actions") or []
        ]
        recorded_action = Action.from_dict(decision.get("action") or {})
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_recorded_action") from exc
    current_keys = [action.key for action in legal]
    if current_keys != [action.key for action in recorded_legal]:
        raise ValueError("legal_actions_changed")
    if recorded_action.key not in set(current_keys):
        raise ValueError("recorded_action_not_legal")
    expected_ruleset = str(
        (manifest.get("metadata") or {}).get("ruleset_fingerprint") or ""
    )
    if expected_ruleset and env.ruleset_fingerprint != expected_ruleset:
        raise ValueError("ruleset_fingerprint_changed")
    return observation, legal, recorded_action


def _teacher_label(
    policy,
    env: Garden1v1Env,
    observation: dict[str, Any],
    legal: Sequence[Action],
    actor: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selector = getattr(policy, "select_action_with_env", None)
    if not callable(selector):
        raise ValueError("teacher_policy_requires_environment")
    selector(env, observation, legal, actor)
    teacher = getattr(policy, "last_teacher_metadata", None)
    search = getattr(policy, "last_search_metadata", None)
    if not isinstance(teacher, dict):
        raise ValueError("teacher_policy_did_not_produce_label")
    logits = [float(value) for value in teacher.get("logits") or ()]
    value = float(teacher.get("value"))
    action_key = str(teacher.get("action_key") or "")
    legal_keys = [action.key for action in legal]
    if action_key not in legal_keys:
        raise ValueError("teacher_action_not_legal")
    if len(logits) != len(legal) or not logits:
        raise ValueError("teacher_logits_mismatch")
    if not all(math.isfinite(item) for item in (*logits, value)):
        raise ValueError("teacher_output_not_finite")
    if legal_keys[max(range(len(logits)), key=logits.__getitem__)] != action_key:
        raise ValueError("teacher_argmax_mismatch")
    label = dict(teacher)
    label["value"] = max(-1.0, min(1.0, value))
    return label, dict(search or {})


def _diagnostic_episode(
    bundle: DiagnosticBundle,
    decisions: list[dict[str, Any]],
    *,
    policy_name: str,
) -> dict[str, Any]:
    first = decisions[0]["observation"]
    loadout = first.get("loadout") if isinstance(first.get("loadout"), dict) else {}
    manifest = bundle.manifest
    outcome = manifest.get("outcome") if isinstance(manifest.get("outcome"), dict) else {}
    winner = _winner_from_manifest(manifest)
    reason = str(outcome.get("reason") or "")
    terminated = (
        str(manifest.get("status") or "") == "finished"
        and (winner in (0, 1) or reason in {"game_over", "draw"})
    )
    episode = Episode(
        seed=_stable_seed(0, str(manifest.get("session_id") or bundle.source), 0),
        official_mods=list(loadout.get("official_mods") or []),
        loadout_fingerprint=str(loadout.get("fingerprint") or ""),
        ruleset_fingerprint=str(loadout.get("ruleset_fingerprint") or ""),
        policies=["diagnostic-recorded", "diagnostic-recorded"],
        winner=winner,
        terminated=terminated,
        truncated=not terminated,
        steps=len(decisions),
        rounds=max(_as_int(item["observation"].get("round"), 0) for item in decisions),
        opening_events=list(first.get("opening_events") or []),
        first_player=_as_int(first.get("first_player"), -1),
        combat_steps=len(decisions),
    ).to_dict()
    episode["decisions"] = decisions
    episode["source_kind"] = "diagnostic_offline_teacher"
    episode["diagnostic_session"] = _opaque_session_id(
        str(manifest.get("session_id") or bundle.source)
    )
    episode["relabel_policy"] = policy_name
    return episode


def _deduplicated_sources(sources: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        resolved = source.resolve()
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _stable_seed(base_seed: int, session_id: str, decision_id: int) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}:{session_id}:{int(decision_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _opaque_session_id(session_id: str) -> str:
    return hashlib.sha256(
        f"gtn-relabel:{session_id}".encode("utf-8")
    ).hexdigest()[:24]


def _rejection_reason(exc: Exception) -> str:
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return "snapshot_missing"
    if isinstance(exc, (pickle.UnpicklingError, EOFError, gzip.BadGzipFile)):
        return "snapshot_invalid"
    message = str(exc).strip()
    if isinstance(exc, ValueError) and message and " " not in message:
        return message[:80]
    return f"relabel_error:{type(exc).__name__}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Relabel trusted human-vs-AI diagnostics with offline belief search",
    )
    parser.add_argument("paths", nargs="+", help="Session directories, roots, or .gtnai.zip files")
    parser.add_argument("--output", required=True, help="Output trajectory .jsonl or .jsonl.gz")
    parser.add_argument("--game-root", default=str(default_game_root()))
    parser.add_argument("--policy", default=default_relabel_policy())
    parser.add_argument("--actors", default="human,ai")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only-marked", action="store_true")
    selection.add_argument(
        "--only-candidates",
        action="store_true",
        help="Relabel markers, live-search disagreements/uncertainty, and stable anchors",
    )
    parser.add_argument(
        "--marker-lookback",
        type=int,
        default=2,
        help="With --only-marked, also relabel this many decisions before each marker",
    )
    parser.add_argument(
        "--marker-lookahead",
        type=int,
        default=0,
        help="With --only-marked, also relabel this many decisions after each marker",
    )
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    parser.add_argument("--anchor-rate", type=float, default=0.1)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--trust-private-snapshots",
        action="store_true",
        help="Required acknowledgement that local Pickle snapshots are trusted",
    )
    args = parser.parse_args(argv)
    report = relabel_diagnostic_sessions(
        args.paths,
        output=args.output,
        game_root=args.game_root,
        policy_name=args.policy,
        actor_kinds=[value.strip() for value in args.actors.split(",") if value.strip()],
        include_unfinished=args.include_unfinished,
        only_marked=args.only_marked,
        only_candidates=args.only_candidates,
        marker_lookback=args.marker_lookback,
        marker_lookahead=args.marker_lookahead,
        uncertainty_margin=args.uncertainty_margin,
        anchor_rate=args.anchor_rate,
        max_decisions=args.max_decisions,
        seed=args.seed,
        trust_private_snapshots=args.trust_private_snapshots,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
        show_progress=not args.quiet,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
