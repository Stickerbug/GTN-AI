from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .neural_training import iter_trajectory_episodes
from .policies import policy_from_name
from .protocol import Action


def default_base_policy() -> str:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "structured-v2-search-combined-m05-head-v1.pt"
    )
    return f"structured-cpu:{checkpoint}"


def curate_hard_examples(
    paths: Iterable[str | Path],
    *,
    train_output: str | Path,
    validation_output: str | Path,
    base_policy_name: str | None = None,
    base_policy: Any | None = None,
    policy_factory: Callable[..., Any] = policy_from_name,
    minimum_teacher_margin: float = 0.05,
    validation_fraction: float = 0.2,
    anchor_ratio: float = 0.25,
    max_examples: int = 0,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select high-value correction states and split them by whole session.

    Marker windows are always retained for review. Outside those windows, a
    decision is useful when a confident offline teacher disagrees with the
    frozen base policy. A small deterministic sample of confident agreements
    acts as an anchor against over-correction.
    """

    source_paths = [Path(path).resolve() for path in paths]
    if not source_paths:
        raise ValueError("at least one relabelled trajectory is required")
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    train_target = Path(train_output).resolve()
    validation_target = Path(validation_output).resolve()
    if train_target == validation_target:
        raise ValueError("train and validation outputs must be different")
    for target in (train_target, validation_target):
        if target.exists() and not overwrite:
            raise FileExistsError(target)

    minimum_margin = max(0.0, float(minimum_teacher_margin))
    fraction = max(0.0, min(0.5, float(validation_fraction)))
    anchors_per_hard = max(0.0, float(anchor_ratio))
    limit = max(0, int(max_examples))
    chosen_policy_name = str(base_policy_name or default_base_policy())
    policy = base_policy or policy_factory(chosen_policy_name, seed=int(seed))

    sessions: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    rulesets: set[str] = set()
    counters: Counter[str] = Counter()
    for source_index, episode in enumerate(iter_trajectory_episodes(source_paths)):
        counters["episodes_seen"] += 1
        session_id = _session_id(episode, source_paths, source_index)
        if session_id in seen_sessions:
            counters["duplicate_sessions"] += 1
            continue
        seen_sessions.add(session_id)
        ruleset = str(episode.get("ruleset_fingerprint") or "")
        if not ruleset:
            counters["episodes_without_ruleset"] += 1
            continue
        rulesets.add(ruleset)
        policy_ruleset = str(getattr(policy, "ruleset_fingerprint", "") or "")
        if policy_ruleset and policy_ruleset != ruleset:
            raise ValueError("base policy ruleset does not match relabelled trajectories")

        selected, decision_counts = _select_session_decisions(
            episode,
            policy=policy,
            minimum_margin=minimum_margin,
            anchor_ratio=anchors_per_hard,
            seed=seed,
            session_id=session_id,
        )
        counters.update(decision_counts)
        if not selected:
            counters["episodes_without_hard_examples"] += 1
            continue
        curated = dict(episode)
        curated["decisions"] = selected
        curated["steps"] = len(selected)
        curated["pregame_steps"] = sum(
            _is_pregame(item.get("observation")) for item in selected
        )
        curated["combat_steps"] = len(selected) - int(curated["pregame_steps"])
        curated["hard_example_source_session"] = session_id
        curated["hard_example_counts"] = dict(sorted(decision_counts.items()))
        sessions.append({
            "session_id": session_id,
            "stratum": tuple(sorted(str(value) for value in episode.get("official_mods") or ())),
            "episode": curated,
            "examples": len(selected),
        })

    if len(rulesets) > 1:
        raise ValueError("hard-example inputs contain multiple rulesets")
    if not sessions:
        raise ValueError("no hard examples passed curation")

    validation_ids, split_report = _stratified_session_split(
        sessions,
        validation_fraction=fraction,
        seed=seed,
    )
    train_sessions = [item for item in sessions if item["session_id"] not in validation_ids]
    validation_sessions = [item for item in sessions if item["session_id"] in validation_ids]
    if limit:
        train_sessions, validation_sessions = _apply_example_limit(
            train_sessions,
            validation_sessions,
            limit=limit,
            seed=seed,
        )
    if not train_sessions:
        raise ValueError("curation produced no training sessions")
    if fraction > 0.0 and not validation_sessions:
        raise ValueError("curation produced no validation sessions")

    for item in train_sessions:
        item["episode"]["hard_example_split"] = "train"
    for item in validation_sessions:
        item["episode"]["hard_example_split"] = "validation"
    _write_episodes_atomic(
        train_target,
        [item["episode"] for item in train_sessions],
    )
    _write_episodes_atomic(
        validation_target,
        [item["episode"] for item in validation_sessions],
    )

    train_examples = sum(item["examples"] for item in train_sessions)
    validation_examples = sum(item["examples"] for item in validation_sessions)
    return {
        "train_output": str(train_target),
        "validation_output": str(validation_target),
        "base_policy": chosen_policy_name,
        "ruleset_fingerprints": sorted(rulesets),
        "minimum_teacher_margin": minimum_margin,
        "anchor_ratio": anchors_per_hard,
        "sessions": len(sessions),
        "train_sessions": len(train_sessions),
        "validation_sessions": len(validation_sessions),
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "counters": dict(sorted(counters.items())),
        "split": split_report,
    }


def _select_session_decisions(
    episode: dict[str, Any],
    *,
    policy: Any,
    minimum_margin: float,
    anchor_ratio: float,
    seed: int,
    session_id: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    counters: Counter[str] = Counter()
    hard: list[tuple[int, dict[str, Any]]] = []
    anchors: list[tuple[int, dict[str, Any]]] = []
    for index, decision in enumerate(episode.get("decisions") or ()):
        counters["decisions_seen"] += 1
        teacher = decision.get("teacher")
        observation = decision.get("observation")
        diagnostic = decision.get("diagnostic")
        if not isinstance(teacher, dict) or not isinstance(observation, dict):
            counters["invalid_decisions"] += 1
            continue
        try:
            legal = [
                Action.from_dict(item)
                for item in decision.get("legal_actions") or ()
            ]
            recorded = Action.from_dict(decision.get("action") or {})
            teacher_logits = [float(value) for value in teacher.get("logits") or ()]
            if not legal or len(teacher_logits) != len(legal):
                raise ValueError("teacher logits do not align")
            teacher_index = max(range(len(legal)), key=teacher_logits.__getitem__)
            teacher_key = str(teacher.get("action_key") or "")
            if legal[teacher_index].key != teacher_key:
                raise ValueError("teacher action does not match argmax")
            base_scores, _ = policy.evaluate_actions(observation, legal)
            if len(base_scores) != len(legal):
                raise ValueError("base policy scores do not align")
            base_index = max(range(len(legal)), key=base_scores.__getitem__)
            margin = _top_two_margin(teacher_logits)
        except (KeyError, TypeError, ValueError):
            counters["invalid_decisions"] += 1
            continue

        marker_window = bool(
            isinstance(diagnostic, dict)
            and (diagnostic.get("marker_window") or diagnostic.get("marked"))
        )
        confident = margin >= minimum_margin
        base_disagrees = base_index != teacher_index
        recorded_disagrees = recorded.key != teacher_key
        reasons: list[str] = []
        if marker_window:
            reasons.append("marker_window")
        if confident and base_disagrees:
            reasons.append("base_teacher_disagreement")
        if confident and recorded_disagrees:
            reasons.append("recorded_teacher_disagreement")

        curated = dict(decision)
        curated_diagnostic = dict(diagnostic or {})
        curated_diagnostic.update({
            "teacher_margin": round(margin, 7),
            "base_action_key": legal[base_index].key,
            "base_matches_teacher": not base_disagrees,
            "recorded_matches_teacher": not recorded_disagrees,
        })
        curated["diagnostic"] = curated_diagnostic
        if marker_window or (confident and base_disagrees):
            curated_diagnostic["hard_example_role"] = "correction_or_review"
            curated_diagnostic["hard_example_reasons"] = reasons
            curated_diagnostic["hard_example_weight"] = _hard_example_weight(
                marker_window=marker_window,
                base_disagrees=base_disagrees,
                confident=confident,
            )
            hard.append((index, curated))
            counters["hard_examples"] += 1
            counters["marker_window_examples"] += int(marker_window)
            counters["base_teacher_disagreements"] += int(confident and base_disagrees)
        elif confident and not base_disagrees:
            curated_diagnostic["hard_example_role"] = "agreement_anchor"
            curated_diagnostic["hard_example_reasons"] = ["confident_base_agreement"]
            curated_diagnostic["hard_example_weight"] = 0.5
            anchors.append((index, curated))
            counters["anchor_candidates"] += 1
        else:
            counters["low_margin_non_marker"] += 1

    anchor_count = min(
        len(anchors),
        int(math.ceil(len(hard) * max(0.0, float(anchor_ratio)))),
    )
    anchors.sort(key=lambda item: _stable_order(seed, session_id, item[0], "anchor"))
    selected = hard + anchors[:anchor_count]
    selected.sort(key=lambda item: item[0])
    counters["anchors_selected"] += anchor_count
    counters["examples_selected"] += len(selected)
    return [item for _, item in selected], counters


def _hard_example_weight(
    *,
    marker_window: bool,
    base_disagrees: bool,
    confident: bool,
) -> float:
    if marker_window and base_disagrees and confident:
        return 2.0
    if base_disagrees and confident:
        return 1.5
    return 1.0


def _stratified_session_split(
    sessions: Sequence[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], dict[str, Any]]:
    if validation_fraction <= 0.0:
        return set(), {
            "validation_fraction": 0.0,
            "strata": {},
            "single_session_strata": [],
        }
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in sessions:
        strata[item["stratum"]].append(item)
    validation: set[str] = set()
    report: dict[str, Any] = {}
    singletons: list[str] = []
    for stratum, items in sorted(strata.items()):
        ordered = sorted(
            items,
            key=lambda item: _stable_order(seed, item["session_id"], 0, "split"),
        )
        if len(ordered) < 2:
            validation_count = 0
            singletons.append(_stratum_name(stratum))
        else:
            validation_count = max(1, int(round(len(ordered) * validation_fraction)))
            validation_count = min(len(ordered) - 1, validation_count)
        chosen = ordered[:validation_count]
        validation.update(item["session_id"] for item in chosen)
        report[_stratum_name(stratum)] = {
            "sessions": len(ordered),
            "validation_sessions": validation_count,
            "examples": sum(item["examples"] for item in ordered),
            "validation_examples": sum(item["examples"] for item in chosen),
        }
    if not validation and len(sessions) >= 2:
        ordered = sorted(
            sessions,
            key=lambda item: _stable_order(seed, item["session_id"], 0, "fallback"),
        )
        validation.add(ordered[0]["session_id"])
        report[_stratum_name(ordered[0]["stratum"])]["validation_sessions"] += 1
        report[_stratum_name(ordered[0]["stratum"])]["validation_examples"] += ordered[0]["examples"]
    return validation, {
        "validation_fraction": validation_fraction,
        "strata": report,
        "single_session_strata": singletons,
    }


def _apply_example_limit(
    train_sessions: list[dict[str, Any]],
    validation_sessions: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total = sum(item["examples"] for item in (*train_sessions, *validation_sessions))
    if total <= limit:
        return train_sessions, validation_sessions
    validation_budget = min(
        sum(item["examples"] for item in validation_sessions),
        max(1, int(round(limit * 0.2))),
    )
    train_budget = max(1, limit - validation_budget)
    return (
        _limit_session_examples(train_sessions, train_budget, seed=seed, label="train"),
        _limit_session_examples(
            validation_sessions, validation_budget, seed=seed, label="validation"
        ),
    )


def _limit_session_examples(
    sessions: list[dict[str, Any]],
    budget: int,
    *,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    remaining = max(0, int(budget))
    for item in sorted(
        sessions,
        key=lambda value: _stable_order(seed, value["session_id"], 0, label),
    ):
        if remaining <= 0:
            break
        copied = dict(item)
        episode = dict(item["episode"])
        decisions = list(episode.get("decisions") or ())
        decisions.sort(key=lambda decision: (
            -float((decision.get("diagnostic") or {}).get("hard_example_weight", 1.0)),
            int(decision.get("step", 0)),
        ))
        decisions = decisions[:remaining]
        decisions.sort(key=lambda decision: int(decision.get("step", 0)))
        episode["decisions"] = decisions
        episode["steps"] = len(decisions)
        copied["episode"] = episode
        copied["examples"] = len(decisions)
        output.append(copied)
        remaining -= len(decisions)
    return output


def _top_two_margin(logits: Sequence[float]) -> float:
    ordered = sorted((float(value) for value in logits), reverse=True)
    if not ordered:
        raise ValueError("teacher logits are empty")
    if len(ordered) == 1:
        return math.inf
    return max(0.0, ordered[0] - ordered[1])


def _session_id(
    episode: dict[str, Any],
    source_paths: Sequence[Path],
    source_index: int,
) -> str:
    explicit = str(
        episode.get("diagnostic_session")
        or episode.get("hard_example_source_session")
        or ""
    ).strip()
    if explicit:
        return explicit
    identity = {
        "sources": [str(path) for path in source_paths],
        "source_index": int(source_index),
        "seed": episode.get("seed"),
        "loadout": episode.get("loadout_fingerprint"),
        "ruleset": episode.get("ruleset_fingerprint"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _stable_order(seed: int, session_id: str, index: int, purpose: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}:{purpose}:{session_id}:{int(index)}".encode("utf-8")
    ).hexdigest()


def _stratum_name(stratum: Sequence[str]) -> str:
    return "+".join(stratum) if stratum else "no-mods"


def _is_pregame(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    phase = observation.get("phase")
    return phase in {0, "pregame", "loadout", "draft"}


def _write_episodes_atomic(path: Path, episodes: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".gz" if path.suffix.lower() == ".gz" else ".jsonl",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Curate relabelled human sessions into leak-free hard-example splits",
    )
    parser.add_argument("input", nargs="+", help="Relabelled trajectory JSONL files")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--base-policy", default=default_base_policy())
    parser.add_argument("--minimum-teacher-margin", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--anchor-ratio", type=float, default=0.25)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = curate_hard_examples(
        args.input,
        train_output=args.train_output,
        validation_output=args.validation_output,
        base_policy_name=args.base_policy,
        minimum_teacher_margin=args.minimum_teacher_margin,
        validation_fraction=args.validation_fraction,
        anchor_ratio=args.anchor_ratio,
        max_examples=args.max_examples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
