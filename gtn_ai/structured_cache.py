from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .neural_model import (
    collate_decisions,
    encode_decision,
    load_neural_checkpoint,
    neural_state_dict_fingerprint,
    require_torch,
    resolve_device,
    torch,
)
from .neural_training import iter_trajectory_episodes
from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, Action
from .progress import ProgressReporter
from .structured_features import (
    STRUCTURED_FEATURE_SCHEMA_VERSION,
    EntityToken,
    StructuredDecision,
    encode_structured_decision,
)
from .structured_model import StructuredModelConfig
from .trajectory import TRAJECTORY_SCHEMA_VERSION


DISTILLATION_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StructuredDistillationExample:
    decision: StructuredDecision
    teacher_logits: tuple[float, ...]
    teacher_value: float
    base_decision: StructuredDecision | None = None
    example_weight: float = 1.0


def _encode_paired_decisions(
    observation: dict[str, Any],
    legal: Sequence[Action],
    *,
    config: StructuredModelConfig,
    selected_index: int,
) -> tuple[StructuredDecision, StructuredDecision | None]:
    """Encode context-rich and legacy views of one decision when requested."""

    decision = encode_structured_decision(
        observation,
        legal,
        config=config.feature_config,
        selected_index=selected_index,
    )
    if not config.contextual_value_features:
        return decision, None
    base_config = replace(config, contextual_value_features=False)
    base_decision = encode_structured_decision(
        observation,
        legal,
        config=base_config.feature_config,
        selected_index=selected_index,
    )
    return decision, base_decision


def build_distillation_cache(
    paths: Sequence[str | Path],
    *,
    teacher_checkpoint: str | Path,
    output_dir: str | Path,
    config: StructuredModelConfig,
    deck_prior_path: str | Path | None = None,
    dynamic_deck_belief: bool = False,
    teacher_batch_size: int = 256,
    shard_size: int = 4096,
    device: str = "auto",
    max_decisions: int = 0,
    expected_decisions: int = 0,
    skip_recovered_episodes: bool = True,
    overwrite: bool = False,
    progress_interval: float = 10.0,
    show_progress: bool = True,
) -> dict[str, Any]:
    require_torch()
    source_paths = [Path(path).resolve() for path in paths]
    if not source_paths:
        raise ValueError("at least one trajectory file is required")
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = Path(output_dir).resolve()
    _prepare_cache_directory(target, overwrite=overwrite)
    resolved_device = resolve_device(device)
    teacher = load_neural_checkpoint(teacher_checkpoint, device=resolved_device)
    teacher_fingerprint = neural_state_dict_fingerprint(teacher["model"].state_dict())
    teacher["model"].eval()
    deck_prior = _load_deck_prior(deck_prior_path)
    deck_belief_schema_version = _deck_belief_schema_version(
        deck_prior, dynamic=dynamic_deck_belief
    )

    source_identity = [_file_identity(path) for path in source_paths]
    cache_identity = {
        "cache_schema_version": DISTILLATION_CACHE_SCHEMA_VERSION,
        "structured_feature_schema_version": STRUCTURED_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "teacher_fingerprint": teacher_fingerprint,
        "deck_prior_fingerprint": (
            deck_prior.fingerprint if deck_prior is not None else None
        ),
        "deck_belief_schema_version": deck_belief_schema_version,
        "structured_config": asdict(config),
        "paired_base_features": bool(config.contextual_value_features),
        "sources": source_identity,
        "skip_recovered_episodes": bool(skip_recovered_episodes),
        "max_decisions": max(0, int(max_decisions)),
    }
    cache_fingerprint = _json_fingerprint(cache_identity)

    pending_teacher = []
    pending_structured: list[
        tuple[StructuredDecision, StructuredDecision | None]
    ] = []
    shard_buffer: list[StructuredDistillationExample] = []
    shard_records: list[dict[str, Any]] = []
    rulesets: set[str] = set()
    counters = {
        "episodes_seen": 0,
        "episodes_used": 0,
        "skipped_recovered_episodes": 0,
        "skipped_truncated_episodes": 0,
        "decisions_seen": 0,
        "examples": 0,
        "invalid_decisions": 0,
    }
    progress = ProgressReporter(
        "build-cache",
        total=(
            max(0, int(max_decisions))
            or max(0, int(expected_decisions))
            or None
        ),
        interval=progress_interval,
        enabled=show_progress,
    )

    def flush_teacher_batch() -> None:
        if not pending_teacher:
            return
        batch = collate_decisions(pending_teacher, device=resolved_device)
        with torch.inference_mode():
            raw_scores, values = teacher["model"](batch)
        scores = raw_scores + batch["action_logit_biases"]
        offsets = batch["action_set_offsets"].detach().to("cpu").tolist()
        scores_cpu = scores.detach().to("cpu")
        values_cpu = values.detach().to("cpu").tolist()
        for row, (structured, base_structured) in enumerate(pending_structured):
            start, end = int(offsets[row]), int(offsets[row + 1])
            logits = scores_cpu[start:end]
            logits = logits - logits.max()
            shard_buffer.append(StructuredDistillationExample(
                decision=structured,
                teacher_logits=tuple(float(value) for value in logits.tolist()),
                teacher_value=float(values_cpu[row]),
                base_decision=base_structured,
            ))
        pending_teacher.clear()
        pending_structured.clear()
        while len(shard_buffer) >= max(1, int(shard_size)):
            chunk = shard_buffer[:shard_size]
            del shard_buffer[:shard_size]
            shard_records.append(_write_shard(target, len(shard_records), chunk, config=config))

    limit = max(0, int(max_decisions))
    stop = False
    for episode in iter_trajectory_episodes(source_paths):
        counters["episodes_seen"] += 1
        if bool(episode.get("truncated")) or not bool(episode.get("terminated")):
            counters["skipped_truncated_episodes"] += 1
            continue
        if skip_recovered_episodes and _as_int(episode.get("loop_recoveries"), 0) > 0:
            counters["skipped_recovered_episodes"] += 1
            continue
        ruleset = str(episode.get("ruleset_fingerprint") or "")
        if ruleset:
            rulesets.add(ruleset)
            if ruleset != teacher["ruleset_fingerprint"]:
                raise ValueError(
                    "teacher checkpoint ruleset does not match trajectory data"
                )
        counters["episodes_used"] += 1
        for decision in episode.get("decisions") or []:
            counters["decisions_seen"] += 1
            if decision.get("forced_fallback"):
                counters["invalid_decisions"] += 1
                continue
            try:
                observation = decision.get("observation")
                if not isinstance(observation, dict):
                    raise ValueError("missing observation")
                legal = [Action.from_dict(item) for item in decision.get("legal_actions") or []]
                selected = Action.from_dict(decision.get("action") or {})
                selected_index = next(
                    index for index, action in enumerate(legal)
                    if action.key == selected.key
                )
                structured_observation = _augment_with_deck_prior(
                    observation, deck_prior, dynamic=dynamic_deck_belief
                )
                structured, base_structured = _encode_paired_decisions(
                    structured_observation,
                    legal,
                    config=config,
                    selected_index=selected_index,
                )
                teacher_encoded = encode_decision(
                    observation,
                    legal,
                    config=teacher["config"],
                    selected_index=selected_index,
                )
            except (KeyError, StopIteration, TypeError, ValueError):
                counters["invalid_decisions"] += 1
                continue
            pending_structured.append((structured, base_structured))
            pending_teacher.append(teacher_encoded)
            counters["examples"] += 1
            if len(pending_teacher) >= max(1, int(teacher_batch_size)):
                flush_teacher_batch()
                progress.update(
                    counters["examples"],
                    episodes=counters["episodes_used"],
                    shards=len(shard_records),
                )
            if limit and counters["examples"] >= limit:
                stop = True
                break
        if stop:
            break
    flush_teacher_batch()
    if shard_buffer:
        shard_records.append(_write_shard(
            target, len(shard_records), shard_buffer, config=config
        ))
        shard_buffer.clear()
    if not shard_records:
        raise ValueError("trajectory data contains no usable decisions")
    if len(rulesets) != 1:
        raise ValueError("trajectory data must contain exactly one ruleset fingerprint")
    ruleset = next(iter(rulesets))
    manifest = {
        **cache_identity,
        "cache_fingerprint": cache_fingerprint,
        "ruleset_fingerprint": ruleset,
        "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
        "teacher_name": teacher["name"],
        "counters": counters,
        "shards": shard_records,
    }
    _write_json_atomic(target / "manifest.json", manifest)
    progress.finish(
        counters["examples"],
        episodes=counters["episodes_used"],
        shards=len(shard_records),
    )
    return manifest


def build_recorded_teacher_cache(
    paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    config: StructuredModelConfig,
    deck_prior_path: str | Path | None = None,
    dynamic_deck_belief: bool = False,
    shard_size: int = 4096,
    max_decisions: int = 0,
    expected_decisions: int = 0,
    skip_recovered_episodes: bool = True,
    min_teacher_margin: float = 0.0,
    only_teacher_disagreements: bool = False,
    preserve_winner_actions: bool = False,
    overwrite: bool = False,
    progress_interval: float = 10.0,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Encode decision-local offline teacher scores into distillation shards."""

    require_torch()
    source_paths = [Path(path).resolve() for path in paths]
    if not source_paths:
        raise ValueError("at least one trajectory file is required")
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = Path(output_dir).resolve()
    _prepare_cache_directory(target, overwrite=overwrite)
    source_identity = [_file_identity(path) for path in source_paths]
    minimum_margin = max(0.0, float(min_teacher_margin))
    deck_prior = _load_deck_prior(deck_prior_path)
    deck_belief_schema_version = _deck_belief_schema_version(
        deck_prior, dynamic=dynamic_deck_belief
    )
    cache_identity = {
        "cache_schema_version": DISTILLATION_CACHE_SCHEMA_VERSION,
        "structured_feature_schema_version": STRUCTURED_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "source_kind": "recorded_offline_teacher",
        "deck_prior_fingerprint": (
            deck_prior.fingerprint if deck_prior is not None else None
        ),
        "deck_belief_schema_version": deck_belief_schema_version,
        "structured_config": asdict(config),
        "paired_base_features": bool(config.contextual_value_features),
        "sources": source_identity,
        "skip_recovered_episodes": bool(skip_recovered_episodes),
        "max_decisions": max(0, int(max_decisions)),
        "min_teacher_margin": minimum_margin,
        "only_teacher_disagreements": bool(only_teacher_disagreements),
        "preserve_winner_actions": bool(preserve_winner_actions),
    }
    cache_fingerprint = _json_fingerprint(cache_identity)
    shard_buffer: list[StructuredDistillationExample] = []
    shard_records: list[dict[str, Any]] = []
    rulesets: set[str] = set()
    teacher_kinds: set[str] = set()
    counters = {
        "episodes_seen": 0,
        "episodes_used": 0,
        "skipped_recovered_episodes": 0,
        "skipped_truncated_episodes": 0,
        "decisions_seen": 0,
        "decisions_without_teacher": 0,
        "invalid_decisions": 0,
        "decisions_missing_teacher_margin": 0,
        "decisions_below_teacher_margin": 0,
        "teacher_behavior_agreements": 0,
        "teacher_behavior_disagreements": 0,
        "decisions_filtered_teacher_agreement": 0,
        "winner_actions_preserved": 0,
        "winner_teacher_disagreements": 0,
        "winner_teacher_actions": 0,
        "loser_teacher_actions": 0,
        "draw_teacher_actions": 0,
        "examples": 0,
    }
    progress = ProgressReporter(
        "build-search-cache",
        total=(max(0, int(max_decisions)) or max(0, int(expected_decisions)) or None),
        interval=progress_interval,
        enabled=show_progress,
    )

    def flush_full_shards() -> None:
        while len(shard_buffer) >= max(1, int(shard_size)):
            chunk = shard_buffer[:shard_size]
            del shard_buffer[:shard_size]
            shard_records.append(
                _write_shard(target, len(shard_records), chunk, config=config)
            )

    limit = max(0, int(max_decisions))
    stop = False
    for episode in iter_trajectory_episodes(source_paths):
        counters["episodes_seen"] += 1
        if bool(episode.get("truncated")) or not bool(episode.get("terminated")):
            counters["skipped_truncated_episodes"] += 1
            continue
        if skip_recovered_episodes and _as_int(episode.get("loop_recoveries"), 0) > 0:
            counters["skipped_recovered_episodes"] += 1
            continue
        ruleset = str(episode.get("ruleset_fingerprint") or "")
        if ruleset:
            rulesets.add(ruleset)
        counters["episodes_used"] += 1
        winner = _as_int(episode.get("winner"), -1)
        for decision in episode.get("decisions") or []:
            counters["decisions_seen"] += 1
            teacher = decision.get("teacher")
            if not isinstance(teacher, dict):
                counters["decisions_without_teacher"] += 1
                continue
            try:
                observation = decision.get("observation")
                if not isinstance(observation, dict):
                    raise ValueError("missing observation")
                legal = [
                    Action.from_dict(item)
                    for item in decision.get("legal_actions") or []
                ]
                executed_action = Action.from_dict(decision.get("action") or {})
                teacher_action_key = str(teacher.get("action_key") or "")
                teacher_index = next(
                    index for index, action in enumerate(legal)
                    if action.key == teacher_action_key
                )
                logits = tuple(float(value) for value in teacher.get("logits") or ())
                value = float(teacher.get("value"))
                if len(logits) != len(legal) or not logits:
                    raise ValueError("teacher logits do not match legal actions")
                if not all(math.isfinite(item) for item in (*logits, value)):
                    raise ValueError("teacher output contains non-finite values")
                teacher_argmax = max(range(len(logits)), key=logits.__getitem__)
                if teacher_argmax != teacher_index:
                    raise ValueError("teacher argmax does not match recorded action")
                executed_index = next(
                    index for index, action in enumerate(legal)
                    if action.key == executed_action.key
                )
                teacher_disagrees = teacher_index != executed_index
                if teacher_disagrees:
                    counters["teacher_behavior_disagreements"] += 1
                else:
                    counters["teacher_behavior_agreements"] += 1
                    if only_teacher_disagreements:
                        counters["decisions_filtered_teacher_agreement"] += 1
                        continue
                kind = str(teacher.get("kind") or "").strip()
                if not kind:
                    raise ValueError("teacher kind is missing")
                if minimum_margin > 0.0:
                    raw_margin = teacher.get("search_margin")
                    if raw_margin is None:
                        counters["decisions_missing_teacher_margin"] += 1
                        continue
                    margin = float(raw_margin)
                    if not math.isfinite(margin):
                        raise ValueError("teacher margin is not finite")
                    if margin < minimum_margin:
                        counters["decisions_below_teacher_margin"] += 1
                        continue
                decision_player = _as_int(decision.get("player"), -1)
                preserve_action = (
                    bool(preserve_winner_actions)
                    and winner in (0, 1)
                    and decision_player == winner
                )
                if preserve_action:
                    selected_index = executed_index
                    counters["winner_actions_preserved"] += 1
                    if selected_index != teacher_index:
                        counters["winner_teacher_disagreements"] += 1
                        logits = tuple(
                            0.0 if index == selected_index else -4.0
                            for index in range(len(legal))
                        )
                else:
                    selected_index = teacher_index
                    if winner not in (0, 1):
                        counters["draw_teacher_actions"] += 1
                    elif decision_player == winner:
                        counters["winner_teacher_actions"] += 1
                    else:
                        counters["loser_teacher_actions"] += 1
                structured_observation = _augment_with_deck_prior(
                    observation, deck_prior, dynamic=dynamic_deck_belief
                )
                structured, base_structured = _encode_paired_decisions(
                    structured_observation,
                    legal,
                    config=config,
                    selected_index=selected_index,
                )
                diagnostic = decision.get("diagnostic")
                example_weight = float(
                    (diagnostic or {}).get("hard_example_weight", 1.0)
                    if isinstance(diagnostic, dict)
                    else 1.0
                )
                if not math.isfinite(example_weight) or example_weight <= 0.0:
                    raise ValueError("hard-example weight must be finite and positive")
                example_weight = min(10.0, example_weight)
            except (KeyError, StopIteration, TypeError, ValueError):
                counters["invalid_decisions"] += 1
                continue
            teacher_kinds.add(kind)
            shard_buffer.append(StructuredDistillationExample(
                decision=structured,
                teacher_logits=logits,
                teacher_value=max(-1.0, min(1.0, value)),
                base_decision=base_structured,
                example_weight=example_weight,
            ))
            counters["examples"] += 1
            flush_full_shards()
            progress.update(
                counters["examples"],
                episodes=counters["episodes_used"],
                shards=len(shard_records),
            )
            if limit and counters["examples"] >= limit:
                stop = True
                break
        if stop:
            break
    if shard_buffer:
        shard_records.append(
            _write_shard(target, len(shard_records), shard_buffer, config=config)
        )
        shard_buffer.clear()
    if not shard_records:
        raise ValueError("trajectory data contains no recorded teacher decisions")
    if len(rulesets) != 1:
        raise ValueError("trajectory data must contain exactly one ruleset fingerprint")
    teacher_identity = {
        "kinds": sorted(teacher_kinds),
        "sources": source_identity,
    }
    manifest = {
        **cache_identity,
        "cache_fingerprint": cache_fingerprint,
        "ruleset_fingerprint": next(iter(rulesets)),
        "teacher_checkpoint": None,
        "teacher_name": ",".join(sorted(teacher_kinds)),
        "teacher_fingerprint": _json_fingerprint(teacher_identity),
        "counters": counters,
        "shards": shard_records,
    }
    _write_json_atomic(target / "manifest.json", manifest)
    progress.finish(
        counters["examples"],
        episodes=counters["episodes_used"],
        shards=len(shard_records),
    )
    return manifest


def _load_deck_prior(path: str | Path | None):
    if path is None:
        return None
    from .deck_prior import DeckPrior

    return DeckPrior.load(path)


def _deck_belief_schema_version(prior, *, dynamic: bool) -> int | None:
    if prior is None or not dynamic:
        return None
    from .deck_prior import DECK_BELIEF_FEATURE_SCHEMA_VERSION

    return int(DECK_BELIEF_FEATURE_SCHEMA_VERSION)


def _augment_with_deck_prior(
    observation: dict[str, Any], prior, *, dynamic: bool = False
):
    if prior is None:
        return observation
    from .deck_prior import augment_observation_with_deck_prior

    return augment_observation_with_deck_prior(
        observation,
        prior,
        include_public_evidence=dynamic,
    )


def load_cache_manifest(path: str | Path) -> dict[str, Any]:
    cache_dir = Path(path).resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("distillation cache manifest must contain an object")
    expected = {
        "cache_schema_version": DISTILLATION_CACHE_SCHEMA_VERSION,
        "structured_feature_schema_version": STRUCTURED_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
    }
    for key, expected_value in expected.items():
        if int(value.get(key, -1)) != int(expected_value):
            raise ValueError(f"unsupported distillation cache {key}")
    if not value.get("shards"):
        raise ValueError("distillation cache has no shards")
    return value


def iter_cached_examples(
    path: str | Path,
    *,
    split: str = "all",
    validation_fraction: float = 0.05,
    shuffle_shards: bool = False,
    seed: int = 0,
) -> Iterator[StructuredDistillationExample]:
    cache_dir = Path(path).resolve()
    manifest = load_cache_manifest(cache_dir)
    normalized_split = str(split or "all").lower()
    if normalized_split not in {"all", "train", "validation"}:
        raise ValueError(f"unsupported split: {split}")
    fraction = max(0.0, min(0.5, float(validation_fraction)))
    shards = []
    shard_start = 0
    for shard_record in manifest["shards"]:
        shards.append((shard_record, shard_start))
        shard_start += int(shard_record["examples"])
    if shuffle_shards:
        random.Random(int(seed)).shuffle(shards)
    for shard_record, shard_start in shards:
        shard_path = cache_dir / str(shard_record["file"])
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        count = int(payload["example_count"])
        for local_index in range(count):
            global_index = shard_start + local_index
            in_validation = _is_validation_example(
                manifest["cache_fingerprint"], global_index, fraction
            )
            if normalized_split == "train" and in_validation:
                continue
            if normalized_split == "validation" and not in_validation:
                continue
            yield _decode_example(payload, local_index)


def cached_split_count(
    manifest: dict[str, Any],
    *,
    split: str,
    validation_fraction: float,
) -> int:
    normalized_split = str(split or "all").lower()
    if normalized_split not in {"all", "train", "validation"}:
        raise ValueError(f"unsupported split: {split}")
    total = int((manifest.get("counters") or {}).get("examples", 0))
    if normalized_split == "all" or validation_fraction <= 0:
        return total if normalized_split != "validation" else 0
    fraction = max(0.0, min(0.5, float(validation_fraction)))
    validation = sum(
        _is_validation_example(manifest["cache_fingerprint"], index, fraction)
        for index in range(total)
    )
    return validation if normalized_split == "validation" else total - validation


def iter_cached_batches(
    examples: Iterable[StructuredDistillationExample],
    *,
    batch_size: int,
    shuffle_buffer: int,
    rng: random.Random,
    bucket_by_size: bool = True,
) -> Iterator[list[StructuredDistillationExample]]:
    size = max(1, int(batch_size))
    capacity = max(size, int(shuffle_buffer))
    buffer: list[StructuredDistillationExample] = []
    for example in examples:
        buffer.append(example)
        if len(buffer) < capacity:
            continue
        yield from _drain_cached_buffer(
            buffer, batch_size=size, rng=rng, bucket_by_size=bucket_by_size
        )
    if buffer:
        yield from _drain_cached_buffer(
            buffer, batch_size=size, rng=rng, bucket_by_size=bucket_by_size
        )


def _drain_cached_buffer(
    buffer: list[StructuredDistillationExample],
    *,
    batch_size: int,
    rng: random.Random,
    bucket_by_size: bool,
) -> Iterator[list[StructuredDistillationExample]]:
    rng.shuffle(buffer)
    if bucket_by_size:
        buffer.sort(key=_example_attention_cost)
    batches = [
        buffer[start:start + batch_size]
        for start in range(0, len(buffer), batch_size)
    ]
    buffer.clear()
    rng.shuffle(batches)
    yield from batches


def _example_attention_cost(example: StructuredDistillationExample) -> int:
    state = len(example.decision.state_tokens)
    actions = example.decision.action_count
    return state * state + actions * actions + state * actions


def _write_shard(
    target: Path,
    shard_index: int,
    examples: Sequence[StructuredDistillationExample],
    *,
    config: StructuredModelConfig,
) -> dict[str, Any]:
    paired_base_features = any(
        example.base_decision is not None for example in examples
    )
    if paired_base_features and any(
        example.base_decision is None for example in examples
    ):
        raise ValueError("a cache shard cannot mix paired and unpaired features")
    state_tokens: list[EntityToken] = []
    action_tokens: list[EntityToken] = []
    state_offsets = [0]
    action_offsets = [0]
    base_state_tokens: list[EntityToken] = []
    base_action_tokens: list[EntityToken] = []
    base_state_offsets = [0]
    base_action_offsets = [0]
    teacher_logits: list[float] = []
    action_biases: list[float] = []
    for example in examples:
        state_tokens.extend(example.decision.state_tokens)
        action_tokens.extend(example.decision.action_tokens)
        state_offsets.append(len(state_tokens))
        action_offsets.append(len(action_tokens))
        if example.base_decision is not None:
            base_state_tokens.extend(example.base_decision.state_tokens)
            base_action_tokens.extend(example.base_decision.action_tokens)
            base_state_offsets.append(len(base_state_tokens))
            base_action_offsets.append(len(base_action_tokens))
        teacher_logits.extend(example.teacher_logits)
        biases = example.decision.action_logit_biases or (
            0.0,
        ) * example.decision.action_count
        action_biases.extend(biases)
    payload: dict[str, Any] = {
        "schema_version": DISTILLATION_CACHE_SCHEMA_VERSION,
        "example_count": len(examples),
        "state_offsets": torch.tensor(state_offsets, dtype=torch.int64),
        "action_offsets": torch.tensor(action_offsets, dtype=torch.int64),
        "teacher_logits": torch.tensor(teacher_logits, dtype=torch.float32),
        "teacher_values": torch.tensor(
            [example.teacher_value for example in examples], dtype=torch.float32
        ),
        "example_weights": torch.tensor(
            [example.example_weight for example in examples], dtype=torch.float32
        ),
        "action_biases": torch.tensor(action_biases, dtype=torch.float32),
        "phases": torch.tensor(
            [example.decision.phase for example in examples], dtype=torch.uint8
        ),
        "selected_indices": torch.tensor(
            [example.decision.selected_index for example in examples], dtype=torch.int32
        ),
    }
    payload.update(_encode_token_block("state", state_tokens, config=config))
    payload.update(_encode_token_block("action", action_tokens, config=config))
    if paired_base_features:
        payload["base_state_offsets"] = torch.tensor(
            base_state_offsets, dtype=torch.int64
        )
        payload["base_action_offsets"] = torch.tensor(
            base_action_offsets, dtype=torch.int64
        )
        payload.update(_encode_token_block(
            "base_state", base_state_tokens, config=config
        ))
        payload.update(_encode_token_block(
            "base_action", base_action_tokens, config=config
        ))
    filename = f"shard-{shard_index:05d}.pt"
    final_path = target / filename
    temporary = final_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(final_path)
    return {
        "file": filename,
        "examples": len(examples),
        "state_tokens": len(state_tokens),
        "action_tokens": len(action_tokens),
        "paired_base_features": paired_base_features,
        "bytes": final_path.stat().st_size,
    }


def _encode_token_block(
    prefix: str,
    tokens: Sequence[EntityToken],
    *,
    config: StructuredModelConfig,
) -> dict[str, Any]:
    categorical_ids: list[int] = []
    categorical_offsets = [0]
    numeric_indices: list[int] = []
    numeric_values: list[float] = []
    numeric_offsets = [0]
    for token in tokens:
        categorical_ids.extend(token.categorical_ids[:config.categorical_slots])
        categorical_offsets.append(len(categorical_ids))
        for index, value in enumerate(token.numeric_values[:config.numeric_buckets]):
            if value:
                numeric_indices.append(index)
                numeric_values.append(float(value))
        numeric_offsets.append(len(numeric_indices))
    return {
        f"{prefix}_types": torch.tensor(
            [token.token_type for token in tokens], dtype=torch.uint8
        ),
        f"{prefix}_categorical_ids": torch.tensor(categorical_ids, dtype=torch.int32),
        f"{prefix}_categorical_offsets": torch.tensor(
            categorical_offsets, dtype=torch.int64
        ),
        f"{prefix}_numeric_indices": torch.tensor(numeric_indices, dtype=torch.uint8),
        f"{prefix}_numeric_values": torch.tensor(numeric_values, dtype=torch.float32),
        f"{prefix}_numeric_offsets": torch.tensor(numeric_offsets, dtype=torch.int64),
    }


def _decode_example(payload: dict[str, Any], index: int) -> StructuredDistillationExample:
    state_start = int(payload["state_offsets"][index])
    state_end = int(payload["state_offsets"][index + 1])
    action_start = int(payload["action_offsets"][index])
    action_end = int(payload["action_offsets"][index + 1])
    state_tokens = _decode_token_block(payload, "state", state_start, state_end)
    action_tokens = _decode_token_block(payload, "action", action_start, action_end)
    base_decision = None
    if "base_state_offsets" in payload:
        base_state_start = int(payload["base_state_offsets"][index])
        base_state_end = int(payload["base_state_offsets"][index + 1])
        base_action_start = int(payload["base_action_offsets"][index])
        base_action_end = int(payload["base_action_offsets"][index + 1])
        base_decision = StructuredDecision(
            state_tokens=tuple(_decode_token_block(
                payload, "base_state", base_state_start, base_state_end
            )),
            action_tokens=tuple(_decode_token_block(
                payload, "base_action", base_action_start, base_action_end
            )),
            phase=int(payload["phases"][index]),
            selected_index=int(payload["selected_indices"][index]),
            action_logit_biases=tuple(
                float(value)
                for value in payload[
                    "action_biases"
                ][action_start:action_end].tolist()
            ),
        )
    return StructuredDistillationExample(
        decision=StructuredDecision(
            state_tokens=tuple(state_tokens),
            action_tokens=tuple(action_tokens),
            phase=int(payload["phases"][index]),
            selected_index=int(payload["selected_indices"][index]),
            action_logit_biases=tuple(
                float(value)
                for value in payload["action_biases"][action_start:action_end].tolist()
            ),
        ),
        teacher_logits=tuple(
            float(value)
            for value in payload["teacher_logits"][action_start:action_end].tolist()
        ),
        teacher_value=float(payload["teacher_values"][index]),
        base_decision=base_decision,
        example_weight=(
            float(payload["example_weights"][index])
            if "example_weights" in payload
            else 1.0
        ),
    )


def _decode_token_block(
    payload: dict[str, Any],
    prefix: str,
    start: int,
    end: int,
) -> list[EntityToken]:
    output: list[EntityToken] = []
    categorical_offsets = payload[f"{prefix}_categorical_offsets"]
    numeric_offsets = payload[f"{prefix}_numeric_offsets"]
    categorical_ids = payload[f"{prefix}_categorical_ids"]
    numeric_indices = payload[f"{prefix}_numeric_indices"]
    numeric_values = payload[f"{prefix}_numeric_values"]
    numeric_size = 0
    if numeric_indices.numel():
        numeric_size = int(numeric_indices.max()) + 1
    for token_index in range(start, end):
        cat_start = int(categorical_offsets[token_index])
        cat_end = int(categorical_offsets[token_index + 1])
        num_start = int(numeric_offsets[token_index])
        num_end = int(numeric_offsets[token_index + 1])
        dense_numeric = [0.0] * numeric_size
        for numeric_index, value in zip(
            numeric_indices[num_start:num_end].tolist(),
            numeric_values[num_start:num_end].tolist(),
        ):
            dense_numeric[int(numeric_index)] = float(value)
        output.append(EntityToken(
            token_type=int(payload[f"{prefix}_types"][token_index]),
            categorical_ids=tuple(
                int(value) for value in categorical_ids[cat_start:cat_end].tolist()
            ),
            numeric_values=tuple(dense_numeric),
        ))
    return output


def _prepare_cache_directory(target: Path, *, overwrite: bool) -> None:
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"distillation cache directory is not empty: {target}"
            )
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    target.mkdir(parents=True, exist_ok=True)


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _is_validation_example(cache_fingerprint: str, index: int, fraction: float) -> bool:
    if fraction <= 0:
        return False
    digest = hashlib.blake2b(
        f"{cache_fingerprint}:{index}".encode("ascii"),
        digest_size=8,
        person=b"GTNAISPL",
    ).digest()
    return int.from_bytes(digest, "little") / float(1 << 64) < fraction


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
