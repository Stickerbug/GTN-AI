from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .neural_model import (
    EncodedDecision,
    NeuralModelConfig,
    NeuralPolicy,
    VariableActionNetwork,
    collate_decisions,
    encode_decision,
    load_neural_checkpoint,
    require_torch,
    resolve_device,
    torch,
)
from .protocol import Action
from .trajectory import TRAJECTORY_SCHEMA_VERSION


def iter_trajectory_episodes(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: episode must be an object")
                version = int(value.get("schema_version", -1))
                if version != TRAJECTORY_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: unsupported trajectory schema {version}; "
                        f"expected {TRAJECTORY_SCHEMA_VERSION}"
                    )
                yield value


def inspect_trajectory_files(
    paths: Iterable[str | Path],
    *,
    skip_recovered_episodes: bool = True,
) -> dict[str, Any]:
    fingerprints: set[str] = set()
    episodes = 0
    decisions = 0
    usable_decisions = 0
    phases: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    recovered_episodes = 0
    recovered_decisions = 0
    for episode in iter_trajectory_episodes(paths):
        if skip_recovered_episodes and _as_int(episode.get("loop_recoveries"), 0) > 0:
            recovered_episodes += 1
            recovered_decisions += len(episode.get("decisions") or [])
            continue
        episodes += 1
        fingerprint = str(episode.get("ruleset_fingerprint") or "")
        if fingerprint:
            fingerprints.add(fingerprint)
        winner = _as_int(episode.get("winner"), -1)
        outcomes["draw" if winner not in (0, 1) else f"player_{winner}_win"] += 1
        for decision in episode.get("decisions") or []:
            decisions += 1
            if decision.get("forced_fallback"):
                continue
            observation = decision.get("observation")
            if not isinstance(observation, dict):
                continue
            usable_decisions += 1
            phases[_phase_name(observation)] += 1
    if not fingerprints:
        raise ValueError("training data has no ruleset fingerprint")
    if len(fingerprints) != 1:
        raise ValueError("training data mixes different game rulesets; split or regenerate the dataset")
    if not usable_decisions:
        raise ValueError("training data contains no usable decisions")
    return {
        "episodes": episodes,
        "decisions": decisions,
        "usable_decisions": usable_decisions,
        "phase_decisions": dict(sorted(phases.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "ruleset_fingerprint": next(iter(fingerprints)),
        "skipped_recovered_episodes": recovered_episodes,
        "skipped_recovered_decisions": recovered_decisions,
    }


def iter_encoded_decisions(
    paths: Iterable[str | Path],
    *,
    config: NeuralModelConfig,
    split: str = "all",
    validation_fraction: float = 0.1,
    winner_policy_weight: float = 1.0,
    loser_policy_weight: float = 1.0,
    draw_policy_weight: float = 1.0,
    skip_recovered_episodes: bool = True,
) -> Iterator[EncodedDecision]:
    normalized_split = str(split or "all").lower()
    if normalized_split not in {"all", "train", "validation"}:
        raise ValueError(f"unsupported split: {split}")
    fraction = max(0.0, min(0.5, float(validation_fraction)))
    for episode in iter_trajectory_episodes(paths):
        if skip_recovered_episodes and _as_int(episode.get("loop_recoveries"), 0) > 0:
            continue
        in_validation = _episode_validation_partition(episode, fraction)
        if normalized_split == "train" and in_validation:
            continue
        if normalized_split == "validation" and not in_validation:
            continue
        winner = _as_int(episode.get("winner"), -1)
        for decision in episode.get("decisions") or []:
            if decision.get("forced_fallback"):
                continue
            observation = decision.get("observation")
            if not isinstance(observation, dict):
                continue
            try:
                selected = Action.from_dict(decision.get("action") or {})
                legal = [Action.from_dict(item) for item in decision.get("legal_actions") or []]
            except (TypeError, ValueError):
                continue
            selected_index = next(
                (index for index, action in enumerate(legal) if action.key == selected.key),
                None,
            )
            if selected_index is None or not legal:
                continue
            player = _as_int(decision.get("player"), -1)
            target = 0.0 if winner not in (0, 1) else (1.0 if winner == player else -1.0)
            if winner not in (0, 1):
                policy_weight = draw_policy_weight
            elif winner == player:
                policy_weight = winner_policy_weight
            else:
                policy_weight = loser_policy_weight
            yield encode_decision(
                observation,
                legal,
                config=config,
                selected_index=selected_index,
                value_target=target,
                policy_weight=policy_weight,
            )


def iter_shuffled_batches(
    examples: Iterable[EncodedDecision],
    *,
    batch_size: int,
    shuffle_buffer: int,
    rng: random.Random,
) -> Iterator[list[EncodedDecision]]:
    size = max(1, int(batch_size))
    capacity = max(size, int(shuffle_buffer))
    buffer: list[EncodedDecision] = []
    batch: list[EncodedDecision] = []
    for example in examples:
        buffer.append(example)
        if len(buffer) < capacity:
            continue
        selected = rng.randrange(len(buffer))
        batch.append(buffer.pop(selected))
        if len(batch) >= size:
            yield batch
            batch = []
    rng.shuffle(buffer)
    for example in buffer:
        batch.append(example)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_neural_behavior_policy(
    paths: Sequence[str | Path],
    *,
    config: NeuralModelConfig | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    shuffle_buffer: int = 4096,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    value_loss_weight: float = 0.25,
    validation_fraction: float = 0.1,
    max_grad_norm: float = 2.0,
    device: str = "auto",
    seed: int = 1,
    initial_checkpoint: str | Path | None = None,
    winner_policy_weight: float = 1.0,
    loser_policy_weight: float = 1.0,
    draw_policy_weight: float = 1.0,
    skip_recovered_episodes: bool = True,
) -> tuple[NeuralPolicy, dict[str, Any]]:
    require_torch()
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("at least one trajectory file is required")
    dataset = inspect_trajectory_files(
        files,
        skip_recovered_episodes=skip_recovered_episodes,
    )
    resolved_device = resolve_device(device)
    initial = (
        load_neural_checkpoint(initial_checkpoint, device=resolved_device)
        if initial_checkpoint is not None
        else None
    )
    if initial is not None:
        checkpoint_config = initial["config"]
        if config is not None and config != checkpoint_config:
            raise ValueError("initial checkpoint architecture does not match the requested config")
        if initial["ruleset_fingerprint"] != dataset["ruleset_fingerprint"]:
            raise ValueError("initial checkpoint ruleset does not match the training data")
        model_config = checkpoint_config
    else:
        model_config = config or NeuralModelConfig()
    torch.manual_seed(int(seed))
    if resolved_device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(int(seed))
    elif resolved_device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = (
        initial["model"] if initial is not None else VariableActionNetwork(model_config)
    ).to(torch.device(resolved_device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    epoch_metrics: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(max(0, int(epochs))):
        model.train()
        training = _MetricAccumulator()
        examples = iter_encoded_decisions(
            files,
            config=model_config,
            split="train" if validation_fraction > 0 else "all",
            validation_fraction=validation_fraction,
            winner_policy_weight=winner_policy_weight,
            loser_policy_weight=loser_policy_weight,
            draw_policy_weight=draw_policy_weight,
            skip_recovered_episodes=skip_recovered_episodes,
        )
        batches = iter_shuffled_batches(
            examples,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            rng=random.Random(int(seed) + epoch * 104729),
        )
        for encoded in batches:
            batch = collate_decisions(encoded, device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            scores, values = model(batch)
            policy_loss, correct = _grouped_policy_loss(scores, batch)
            value_loss = torch.nn.functional.mse_loss(values, batch["value_targets"])
            loss = policy_loss + float(value_loss_weight) * value_loss
            if not torch.isfinite(loss):
                raise RuntimeError("training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()
            training.add(
                encoded,
                policy_loss=float(policy_loss.detach().to("cpu").item()),
                value_loss=float(value_loss.detach().to("cpu").item()),
                correct=correct,
                value_predictions=values.detach().to("cpu").tolist(),
            )

        if training.examples == 0:
            raise ValueError(
                "training split contains no usable decisions; add episodes or set "
                "validation_fraction to 0"
            )

        validation = evaluate_neural_model(
            model,
            iter_encoded_decisions(
                files,
                config=model_config,
                split="validation",
                validation_fraction=validation_fraction,
                winner_policy_weight=winner_policy_weight,
                loser_policy_weight=loser_policy_weight,
                draw_policy_weight=draw_policy_weight,
                skip_recovered_episodes=skip_recovered_episodes,
            ),
            batch_size=batch_size,
            device=resolved_device,
        ) if validation_fraction > 0 else {}
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train": training.summary(),
            "validation": validation,
        })

    elapsed = time.perf_counter() - started
    policy = NeuralPolicy(
        model,
        config=model_config,
        ruleset_fingerprint=dataset["ruleset_fingerprint"],
        device=resolved_device,
        seed=seed,
    )
    metrics = {
        "objective": "outcome_weighted_behavior_cloning_with_terminal_value",
        "device": resolved_device,
        "epochs": max(0, int(epochs)),
        "seconds": round(elapsed, 3),
        "dataset": dataset,
        "epoch_metrics": epoch_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_checkpoint": str(Path(initial_checkpoint).resolve()) if initial_checkpoint else None,
        "policy_weights": {
            "winner": float(winner_policy_weight),
            "loser": float(loser_policy_weight),
            "draw": float(draw_policy_weight),
        },
        "skip_recovered_episodes": bool(skip_recovered_episodes),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    return policy, metrics


def evaluate_neural_model(
    model: VariableActionNetwork,
    examples: Iterable[EncodedDecision],
    *,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    require_torch()
    model.eval()
    metrics = _MetricAccumulator()
    batch: list[EncodedDecision] = []
    with torch.inference_mode():
        for example in examples:
            batch.append(example)
            if len(batch) >= max(1, int(batch_size)):
                _evaluate_batch(model, batch, metrics, device)
                batch = []
        if batch:
            _evaluate_batch(model, batch, metrics, device)
    return metrics.summary()


def _evaluate_batch(model, encoded, metrics, device) -> None:
    batch = collate_decisions(encoded, device=device)
    scores, values = model(batch)
    policy_loss, correct = _grouped_policy_loss(scores, batch)
    value_loss = torch.nn.functional.mse_loss(values, batch["value_targets"])
    metrics.add(
        encoded,
        policy_loss=float(policy_loss.detach().to("cpu").item()),
        value_loss=float(value_loss.detach().to("cpu").item()),
        correct=correct,
        value_predictions=values.detach().to("cpu").tolist(),
    )


def _grouped_policy_loss(scores, batch):
    losses = []
    weights = []
    correct: list[bool] = []
    offsets = batch["action_set_offsets"].detach().to("cpu").tolist()
    selected = batch["selected_absolute"].detach().to("cpu").tolist()
    for index in range(len(offsets) - 1):
        start, end = int(offsets[index]), int(offsets[index + 1])
        target = int(selected[index]) - start
        if target < 0 or start >= end:
            raise ValueError("batch contains an invalid selected action")
        group = scores[start:end]
        losses.append(torch.nn.functional.cross_entropy(
            group.unsqueeze(0),
            torch.tensor([target], dtype=torch.long, device=group.device),
        ))
        weights.append(batch["policy_weights"][index])
        correct.append(int(group.argmax().detach().to("cpu").item()) == target)
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).to(loss_tensor.device)
    weight_sum = weight_tensor.sum()
    if float(weight_sum.detach().to("cpu").item()) <= 0:
        raise ValueError("batch policy weights must include at least one positive value")
    return (loss_tensor * weight_tensor).sum() / weight_sum, correct


class _MetricAccumulator:
    def __init__(self):
        self.examples = 0
        self.batches = 0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.correct = 0
        self.value_squared_error = 0.0
        self.phase_examples: Counter[str] = Counter()
        self.phase_correct: Counter[str] = Counter()

    def add(
        self,
        examples: Sequence[EncodedDecision],
        *,
        policy_loss: float,
        value_loss: float,
        correct: Sequence[bool],
        value_predictions: Sequence[float],
    ) -> None:
        count = len(examples)
        self.examples += count
        self.batches += 1
        self.policy_loss += float(policy_loss) * count
        self.value_loss += float(value_loss) * count
        for item, is_correct, prediction in zip(examples, correct, value_predictions):
            phase = "pregame" if item.phase == 0 else "combat"
            self.phase_examples[phase] += 1
            self.phase_correct[phase] += int(bool(is_correct))
            self.correct += int(bool(is_correct))
            self.value_squared_error += (float(prediction) - item.value_target) ** 2

    def summary(self) -> dict[str, Any]:
        count = max(1, self.examples)
        return {
            "examples": self.examples,
            "batches": self.batches,
            "policy_loss": round(self.policy_loss / count, 6),
            "value_loss": round(self.value_loss / count, 6),
            "accuracy": round(self.correct / count, 6),
            "value_rmse": round(math.sqrt(self.value_squared_error / count), 6),
            "phase_examples": dict(sorted(self.phase_examples.items())),
            "phase_accuracies": {
                phase: round(self.phase_correct[phase] / max(1, total), 6)
                for phase, total in sorted(self.phase_examples.items())
            },
        }


def _episode_validation_partition(episode: dict[str, Any], fraction: float) -> bool:
    if fraction <= 0:
        return False
    identity = json.dumps({
        "seed": episode.get("seed"),
        "mods": episode.get("official_mods") or [],
        "fingerprint": episode.get("ruleset_fingerprint") or "",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8, person=b"GTNAISPL").digest()
    ratio = int.from_bytes(digest, "little") / float(1 << 64)
    return ratio < fraction


def _phase_name(observation: dict[str, Any]) -> str:
    return "pregame" if observation.get("phase") == "pregame" else "combat"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
