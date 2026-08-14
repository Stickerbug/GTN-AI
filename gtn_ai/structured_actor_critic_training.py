from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .actor_critic_training import (
    generalized_advantage_targets,
    inspect_on_policy_trajectories,
)
from .neural_model import (
    neural_state_dict_fingerprint,
    require_torch,
    resolve_device,
    torch,
)
from .neural_training import iter_trajectory_episodes
from .progress import ProgressReporter
from .protocol import Action
from .structured_cache import iter_cached_batches
from .structured_features import StructuredDecision, encode_structured_decision
from .structured_model import (
    StructuredPolicy,
    StructuredModelConfig,
    collate_structured_decisions,
    load_structured_checkpoint,
)


@dataclass(frozen=True)
class StructuredOnPolicyExample:
    decision: StructuredDecision
    value_target: float
    behavior_log_prob: float
    behavior_value: float
    behavior_temperature: float
    advantage_target: float


def iter_structured_on_policy_decisions(
    paths: Sequence[str | Path],
    *,
    config: StructuredModelConfig,
    expected_policy_fingerprint: str,
    skip_recovered_episodes: bool = True,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Iterator[StructuredOnPolicyExample]:
    for episode in iter_trajectory_episodes(paths):
        if bool(episode.get("truncated")) or not bool(episode.get("terminated")):
            continue
        if skip_recovered_episodes and _as_int(episode.get("loop_recoveries"), 0) > 0:
            continue
        fingerprints = [
            str(value or "") for value in episode.get("policy_fingerprints") or []
        ]
        if len(fingerprints) != 2:
            raise ValueError("on-policy episode is missing its two policy fingerprints")
        actor_seats = {
            seat for seat, value in enumerate(fingerprints)
            if value == expected_policy_fingerprint
        }
        if not actor_seats:
            continue
        targets = _episode_targets(
            episode,
            actor_seats,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for decision_index, raw in enumerate(episode.get("decisions") or []):
            player = _as_int(raw.get("player"), -1)
            if player not in actor_seats:
                continue
            if raw.get("forced_fallback"):
                raise ValueError("on-policy episode contains a forced fallback decision")
            metadata = _behavior_metadata(raw)
            try:
                selected = Action.from_dict(raw.get("action") or {})
                legal = [
                    Action.from_dict(item) for item in raw.get("legal_actions") or []
                ]
            except (TypeError, ValueError):
                continue
            selected_index = next(
                (
                    index for index, action in enumerate(legal)
                    if action.key == selected.key
                ),
                None,
            )
            observation = raw.get("observation")
            if selected_index is None or not legal or not isinstance(observation, dict):
                continue
            advantage, target = targets[decision_index]
            yield StructuredOnPolicyExample(
                decision=encode_structured_decision(
                    observation,
                    legal,
                    config=config.feature_config,
                    selected_index=selected_index,
                ),
                value_target=float(target),
                behavior_log_prob=metadata["log_prob"],
                behavior_value=metadata["value"],
                behavior_temperature=metadata["temperature"],
                advantage_target=float(advantage),
            )


def train_structured_actor_critic_policy(
    paths: Sequence[str | Path],
    *,
    initial_checkpoint: str | Path,
    epochs: int = 1,
    batch_size: int = 128,
    shuffle_buffer: int = 256,
    learning_rate: float = 2.5e-6,
    weight_decay: float = 1e-5,
    clip_ratio: float = 0.15,
    value_clip: float = 0.15,
    value_loss_weight: float = 0.25,
    entropy_weight: float = 0.004,
    max_grad_norm: float = 1.0,
    target_kl: float = 0.02,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    device: str = "auto",
    seed: int = 1,
    skip_recovered_episodes: bool = True,
    progress_interval: float = 10.0,
    show_progress: bool = True,
) -> tuple[StructuredPolicy, dict[str, Any]]:
    require_torch()
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("at least one trajectory file is required")
    resolved_device = resolve_device(device)
    initial = load_structured_checkpoint(initial_checkpoint, device=resolved_device)
    initial_fingerprint = neural_state_dict_fingerprint(initial["model"].state_dict())
    dataset = inspect_on_policy_trajectories(
        files,
        expected_policy_fingerprint=initial_fingerprint,
        skip_recovered_episodes=skip_recovered_episodes,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    if initial["ruleset_fingerprint"] != dataset["ruleset_fingerprint"]:
        raise ValueError("initial checkpoint ruleset does not match the on-policy data")

    torch.manual_seed(int(seed))
    if resolved_device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(int(seed))
    elif resolved_device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = initial["model"].to(torch.device(resolved_device))
    model.eval()  # Behavior log-probabilities were recorded with dropout disabled.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    started = time.perf_counter()
    epoch_metrics: list[dict[str, Any]] = []
    stopped_early = False
    requested_epochs = max(0, int(epochs))
    for epoch in range(requested_epochs):
        accumulator = _PpoAccumulator()
        progress = ProgressReporter(
            f"structured-ppo {epoch + 1}/{requested_epochs}",
            total=int(dataset["decisions"]),
            interval=progress_interval,
            enabled=show_progress,
        )
        progress.update(0, force=True, stage="loading first bucket")
        examples = iter_structured_on_policy_decisions(
            files,
            config=initial["config"],
            expected_policy_fingerprint=initial_fingerprint,
            skip_recovered_episodes=skip_recovered_episodes,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        batches = iter_cached_batches(
            examples,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            rng=random.Random(int(seed) + epoch * 104729),
        )
        for examples_batch in batches:
            batch = collate_structured_ppo_examples(
                examples_batch,
                config=initial["config"],
                device=resolved_device,
            )
            optimizer.zero_grad(set_to_none=True)
            scores, values = model(batch)
            loss, stats = _structured_ppo_loss(
                scores,
                values,
                batch,
                advantage_mean=float(dataset["advantage_mean"]),
                advantage_std=float(dataset["advantage_std"]),
                clip_ratio=float(clip_ratio),
                value_clip=float(value_clip),
                value_loss_weight=float(value_loss_weight),
                entropy_weight=float(entropy_weight),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("structured actor-critic produced a non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(max_grad_norm)
            )
            optimizer.step()
            accumulator.add(
                len(examples_batch),
                stats,
                float(gradient_norm.detach().to("cpu").item()),
            )
            progress.update(
                accumulator.examples,
                loss=f"{stats['loss']:.4f}",
                kl=f"{stats['approx_kl']:.5f}",
                clip=f"{stats['clip_fraction'] * 100:.1f}%",
            )
        summary = accumulator.summary()
        progress.finish(
            accumulator.examples,
            loss=f"{summary['loss']:.4f}",
            kl=f"{summary['approx_kl']:.5f}",
            clip=f"{summary['clip_fraction'] * 100:.1f}%",
        )
        if summary["examples"] == 0:
            raise ValueError("on-policy training split contains no usable decisions")
        epoch_metrics.append({"epoch": epoch + 1, **summary})
        if target_kl > 0 and summary["approx_kl"] > float(target_kl):
            stopped_early = True
            break

    policy = StructuredPolicy(
        model,
        config=initial["config"],
        ruleset_fingerprint=dataset["ruleset_fingerprint"],
        device=resolved_device,
        seed=seed,
        name="structured-actor-critic-candidate",
    )
    elapsed = time.perf_counter() - started
    return policy, {
        "objective": "structured_clipped_on_policy_actor_critic",
        "device": resolved_device,
        "epochs_requested": requested_epochs,
        "epochs_completed": len(epoch_metrics),
        "stopped_early": stopped_early,
        "seconds": round(elapsed, 3),
        "dataset": dataset,
        "epoch_metrics": epoch_metrics,
        "initial_checkpoint": str(Path(initial_checkpoint).resolve()),
        "initial_policy_fingerprint": initial_fingerprint,
        "result_policy_fingerprint": policy.model_fingerprint,
        "hyperparameters": {
            "batch_size": int(batch_size),
            "shuffle_buffer": int(shuffle_buffer),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "clip_ratio": float(clip_ratio),
            "value_clip": float(value_clip),
            "value_loss_weight": float(value_loss_weight),
            "entropy_weight": float(entropy_weight),
            "max_grad_norm": float(max_grad_norm),
            "target_kl": float(target_kl),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "seed": int(seed),
        },
    }


def collate_structured_ppo_examples(
    examples: Sequence[StructuredOnPolicyExample],
    *,
    config: StructuredModelConfig,
    device: str,
) -> dict[str, Any]:
    items = list(examples)
    batch = collate_structured_decisions(
        [item.decision for item in items], config=config, device=device
    )
    tensor_device = torch.device(device)
    for key in (
        "value_target",
        "behavior_log_prob",
        "behavior_value",
        "behavior_temperature",
        "advantage_target",
    ):
        batch[f"{key}s"] = torch.tensor(
            [float(getattr(item, key)) for item in items],
            dtype=torch.float32,
            device=tensor_device,
        )
    return batch


def _structured_ppo_loss(
    scores,
    values,
    batch,
    *,
    advantage_mean: float,
    advantage_std: float,
    clip_ratio: float,
    value_clip: float,
    value_loss_weight: float,
    entropy_weight: float,
):
    mask = batch["action_mask"]
    logits = scores + batch["action_logit_biases"]
    maximum = logits.masked_fill(~mask, -1e9).max(dim=1, keepdim=True).values
    temperature = batch["behavior_temperatures"].clamp_min(1e-6).unsqueeze(1)
    normalized = ((logits - maximum) / temperature).clamp_min(-60.0)
    normalized = normalized.masked_fill(~mask, -1e9)
    log_probs = torch.nn.functional.log_softmax(normalized, dim=1)
    probabilities = log_probs.exp().masked_fill(~mask, 0.0)
    selected = batch["selected_indices"].unsqueeze(1)
    new_log_prob = log_probs.gather(1, selected).squeeze(1)
    entropy = -(probabilities * log_probs.masked_fill(~mask, 0.0)).sum(dim=1).mean()

    old_log_prob = batch["behavior_log_probs"]
    raw_advantage = batch["advantage_targets"]
    advantages = (
        (raw_advantage - float(advantage_mean))
        / max(1e-6, float(advantage_std))
    )
    log_ratio = (new_log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    old_values = batch["behavior_values"]
    targets = batch["value_targets"]
    clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
    value_loss = 0.5 * torch.maximum(
        (values - targets).square(),
        (clipped_values - targets).square(),
    ).mean()
    total = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    return total, {
        "loss": float(total.detach().to("cpu").item()),
        "policy_loss": float(policy_loss.detach().to("cpu").item()),
        "value_loss": float(value_loss.detach().to("cpu").item()),
        "entropy": float(entropy.detach().to("cpu").item()),
        "approx_kl": float(approx_kl.detach().to("cpu").item()),
        "clip_fraction": float(clip_fraction.detach().to("cpu").item()),
    }


def _episode_targets(
    episode: dict[str, Any],
    actor_seats: set[int],
    *,
    gamma: float,
    gae_lambda: float,
) -> dict[int, tuple[float, float]]:
    decisions = list(episode.get("decisions") or [])
    by_seat: dict[int, list[tuple[int, float]]] = {seat: [] for seat in actor_seats}
    for index, decision in enumerate(decisions):
        player = _as_int(decision.get("player"), -1)
        if player not in actor_seats:
            continue
        metadata = _behavior_metadata(decision)
        by_seat[player].append((index, metadata["value"]))
    winner = _as_int(episode.get("winner"), -1)
    targets: dict[int, tuple[float, float]] = {}
    for seat, entries in by_seat.items():
        terminal = 0.0 if winner not in (0, 1) else (1.0 if winner == seat else -1.0)
        computed = generalized_advantage_targets(
            [value for _, value in entries],
            terminal,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        targets.update({index: target for (index, _), target in zip(entries, computed)})
    return targets


def _behavior_metadata(decision: dict[str, Any]) -> dict[str, float]:
    values = {
        "log_prob": decision.get("behavior_log_prob"),
        "value": decision.get("behavior_value"),
        "entropy": decision.get("behavior_entropy"),
        "temperature": decision.get("behavior_temperature"),
    }
    if any(value is None for value in values.values()):
        raise ValueError("on-policy decision is missing behavior statistics")
    result = {key: float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("on-policy decision contains non-finite behavior statistics")
    if result["temperature"] <= 0:
        raise ValueError("behavior temperature must be positive")
    return result


class _PpoAccumulator:
    def __init__(self) -> None:
        self.examples = 0
        self.batches = 0
        self.totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }

    def add(self, count: int, stats: dict[str, float], gradient_norm: float) -> None:
        self.examples += int(count)
        self.batches += 1
        for key in self.totals:
            value = gradient_norm if key == "gradient_norm" else stats[key]
            self.totals[key] += float(value) * int(count)

    def summary(self) -> dict[str, Any]:
        divisor = max(1, self.examples)
        return {
            "examples": self.examples,
            "batches": self.batches,
            **{
                key: round(value / divisor, 7)
                for key, value in self.totals.items()
            },
        }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
