from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .neural_model import (
    EncodedDecision,
    NeuralPolicy,
    collate_decisions,
    encode_decision,
    load_neural_checkpoint,
    neural_state_dict_fingerprint,
    require_torch,
    resolve_device,
    torch,
)
from .neural_training import iter_shuffled_batches, iter_trajectory_episodes
from .protocol import Action


def inspect_on_policy_trajectories(
    paths: Iterable[str | Path],
    *,
    expected_policy_fingerprint: str,
    skip_recovered_episodes: bool = True,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> dict[str, Any]:
    _validate_gae_parameters(gamma, gae_lambda)
    rulesets: set[str] = set()
    episodes = 0
    decisions = 0
    skipped_recovered = 0
    skipped_truncated = 0
    skipped_without_actor = 0
    ignored_opponent_decisions = 0
    advantage_sum = 0.0
    advantage_square_sum = 0.0
    entropies = 0.0
    for episode in iter_trajectory_episodes(paths):
        if bool(episode.get("truncated")) or not bool(episode.get("terminated")):
            skipped_truncated += 1
            continue
        if skip_recovered_episodes and _as_int(episode.get("loop_recoveries"), 0) > 0:
            skipped_recovered += 1
            continue
        ruleset = str(episode.get("ruleset_fingerprint") or "")
        if ruleset:
            rulesets.add(ruleset)
        fingerprints = [str(value or "") for value in episode.get("policy_fingerprints") or []]
        if len(fingerprints) != 2:
            raise ValueError("on-policy episode is missing its two policy fingerprints")
        actor_seats = {
            seat for seat, value in enumerate(fingerprints)
            if value == expected_policy_fingerprint
        }
        if not actor_seats:
            skipped_without_actor += 1
            continue
        episodes += 1
        targets = _episode_actor_targets(
            episode,
            actor_seats,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for index, decision in enumerate(episode.get("decisions") or []):
            player = _as_int(decision.get("player"), -1)
            if player not in actor_seats:
                ignored_opponent_decisions += 1
                continue
            if decision.get("forced_fallback"):
                raise ValueError("on-policy episode contains a forced fallback decision")
            metadata = _behavior_metadata(decision)
            advantage, _ = targets[index]
            advantage_sum += advantage
            advantage_square_sum += advantage * advantage
            entropies += metadata["entropy"]
            decisions += 1
    if len(rulesets) != 1:
        if not rulesets:
            raise ValueError("on-policy data contains no usable ruleset")
        raise ValueError("on-policy data mixes different game rulesets")
    if decisions == 0:
        raise ValueError("on-policy data contains no usable decisions")
    mean = advantage_sum / decisions
    variance = max(1e-8, advantage_square_sum / decisions - mean * mean)
    return {
        "episodes": episodes,
        "decisions": decisions,
        "ruleset_fingerprint": next(iter(rulesets)),
        "policy_fingerprint": expected_policy_fingerprint,
        "skipped_recovered_episodes": skipped_recovered,
        "skipped_truncated_episodes": skipped_truncated,
        "skipped_episodes_without_actor": skipped_without_actor,
        "ignored_opponent_decisions": ignored_opponent_decisions,
        "advantage_mean": mean,
        "advantage_std": math.sqrt(variance),
        "mean_behavior_entropy": entropies / decisions,
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
    }


def iter_on_policy_decisions(
    paths: Iterable[str | Path],
    *,
    config,
    expected_policy_fingerprint: str,
    skip_recovered_episodes: bool = True,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> Iterator[EncodedDecision]:
    _validate_gae_parameters(gamma, gae_lambda)
    for episode in iter_trajectory_episodes(paths):
        if bool(episode.get("truncated")) or not bool(episode.get("terminated")):
            continue
        if skip_recovered_episodes and _as_int(episode.get("loop_recoveries"), 0) > 0:
            continue
        fingerprints = [str(value or "") for value in episode.get("policy_fingerprints") or []]
        if len(fingerprints) != 2:
            raise ValueError("on-policy episode is missing its two policy fingerprints")
        actor_seats = {
            seat for seat, value in enumerate(fingerprints)
            if value == expected_policy_fingerprint
        }
        if not actor_seats:
            continue
        targets = _episode_actor_targets(
            episode,
            actor_seats,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for index, decision in enumerate(episode.get("decisions") or []):
            player = _as_int(decision.get("player"), -1)
            if player not in actor_seats:
                continue
            if decision.get("forced_fallback"):
                raise ValueError("on-policy episode contains a forced fallback decision")
            metadata = _behavior_metadata(decision)
            try:
                selected = Action.from_dict(decision.get("action") or {})
                legal = [Action.from_dict(item) for item in decision.get("legal_actions") or []]
            except (TypeError, ValueError):
                continue
            selected_index = next(
                (index for index, action in enumerate(legal) if action.key == selected.key),
                None,
            )
            observation = decision.get("observation")
            if selected_index is None or not legal or not isinstance(observation, dict):
                continue
            advantage, target = targets[index]
            yield encode_decision(
                observation,
                legal,
                config=config,
                selected_index=selected_index,
                value_target=target,
                behavior_log_prob=metadata["log_prob"],
                behavior_value=metadata["value"],
                behavior_temperature=metadata["temperature"],
                advantage_target=advantage,
            )


def generalized_advantage_targets(
    behavior_values: Sequence[float],
    terminal_result: float,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> list[tuple[float, float]]:
    """Return (advantage, lambda-return) for one actor's decision sequence."""

    _validate_gae_parameters(gamma, gae_lambda)
    values = [max(-1.0, min(1.0, float(value))) for value in behavior_values]
    if not values:
        return []
    terminal = max(-1.0, min(1.0, float(terminal_result)))
    result = [(0.0, 0.0)] * len(values)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(values) - 1, -1, -1):
        reward = terminal if index == len(values) - 1 else 0.0
        delta = reward + float(gamma) * next_value - values[index]
        advantage = delta + float(gamma) * float(gae_lambda) * next_advantage
        value_target = max(-1.0, min(1.0, values[index] + advantage))
        result[index] = (advantage, value_target)
        next_value = values[index]
        next_advantage = advantage
    return result


def _episode_actor_targets(
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
        if decision.get("forced_fallback"):
            raise ValueError("on-policy episode contains a forced fallback decision")
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


def _validate_gae_parameters(gamma: float, gae_lambda: float) -> None:
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")


def train_actor_critic_policy(
    paths: Sequence[str | Path],
    *,
    initial_checkpoint: str | Path,
    epochs: int = 2,
    batch_size: int = 256,
    shuffle_buffer: int = 8192,
    learning_rate: float = 3e-5,
    weight_decay: float = 1e-5,
    clip_ratio: float = 0.15,
    value_clip: float = 0.15,
    value_loss_weight: float = 0.25,
    entropy_weight: float = 0.004,
    max_grad_norm: float = 1.0,
    target_kl: float = 0.03,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    device: str = "auto",
    seed: int = 1,
    skip_recovered_episodes: bool = True,
) -> tuple[NeuralPolicy, dict[str, Any]]:
    require_torch()
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("at least one trajectory file is required")
    resolved_device = resolve_device(device)
    initial = load_neural_checkpoint(initial_checkpoint, device=resolved_device)
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
    model.eval()  # Keep behavior-logit comparisons exact by disabling dropout.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    epoch_metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    stopped_early = False
    for epoch in range(max(0, int(epochs))):
        accumulator = _PpoAccumulator()
        examples = iter_on_policy_decisions(
            files,
            config=initial["config"],
            expected_policy_fingerprint=initial_fingerprint,
            skip_recovered_episodes=skip_recovered_episodes,
            gamma=gamma,
            gae_lambda=gae_lambda,
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
            loss, stats = _ppo_loss(
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
                raise RuntimeError("actor-critic training produced a non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(max_grad_norm)
            )
            optimizer.step()
            accumulator.add(len(encoded), stats, float(gradient_norm.detach().to("cpu").item()))
        summary = accumulator.summary()
        epoch_metrics.append({"epoch": epoch + 1, **summary})
        if summary["examples"] == 0:
            raise ValueError("on-policy training split contains no usable decisions")
        if target_kl > 0 and summary["approx_kl"] > float(target_kl):
            stopped_early = True
            break
    elapsed = time.perf_counter() - started
    policy = NeuralPolicy(
        model,
        config=initial["config"],
        ruleset_fingerprint=dataset["ruleset_fingerprint"],
        device=resolved_device,
        seed=seed,
        name="actor-critic-candidate",
    )
    return policy, {
        "objective": "clipped_on_policy_actor_critic",
        "device": resolved_device,
        "epochs_requested": max(0, int(epochs)),
        "epochs_completed": len(epoch_metrics),
        "stopped_early": stopped_early,
        "seconds": round(elapsed, 3),
        "dataset": dataset,
        "epoch_metrics": epoch_metrics,
        "initial_checkpoint": str(Path(initial_checkpoint).resolve()),
        "initial_policy_fingerprint": initial_fingerprint,
        "result_policy_fingerprint": policy.model_fingerprint,
        "hyperparameters": {
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
        },
    }


def _ppo_loss(
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
    new_log_probs = []
    entropies = []
    offsets = batch["action_set_offsets"].detach().to("cpu").tolist()
    selected = batch["selected_absolute"].detach().to("cpu").tolist()
    for index in range(len(offsets) - 1):
        start, end = int(offsets[index]), int(offsets[index + 1])
        target = int(selected[index]) - start
        if target < 0 or start >= end:
            raise ValueError("batch contains an invalid selected action")
        temperature = batch["behavior_temperatures"][index].clamp_min(1e-6)
        logits = scores[start:end] + batch["action_logit_biases"][start:end]
        normalized = ((logits - logits.max()) / temperature).clamp_min(-60.0)
        log_probs = torch.nn.functional.log_softmax(normalized, dim=0)
        probabilities = log_probs.exp()
        new_log_probs.append(log_probs[target])
        entropies.append(-(probabilities * log_probs).sum())
    new_log_prob = torch.stack(new_log_probs)
    old_log_prob = batch["behavior_log_probs"]
    raw_advantage = batch["advantage_targets"]
    advantages = (raw_advantage - float(advantage_mean)) / max(1e-6, float(advantage_std))
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
    entropy = torch.stack(entropies).mean()
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


class _PpoAccumulator:
    def __init__(self):
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
            **{key: round(value / divisor, 7) for key, value in self.totals.items()},
        }


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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
