from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .features import FEATURE_SCHEMA_VERSION, dot, hashed_action_features
from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, Action


LINEAR_MODEL_SCHEMA_VERSION = 1


@dataclass
class HashedLinearPolicy:
    weights: dict[int, float] = field(default_factory=dict)
    buckets: int = 1 << 16
    seed: int = 0
    ruleset_fingerprint: str = ""
    name: str = "hashed-linear-v1"

    def __post_init__(self) -> None:
        self._rng = random.Random(int(self.seed))

    def score(self, observation: dict[str, Any], action: Action) -> float:
        return dot(self.weights, hashed_action_features(observation, action, buckets=self.buckets))

    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        observed_fingerprint = str(
            (observation.get("loadout") or {}).get("ruleset_fingerprint") or ""
        )
        if (
            self.ruleset_fingerprint
            and observed_fingerprint != self.ruleset_fingerprint
        ):
            raise ValueError(
                "model ruleset fingerprint does not match the current game rules"
            )
        scored = [(self.score(observation, action), action) for action in actions]
        best_score = max(score for score, _ in scored)
        best = [action for score, action in scored if abs(score - best_score) <= 1e-12]
        return self._rng.choice(best)

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": LINEAR_MODEL_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "name": self.name,
            "buckets": int(self.buckets),
            "ruleset_fingerprint": self.ruleset_fingerprint,
            "weights": {str(index): round(value, 10) for index, value in sorted(self.weights.items()) if abs(value) > 1e-12},
            "metadata": dict(metadata or {}),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0) -> "HashedLinearPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != LINEAR_MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported linear model schema")
        if int(payload.get("feature_schema_version", -1)) != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported feature schema")
        if int(payload.get("observation_schema_version", -1)) != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported observation schema")
        if int(payload.get("action_schema_version", -1)) != ACTION_SCHEMA_VERSION:
            raise ValueError("unsupported action schema")
        return cls(
            weights={int(index): float(value) for index, value in (payload.get("weights") or {}).items()},
            buckets=int(payload.get("buckets", 1 << 16)),
            seed=seed,
            ruleset_fingerprint=str(payload.get("ruleset_fingerprint") or ""),
            name=str(payload.get("name") or "hashed-linear-v1"),
        )


def train_monte_carlo(
    episodes: Sequence[dict[str, Any]],
    *,
    buckets: int = 1 << 16,
    epochs: int = 3,
    learning_rate: float = 0.04,
    l2: float = 1e-6,
    seed: int = 0,
) -> tuple[HashedLinearPolicy, dict[str, Any]]:
    rng = random.Random(int(seed))
    weights: dict[int, float] = {}
    examples: list[tuple[dict[str, Any], Action, float]] = []
    fingerprints = set()
    for episode in episodes:
        if episode.get("truncated") or not episode.get("terminated"):
            continue
        winner = _as_int(episode.get("winner"), -1)
        fingerprint = str(episode.get("ruleset_fingerprint") or "")
        if fingerprint:
            fingerprints.add(fingerprint)
        for decision in episode.get("decisions") or []:
            if decision.get("forced_fallback"):
                continue
            player = _as_int(decision.get("player"), -1)
            target = 0.0 if winner not in (0, 1) else (1.0 if winner == player else -1.0)
            try:
                action = Action.from_dict(decision.get("action") or {})
            except (TypeError, ValueError):
                continue
            observation = decision.get("observation")
            if isinstance(observation, dict):
                examples.append((observation, action, target))
    losses = []
    for _ in range(max(0, int(epochs))):
        rng.shuffle(examples)
        loss_sum = 0.0
        for observation, action, target in examples:
            features = hashed_action_features(observation, action, buckets=buckets)
            prediction = math.tanh(dot(weights, features))
            error = target - prediction
            loss_sum += error * error
            gradient_scale = float(learning_rate) * error * (1.0 - prediction * prediction)
            for index, value in features.items():
                old = weights.get(index, 0.0)
                updated = old + gradient_scale * value - float(learning_rate) * float(l2) * old
                if abs(updated) > 1e-12:
                    weights[index] = updated
                elif index in weights:
                    del weights[index]
        losses.append(loss_sum / max(1, len(examples)))
    fingerprint = _require_single_ruleset(fingerprints, examples)
    policy = HashedLinearPolicy(weights=weights, buckets=buckets, seed=seed, ruleset_fingerprint=fingerprint)
    metrics = {
        "examples": len(examples),
        "epochs": max(0, int(epochs)),
        "losses": [round(value, 6) for value in losses],
        "nonzero_weights": len(weights),
        "ruleset_fingerprints": sorted(fingerprints),
    }
    return policy, metrics


def train_behavior_cloning(
    episodes: Sequence[dict[str, Any]],
    *,
    buckets: int = 1 << 16,
    epochs: int = 3,
    learning_rate: float = 0.03,
    l2: float = 1e-6,
    seed: int = 0,
) -> tuple[HashedLinearPolicy, dict[str, Any]]:
    """Fit a masked softmax policy to recorded choices and their legal sets."""

    rng = random.Random(int(seed))
    weights: dict[int, float] = {}
    examples: list[tuple[dict[str, Any], list[Action], int]] = []
    fingerprints = set()
    skipped = 0
    for episode in episodes:
        fingerprint = str(episode.get("ruleset_fingerprint") or "")
        if fingerprint:
            fingerprints.add(fingerprint)
        for decision in episode.get("decisions") or []:
            if decision.get("forced_fallback"):
                skipped += 1
                continue
            observation = decision.get("observation")
            try:
                selected = Action.from_dict(decision.get("action") or {})
                legal = [Action.from_dict(item) for item in decision.get("legal_actions") or []]
            except (TypeError, ValueError):
                skipped += 1
                continue
            selected_index = next(
                (index for index, action in enumerate(legal) if action.key == selected.key),
                None,
            )
            if not isinstance(observation, dict) or selected_index is None or not legal:
                skipped += 1
                continue
            examples.append((observation, legal, selected_index))

    losses = []
    accuracies = []
    phase_examples = Counter(
        "pregame" if observation.get("phase") == "pregame" else "combat"
        for observation, _, _ in examples
    )
    phase_accuracies: dict[str, list[float]] = {
        phase: [] for phase in sorted(phase_examples)
    }
    for _ in range(max(0, int(epochs))):
        rng.shuffle(examples)
        loss_sum = 0.0
        correct = 0
        phase_correct = Counter()
        phase_total = Counter()
        for observation, legal, selected_index in examples:
            vectors = [
                hashed_action_features(observation, action, buckets=buckets)
                for action in legal
            ]
            scores = [dot(weights, vector) for vector in vectors]
            peak = max(scores)
            exponentials = [math.exp(max(-40.0, min(40.0, score - peak))) for score in scores]
            normalizer = max(1e-12, sum(exponentials))
            probabilities = [value / normalizer for value in exponentials]
            loss_sum -= math.log(max(1e-12, probabilities[selected_index]))
            predicted = max(range(len(scores)), key=scores.__getitem__)
            correct += int(predicted == selected_index)
            phase = "pregame" if observation.get("phase") == "pregame" else "combat"
            phase_correct[phase] += int(predicted == selected_index)
            phase_total[phase] += 1

            gradients: dict[int, float] = {}
            for action_index, vector in enumerate(vectors):
                coefficient = (1.0 if action_index == selected_index else 0.0) - probabilities[action_index]
                for feature_index, value in vector.items():
                    gradients[feature_index] = gradients.get(feature_index, 0.0) + coefficient * value
            for feature_index, gradient in gradients.items():
                old = weights.get(feature_index, 0.0)
                updated = old + float(learning_rate) * gradient - float(learning_rate) * float(l2) * old
                if abs(updated) > 1e-12:
                    weights[feature_index] = updated
                elif feature_index in weights:
                    del weights[feature_index]
        losses.append(loss_sum / max(1, len(examples)))
        accuracies.append(correct / max(1, len(examples)))
        for phase in phase_accuracies:
            phase_accuracies[phase].append(
                phase_correct[phase] / max(1, phase_total[phase])
            )

    fingerprint = _require_single_ruleset(fingerprints, examples)
    policy = HashedLinearPolicy(
        weights=weights,
        buckets=buckets,
        seed=seed,
        ruleset_fingerprint=fingerprint,
        name="hashed-linear-bc-v1",
    )
    metrics = {
        "objective": "behavior_cloning",
        "examples": len(examples),
        "skipped_decisions": skipped,
        "epochs": max(0, int(epochs)),
        "losses": [round(value, 6) for value in losses],
        "accuracies": [round(value, 6) for value in accuracies],
        "phase_examples": dict(sorted(phase_examples.items())),
        "phase_accuracies": {
            phase: [round(value, 6) for value in values]
            for phase, values in sorted(phase_accuracies.items())
        },
        "nonzero_weights": len(weights),
        "ruleset_fingerprints": sorted(fingerprints),
    }
    return policy, metrics


def _require_single_ruleset(fingerprints: set[str], examples: Sequence[Any]) -> str:
    if not examples:
        raise ValueError("training data contains no usable decisions")
    if not fingerprints:
        raise ValueError("training data has no ruleset fingerprint")
    if len(fingerprints) != 1:
        raise ValueError(
            "training data mixes different game rulesets; split or regenerate the dataset"
        )
    return next(iter(fingerprints))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
