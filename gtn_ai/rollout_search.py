from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .belief_sampling import BeliefSamplingError
from .environment import Garden1v1Env
from .protocol import Action
from .structured_model import StructuredPolicy


@dataclass(frozen=True)
class UnsafeRolloutConfig:
    """Small offline search used for upper-bound and belief diagnostics."""

    candidates: int = 3
    rollouts: int = 2
    horizon: int = 8
    rollout_exploration: float = 0.08
    prior_weight: float = 0.02
    determinize_hidden: bool = False
    teacher_logit_scale: float = 4.0
    annotate_only: bool = False
    max_rollouts: int | None = None
    confidence_margin: float = 0.0
    rollout_batch: int = 2
    base_margin_gate: float | None = None
    belief_deck_prior_path: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= int(self.candidates) <= 32:
            raise ValueError("search candidates must be in [1, 32]")
        if not 1 <= int(self.rollouts) <= 64:
            raise ValueError("search rollouts must be in [1, 64]")
        maximum = int(self.max_rollouts) if self.max_rollouts is not None else int(self.rollouts)
        if not int(self.rollouts) <= maximum <= 64:
            raise ValueError("search max rollouts must be in [rollouts, 64]")
        if not 1 <= int(self.rollout_batch) <= 64:
            raise ValueError("search rollout batch must be in [1, 64]")
        if float(self.confidence_margin) < 0.0:
            raise ValueError("search confidence margin must not be negative")
        if self.base_margin_gate is not None and float(self.base_margin_gate) < 0.0:
            raise ValueError("search base margin gate must not be negative")
        if not 0 <= int(self.horizon) <= 128:
            raise ValueError("search horizon must be in [0, 128]")
        if not 0.0 <= float(self.rollout_exploration) <= 1.0:
            raise ValueError("rollout exploration must be in [0, 1]")
        if float(self.prior_weight) < 0.0:
            raise ValueError("search prior weight must not be negative")
        if not 0.0 < float(self.teacher_logit_scale) <= 100.0:
            raise ValueError("teacher logit scale must be in (0, 100]")
        if self.belief_deck_prior_path and not self.determinize_hidden:
            raise ValueError("deck prior requires belief determinization")


class UnsafeFullStateRolloutPolicy:
    """Offline diagnostic search over cloned production-engine states.

    Full-state mode intentionally sees private cards and only measures an upper bound.
    Belief mode replaces unknown opponent cards with public-observation-equivalent
    samples before every rollout. Both modes remain offline-only until every private
    production-engine field has a reviewed belief representation.
    """

    offline_only = True
    uses_private_engine_state = True

    def __init__(
        self,
        base_policy: StructuredPolicy,
        *,
        config: UnsafeRolloutConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.base_policy = base_policy
        self.config = config or UnsafeRolloutConfig()
        self.seed = int(seed)
        self._belief_deck_prior = None
        if self.config.belief_deck_prior_path:
            from .deck_prior import DeckPrior

            self._belief_deck_prior = DeckPrior.load(
                self.config.belief_deck_prior_path
            )
        self.name = (
            f"OFFLINE-{'belief+deckprior' if self._belief_deck_prior else 'belief' if self.config.determinize_hidden else 'fullstate'}-"
            f"{'annotator' if self.config.annotate_only else 'rollout'}["
            f"{self.config.candidates}x"
            f"{self.config.rollouts}"
            f"{'-' + str(self.config.max_rollouts) if self.config.max_rollouts else ''}"
            f"x{self.config.horizon}]"
        )
        self.uses_private_engine_state = not bool(self.config.determinize_hidden)
        self.ruleset_fingerprint = base_policy.ruleset_fingerprint
        # Deliberately blank: strict on-policy trainers must reject these trajectories.
        self.model_fingerprint = ""
        self.last_decision_metadata = None
        self.last_teacher_metadata: dict[str, Any] | None = None
        self.last_search_metadata: dict[str, Any] | None = None
        self._decisions = 0
        self._searched_decisions = 0
        self._bypassed_decisions = 0
        self._confidence_bypassed_decisions = 0
        self._candidate_evaluations = 0
        self._rollouts = 0
        self._rollout_steps = 0
        self._terminal_rollouts = 0
        self._leaf_evaluations = 0
        self._action_changes = 0
        self._belief_samples = 0
        self._belief_sampled_cards = 0
        self._belief_history_constraints = 0
        self._belief_sampling_attempts = 0
        self._belief_failures = 0
        self._belief_deck_prior_samples = 0
        self._belief_deck_prior_cards = 0
        self._belief_exact_prior_samples = 0
        self._seconds = 0.0

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> Action:
        raise RuntimeError(
            "offline rollout search requires an engine clone and "
            "cannot be used by the inference service"
        )

    def select_action_with_env(
        self,
        env: Garden1v1Env,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
        player_id: int,
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("search policy received an empty legal action list")
        if env.decision_player() != int(player_id):
            raise ValueError("search policy received an environment for another actor")

        self.last_decision_metadata = None
        self.last_teacher_metadata = None
        self.last_search_metadata = None
        self._decisions += 1
        if observation.get("phase") == "pregame" or len(actions) == 1:
            self._bypassed_decisions += 1
            return self.base_policy.select_action(observation, actions)

        started = time.perf_counter()
        logits, _ = self.base_policy.evaluate_actions(observation, actions)
        ranked = sorted(range(len(actions)), key=lambda index: (-logits[index], index))
        base_margin = (
            float(logits[ranked[0]]) - float(logits[ranked[1]])
            if len(ranked) > 1
            else math.inf
        )
        if (
            self.config.base_margin_gate is not None
            and base_margin > float(self.config.base_margin_gate)
        ):
            self._bypassed_decisions += 1
            self._confidence_bypassed_decisions += 1
            self._seconds += time.perf_counter() - started
            return actions[ranked[0]]
        candidate_indices = ranked[: min(len(ranked), int(self.config.candidates))]
        prior_probabilities = _softmax(logits)
        candidate_returns: list[list[float | None]] = [
            [] for _ in candidate_indices
        ]
        initial_rollouts = int(self.config.rollouts)
        maximum_rollouts = (
            int(self.config.max_rollouts)
            if self.config.max_rollouts is not None
            else initial_rollouts
        )
        rollouts_used = 0
        scored: list[tuple[float, int, int]] = []
        metadata_candidates: list[dict[str, Any]] = []
        score_margin = math.inf

        try:
            target_rollouts = initial_rollouts
            while True:
                self._append_rollout_batch(
                    env,
                    actions,
                    candidate_indices,
                    candidate_returns,
                    root_player=int(player_id),
                    start_rollout=rollouts_used,
                    stop_rollout=target_rollouts,
                )
                rollouts_used = target_rollouts
                scored, metadata_candidates = self._score_candidates(
                    actions,
                    candidate_indices,
                    candidate_returns,
                    prior_probabilities,
                    expected_rollouts=rollouts_used,
                )
                ordered_scores = sorted((entry[0] for entry in scored), reverse=True)
                score_margin = (
                    ordered_scores[0] - ordered_scores[1]
                    if len(ordered_scores) > 1
                    else math.inf
                )
                if (
                    rollouts_used >= maximum_rollouts
                    or score_margin >= float(self.config.confidence_margin)
                ):
                    break
                target_rollouts = min(
                    maximum_rollouts,
                    rollouts_used + int(self.config.rollout_batch),
                )
        except BeliefSamplingError:
            self._belief_failures += 1
            if rollouts_used == 0:
                self._bypassed_decisions += 1
                self._seconds += time.perf_counter() - started
                return actions[ranked[0]]

        selected_index = max(scored)[2]
        self._searched_decisions += 1
        self._candidate_evaluations += len(candidate_indices)
        self._action_changes += int(selected_index != ranked[0])
        elapsed = time.perf_counter() - started
        self._seconds += elapsed
        self.last_search_metadata = {
            "base_action_key": actions[ranked[0]].key,
            "selected_action_key": actions[selected_index].key,
            "executed_action_key": actions[
                ranked[0] if self.config.annotate_only else selected_index
            ].key,
            "changed": selected_index != ranked[0],
            "seconds": round(elapsed, 6),
            "rollouts_used": rollouts_used,
            "score_margin": round(score_margin, 7),
            "candidates": metadata_candidates,
        }
        if self.config.determinize_hidden:
            candidate_scores = {
                action_index: score
                for score, _, action_index in scored
            }
            floor = min(candidate_scores.values()) - 0.5
            teacher_scores = [
                float(candidate_scores.get(index, floor))
                for index in range(len(actions))
            ]
            # Preserve the exact deterministic tie break used by the search policy.
            teacher_scores[selected_index] += 1e-6
            peak = max(teacher_scores)
            scale = float(self.config.teacher_logit_scale)
            selected_position = candidate_indices.index(selected_index)
            selected_values = [
                float(value)
                for value in candidate_returns[selected_position]
                if value is not None
            ]
            teacher_value = sum(selected_values) / len(selected_values)
            self.last_teacher_metadata = {
                "kind": "belief_rollout_v1",
                "action_key": actions[selected_index].key,
                "logits": [round((score - peak) * scale, 7) for score in teacher_scores],
                "value": round(max(-1.0, min(1.0, teacher_value)), 7),
                "search_margin": round(score_margin, 7),
                "rollouts": rollouts_used,
            }
        return actions[ranked[0] if self.config.annotate_only else selected_index]

    def _append_rollout_batch(
        self,
        env: Garden1v1Env,
        actions: Sequence[Action],
        candidate_indices: Sequence[int],
        candidate_returns: list[list[float | None]],
        *,
        root_player: int,
        start_rollout: int,
        stop_rollout: int,
    ) -> None:
        rollout_roots: list[Garden1v1Env] = []
        if self.config.determinize_hidden:
            from .belief_sampling import determinize_hidden_cards

            for rollout_index in range(start_rollout, stop_rollout):
                root = env.clone()
                summary = determinize_hidden_cards(
                    root,
                    root_player,
                    seed=self._belief_seed(rollout_index),
                    deck_prior=self._belief_deck_prior,
                )
                rollout_roots.append(root)
                self._belief_samples += 1
                self._belief_sampled_cards += summary.sampled_cards
                self._belief_history_constraints += summary.history_constrained_cards
                self._belief_sampling_attempts += summary.attempts
                if summary.deck_prior_source != "native":
                    self._belief_deck_prior_samples += 1
                    self._belief_deck_prior_cards += summary.deck_prior_sampled_cards
                if summary.deck_prior_source == "exact":
                    self._belief_exact_prior_samples += 1
        else:
            rollout_roots = [env] * (stop_rollout - start_rollout)

        leaves: list[_Leaf] = []
        for candidate_position, action_index in enumerate(candidate_indices):
            returns = candidate_returns[candidate_position]
            for root_offset, rollout_index in enumerate(range(start_rollout, stop_rollout)):
                outcome = self._rollout(
                    rollout_roots[root_offset],
                    actions[action_index],
                    root_player=root_player,
                    action_index=action_index,
                    rollout_index=rollout_index,
                )
                self._rollouts += 1
                self._rollout_steps += outcome.steps
                if outcome.terminal_value is not None:
                    self._terminal_rollouts += 1
                    returns.append(outcome.terminal_value)
                else:
                    returns.append(None)
                    leaves.append(_Leaf(
                        candidate_position=candidate_position,
                        return_position=len(returns) - 1,
                        actor=outcome.actor,
                        observation=outcome.observation,
                        legal_actions=outcome.legal_actions,
                    ))

        if not leaves:
            return
        values = self.base_policy.estimate_values([
            (leaf.observation, leaf.legal_actions) for leaf in leaves
        ])
        self._leaf_evaluations += len(leaves)
        for leaf, value in zip(leaves, values):
            root_value = value if leaf.actor == root_player else -value
            candidate_returns[leaf.candidate_position][leaf.return_position] = root_value

    def _score_candidates(
        self,
        actions: Sequence[Action],
        candidate_indices: Sequence[int],
        candidate_returns: Sequence[Sequence[float | None]],
        prior_probabilities: Sequence[float],
        *,
        expected_rollouts: int,
    ) -> tuple[list[tuple[float, int, int]], list[dict[str, Any]]]:
        scored: list[tuple[float, int, int]] = []
        metadata_candidates: list[dict[str, Any]] = []
        for position, action_index in enumerate(candidate_indices):
            values = [float(value) for value in candidate_returns[position] if value is not None]
            if len(values) != expected_rollouts:
                raise RuntimeError("rollout search did not resolve every candidate value")
            rollout_mean = sum(values) / len(values)
            score = rollout_mean + float(self.config.prior_weight) * prior_probabilities[action_index]
            scored.append((score, -position, action_index))
            metadata_candidates.append({
                "action_key": actions[action_index].key,
                "rollout_mean": round(rollout_mean, 6),
                "score": round(score, 6),
                "returns": [round(value, 6) for value in values],
            })
        return scored, metadata_candidates

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "offline_only": True,
            "config": asdict(self.config),
            "decisions": self._decisions,
            "searched_decisions": self._searched_decisions,
            "bypassed_decisions": self._bypassed_decisions,
            "confidence_bypassed_decisions": self._confidence_bypassed_decisions,
            "candidate_evaluations": self._candidate_evaluations,
            "rollouts": self._rollouts,
            "rollout_steps": self._rollout_steps,
            "terminal_rollouts": self._terminal_rollouts,
            "leaf_evaluations": self._leaf_evaluations,
            "action_changes": self._action_changes,
            "belief_samples": self._belief_samples,
            "belief_sampled_cards": self._belief_sampled_cards,
            "belief_history_constraints": self._belief_history_constraints,
            "belief_sampling_attempts": self._belief_sampling_attempts,
            "belief_failures": self._belief_failures,
            "belief_deck_prior_samples": self._belief_deck_prior_samples,
            "belief_deck_prior_cards": self._belief_deck_prior_cards,
            "belief_exact_prior_samples": self._belief_exact_prior_samples,
            "belief_deck_prior_fingerprint": (
                self._belief_deck_prior.fingerprint if self._belief_deck_prior else ""
            ),
            "seconds": round(self._seconds, 6),
        }

    def _belief_seed(self, rollout_index: int) -> int:
        return (
            self.seed
            ^ (self._decisions * 0x9E3779B1)
            ^ (int(rollout_index) * 0xC2B2AE3D)
        ) & 0x7FFFFFFF

    def _rollout(
        self,
        env: Garden1v1Env,
        root_action: Action,
        *,
        root_player: int,
        action_index: int,
        rollout_index: int,
    ) -> "_RolloutOutcome":
        from .policies import HeuristicPolicy

        branch = env.clone()
        branch.step(root_action, root_player)
        steps = 1
        rollout_seed = (
            self.seed
            ^ (self._decisions * 0x9E3779B1)
            ^ (int(action_index) * 0x85EBCA77)
            ^ (int(rollout_index) * 0xC2B2AE3D)
        ) & 0x7FFFFFFF
        policies = (
            HeuristicPolicy(
                seed=rollout_seed * 2,
                exploration=float(self.config.rollout_exploration),
            ),
            HeuristicPolicy(
                seed=rollout_seed * 2 + 1,
                exploration=float(self.config.rollout_exploration),
            ),
        )
        for _ in range(int(self.config.horizon)):
            if branch.engine.game_over:
                break
            actor = branch.decision_player()
            if actor not in (0, 1):
                raise RuntimeError("rollout reached a non-terminal state without an actor")
            observation = branch.observe(actor)
            legal = branch.legal_actions(actor)
            if not legal:
                raise RuntimeError("rollout reached a non-terminal state without legal actions")
            action = policies[actor].select_action(observation, legal)
            branch.step(action, actor)
            steps += 1

        if branch.engine.game_over:
            winner = int(branch.engine.winner)
            value = 0.0 if winner not in (0, 1) else (1.0 if winner == root_player else -1.0)
            return _RolloutOutcome(steps=steps, terminal_value=value)
        actor = branch.decision_player()
        if actor not in (0, 1):
            raise RuntimeError("rollout leaf has no decision actor")
        observation = branch.observe(actor)
        legal = branch.legal_actions(actor)
        if not legal:
            raise RuntimeError("rollout leaf has no legal actions")
        return _RolloutOutcome(
            steps=steps,
            actor=actor,
            observation=observation,
            legal_actions=legal,
        )


@dataclass(frozen=True)
class _RolloutOutcome:
    steps: int
    terminal_value: float | None = None
    actor: int = -1
    observation: dict[str, Any] | None = None
    legal_actions: Sequence[Action] = ()


@dataclass(frozen=True)
class _Leaf:
    candidate_position: int
    return_position: int
    actor: int
    observation: dict[str, Any]
    legal_actions: Sequence[Action]


def parse_unsafe_rollout_spec(spec: str) -> tuple[str, UnsafeRolloutConfig]:
    """Parse CHECKPOINT;candidates=N;rollouts=N;horizon=N style policy specs."""

    parts = [part.strip() for part in str(spec).split(";")]
    checkpoint = parts[0]
    if not checkpoint:
        raise ValueError("unsafe rollout policy requires a structured checkpoint")
    values: dict[str, Any] = {}
    converters = {
        "candidates": int,
        "rollouts": int,
        "horizon": int,
        "exploration": float,
        "prior": float,
        "belief": _parse_bool,
        "teacher-scale": float,
        "annotate": _parse_bool,
        "max-rollouts": int,
        "confidence": float,
        "batch": int,
        "gate": float,
        "deck-prior": str,
    }
    names = {
        "exploration": "rollout_exploration",
        "prior": "prior_weight",
        "belief": "determinize_hidden",
        "teacher-scale": "teacher_logit_scale",
        "annotate": "annotate_only",
        "max-rollouts": "max_rollouts",
        "confidence": "confidence_margin",
        "batch": "rollout_batch",
        "gate": "base_margin_gate",
        "deck-prior": "belief_deck_prior_path",
    }
    for option in parts[1:]:
        if not option:
            continue
        key, separator, raw_value = option.partition("=")
        normalized = key.strip().lower()
        if not separator or normalized not in converters:
            raise ValueError(f"unknown unsafe rollout option: {option}")
        values[names.get(normalized, normalized)] = converters[normalized](raw_value.strip())
    return checkpoint, UnsafeRolloutConfig(**values)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(float(value) for value in values)
    exponentials = [math.exp(max(-40.0, min(40.0, float(value) - peak))) for value in values]
    total = max(1e-12, sum(exponentials))
    return [value / total for value in exponentials]


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")
