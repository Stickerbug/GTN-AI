from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .protocol import Action


_NEURAL_TEMPLATE_CACHE: dict[tuple[str, str], Any] = {}
_STRUCTURED_TEMPLATE_CACHE: dict[tuple[str, str], Any] = {}
_CORRECTION_TEMPLATE_CACHE: dict[tuple[str, str], Any] = {}
_EXPERIENCE_PRIOR_CACHE: dict[str, Any] = {}
_DECK_PRIOR_CACHE: dict[str, Any] = {}


class Policy(Protocol):
    name: str

    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action:
        ...


@dataclass
class RandomPolicy:
    seed: int = 0
    name: str = "random-v1"

    def __post_init__(self) -> None:
        self._rng = random.Random(int(self.seed))

    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("policy received an empty legal action list")
        return self._rng.choice(list(legal_actions))


@dataclass
class HeuristicPolicy:
    """Small deterministic baseline; deliberately replaceable by a learned scorer."""

    seed: int = 0
    exploration: float = 0.04
    name: str = "heuristic-v1"

    def __post_init__(self) -> None:
        self._rng = random.Random(int(self.seed))

    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        if self.exploration > 0 and self._rng.random() < self.exploration:
            return self._rng.choice(actions)
        scores = [(self._score(observation, action), -index, action) for index, action in enumerate(actions)]
        best_score = max(item[0] for item in scores)
        best = [item[2] for item in scores if item[0] == best_score]
        return self._rng.choice(best)

    def _score(self, observation: dict[str, Any], action: Action) -> float:
        kind = action.kind
        payload = action.payload
        if kind == "select_opening_event":
            return 20.0
        if kind == "reroll_opening_event":
            return 4.0
        if kind == "confirm_opening_reveal":
            return 40.0
        if kind == "draft_pick":
            card = _find_pregame_card(observation, payload.get("candidate_slot"), "draft_options")
            return 22.0 + _draft_card_score(card)
        if kind == "draft_reroll":
            return 3.0
        if kind == "toggle_pregame_choice":
            selected = set(
                ((((observation.get("self") or {}).get("sub_choice") or {}).get("selection") or {}).get("selected_slots") or [])
            )
            return 32.0 if payload.get("candidate_slot") not in selected else 6.0
        if kind == "select_pregame_choice":
            return 24.0
        if kind == "append_pregame_order":
            return 21.0
        if kind == "reset_pregame_order":
            return -20.0
        if kind == "submit_pregame_choice":
            return 30.0
        if kind == "respond":
            return 35.0 if payload.get("hand_slot") is not None else 2.0
        if kind == "use_trigger":
            target = _as_int(payload.get("target_player_id"), -1)
            return 24.0 + (2.0 if target != observation.get("seat") else 0.0)
        if kind == "submit_choice":
            return 31.0
        if kind == "append_choice_order":
            return 29.0
        if kind == "reset_choice_order":
            return -20.0
        if kind == "toggle_choice":
            selected = set(((observation.get("pending") or {}).get("selection") or {}).get("selected_slots") or [])
            return 28.0 if payload.get("candidate_slot") not in selected else 8.0
        if kind in {"select_choice", "default_choice"}:
            return 26.0
        if kind == "resolve_choice":
            choice = payload.get("choice") or {}
            if choice.get("cancelled") or choice.get("cancel"):
                return -30.0
            target = _as_int(choice.get("target_player_id"), -1)
            if target >= 0:
                return 23.0 + self._target_score(observation, target)
            return 25.0
        if kind == "v2_ui_response":
            return 18.0 if payload.get("button_role") == "cancel" else 27.0
        if kind == "play_card":
            card = _find_card(observation, payload.get("hand_slot"), zone="hand")
            if card is None:
                return 5.0
            card_type = str(card.get("card_type") or "")
            cost = _as_int(card.get("cost_e"), 0) + 1.25 * _as_int(card.get("cost_m"), 0)
            layers = max(1, _as_int(card.get("fission_level"), 1)) * max(1, _as_int(card.get("fusion_level"), 1))
            damage = max(0, _as_int(card.get("base_damage"), 0) + _as_int(card.get("power"), 0))
            hits = max(1, _as_int(card.get("base_hits"), 1) + _as_int(card.get("extra_hits"), 0))
            target = _as_int((payload.get("choice") or {}).get("target_player_id"), -1)
            target_score = self._target_score(observation, target) if target >= 0 else 0.0
            if card_type == "thorn":
                return 18.0 + damage * hits * layers * 0.7 - cost * 0.35 + target_score
            if card_type == "root":
                return 17.0 - cost * 0.2 + target_score * 0.15
            if card_type == "bloom":
                return 15.0 - cost * 0.15 + target_score * 0.1
            return 8.0 - cost * 0.2
        if kind == "end_turn":
            return -4.0
        return 0.0

    @staticmethod
    def _target_score(observation: dict[str, Any], target_id: int) -> float:
        seat = _as_int(observation.get("seat"), 0)
        if target_id == seat:
            return -1.0
        opponent = observation.get("opponent") or {}
        health = max(0, _as_int(opponent.get("health"), 0))
        return 4.0 + max(0.0, (30.0 - health) / 10.0)


def policy_from_name(name: str, *, seed: int = 0, exploration: float | None = None) -> Policy:
    raw_name = str(name or "heuristic").strip()
    normalized = raw_name.lower()
    for prefix, device in (
        ("safe-structured-cpu:", "cpu"),
        ("safe-structured:", "auto"),
    ):
        if normalized.startswith(prefix):
            from .progress_policy import ProgressSafePolicy

            checkpoint_path = raw_name[len(prefix):]
            base = _cached_structured_template(checkpoint_path, device=device).fork(
                seed=seed,
                temperature=0.0,
                record_behavior=False,
                name=Path(checkpoint_path).stem,
            )
            return ProgressSafePolicy(base)
    for prefix, device in (
        ("experience-cpu:", "cpu"),
        ("experience:", "auto"),
    ):
        if normalized.startswith(prefix):
            from .experience_prior import ExperiencePriorPolicy

            value = raw_name[len(prefix):]
            if "|" not in value:
                raise ValueError(
                    "experience policy requires STRUCTURED_CHECKPOINT|PRIOR_JSON"
                )
            checkpoint_path, prior_path = value.rsplit("|", 1)
            base = _cached_structured_template(checkpoint_path, device=device).fork(
                seed=seed,
                name=Path(checkpoint_path).stem,
            )
            prior = _cached_experience_prior(prior_path)
            return ExperiencePriorPolicy(base, prior, seed=seed)
    for prefix, device in (
        ("safe-correction-cpu:", "cpu"),
        ("safe-correction:", "auto"),
    ):
        if normalized.startswith(prefix):
            from .progress_policy import ProgressSafePolicy

            checkpoint_path = raw_name[len(prefix):]
            base = _cached_correction_template(checkpoint_path, device=device).fork(
                seed=seed,
                name=Path(checkpoint_path).stem,
            )
            return ProgressSafePolicy(base)
    for prefix, device in (
        ("correction-cpu:", "cpu"),
        ("correction:", "auto"),
    ):
        if normalized.startswith(prefix):
            checkpoint_path = raw_name[len(prefix):]
            template = _cached_correction_template(checkpoint_path, device=device)
            return template.fork(
                seed=seed,
                name=Path(checkpoint_path).stem,
            )
    for prefix, device in (
        ("structured-ensemble-cpu:", "cpu"),
        ("structured-ensemble:", "auto"),
    ):
        if normalized.startswith(prefix):
            from .structured_model import StructuredEnsemblePolicy

            paths = [
                item.strip()
                for item in raw_name[len(prefix):].split("|")
                if item.strip()
            ]
            if len(paths) < 2:
                raise ValueError("structured ensemble policy requires at least two checkpoints")
            members = [
                _cached_structured_template(path, device=device).fork(seed=seed + index)
                for index, path in enumerate(paths)
            ]
            return StructuredEnsemblePolicy(members, seed=seed)
    for prefix, device in (("ensemble-cpu:", "cpu"), ("ensemble:", "auto")):
        if normalized.startswith(prefix):
            from .neural_model import NeuralEnsemblePolicy

            paths = [item.strip() for item in raw_name[len(prefix):].split("|") if item.strip()]
            if len(paths) < 2:
                raise ValueError("ensemble policy requires at least two checkpoint paths")
            members = [
                _cached_neural_template(path, device=device).fork(seed=seed + index)
                for index, path in enumerate(paths)
            ]
            return NeuralEnsemblePolicy(members, seed=seed)
    if normalized.startswith("linear:"):
        from .linear_model import HashedLinearPolicy

        return HashedLinearPolicy.load(raw_name.split(":", 1)[1], seed=seed)
    unsafe_rollout_prefixes = (
        ("unsafe-rollout-correction-cpu:", "cpu", True),
        ("unsafe-rollout-correction:", "auto", True),
        ("unsafe-rollout-cpu:", "cpu", False),
        ("unsafe-rollout:", "auto", False),
    )
    for prefix, device, correction_base in unsafe_rollout_prefixes:
        if normalized.startswith(prefix):
            from .rollout_search import (
                UnsafeFullStateRolloutPolicy,
                parse_unsafe_rollout_spec,
            )

            checkpoint_path, config = parse_unsafe_rollout_spec(raw_name[len(prefix):])
            template = (
                _cached_correction_template(checkpoint_path, device=device)
                if correction_base
                else _cached_structured_template(checkpoint_path, device=device)
            )
            base = template.fork(
                seed=seed,
                name=f"{Path(checkpoint_path).stem}+search-base",
            )
            return UnsafeFullStateRolloutPolicy(base, config=config, seed=seed)
    for prefix, device, include_public_evidence in (
        ("structured-dynamic-belief-cpu:", "cpu", True),
        ("structured-dynamic-belief:", "auto", True),
        ("structured-belief-cpu:", "cpu", False),
        ("structured-belief:", "auto", False),
    ):
        if normalized.startswith(prefix):
            from .deck_prior import DeckBeliefPolicy

            value = raw_name[len(prefix):]
            if "|" not in value:
                raise ValueError(
                    "structured belief policy requires STRUCTURED_CHECKPOINT|DECK_PRIOR_JSON"
                )
            checkpoint_path, prior_path = value.rsplit("|", 1)
            base = _cached_structured_template(checkpoint_path, device=device).fork(
                seed=seed,
                temperature=0.0,
                record_behavior=False,
                name=Path(checkpoint_path).stem,
            )
            return DeckBeliefPolicy(
                base,
                _cached_deck_prior(prior_path),
                name=(
                    f"{Path(checkpoint_path).stem}+dynamic-deck-belief"
                    if include_public_evidence
                    else f"{Path(checkpoint_path).stem}+deck-belief"
                ),
                include_public_evidence=include_public_evidence,
            )
    structured_prefixes = (
        ("structured-onpolicy-cpu:", "cpu", 0.8, True),
        ("structured-onpolicy:", "auto", 0.8, True),
        ("structured-cpu:", "cpu", 0.0, False),
        ("structured:", "auto", 0.0, False),
    )
    for prefix, device, temperature, record_behavior in structured_prefixes:
        if normalized.startswith(prefix):
            checkpoint_path = raw_name[len(prefix):]
            template = _cached_structured_template(checkpoint_path, device=device)
            return template.fork(
                seed=seed,
                temperature=temperature,
                record_behavior=record_behavior,
                name=(
                    f"{Path(checkpoint_path).stem}+onpolicy"
                    if record_behavior else Path(checkpoint_path).stem
                ),
            )
    neural_prefixes = (
        ("neural-onpolicy-cpu:", "cpu", 0.8, 0.0, True),
        ("neural-onpolicy:", "auto", 0.8, 0.0, True),
        ("neural-explore-cpu:", "cpu", 0.8, 0.02, False),
        ("neural-explore:", "auto", 0.8, 0.02, False),
        ("neural-cpu:", "cpu", 0.0, 0.0, False),
        ("neural:", "auto", 0.0, 0.0, False),
    )
    for prefix, device, temperature, epsilon, record_behavior in neural_prefixes:
        if normalized.startswith(prefix):
            checkpoint_path = raw_name[len(prefix):]
            template = _cached_neural_template(checkpoint_path, device=device)
            suffix = "+onpolicy" if record_behavior else (
                "+explore" if temperature > 0 or epsilon > 0 else ""
            )
            return template.fork(
                seed=seed,
                temperature=temperature,
                epsilon=epsilon,
                record_behavior=record_behavior,
                name=f"{Path(checkpoint_path).stem}{suffix}",
            )
    if normalized in {"random", "random-v1"}:
        return RandomPolicy(seed=seed)
    if normalized in {"heuristic", "heuristic-v1", "baseline"}:
        return HeuristicPolicy(seed=seed, exploration=0.04 if exploration is None else float(exploration))
    raise ValueError(f"unknown policy: {name}")


def _cached_neural_template(path: str, *, device: str):
    from .neural_model import NeuralPolicy

    resolved = str(Path(path).expanduser().resolve())
    key = (resolved, str(device))
    template = _NEURAL_TEMPLATE_CACHE.get(key)
    if template is None:
        template = NeuralPolicy.load(resolved, device=device, seed=0)
        _NEURAL_TEMPLATE_CACHE[key] = template
    return template


def _cached_structured_template(path: str, *, device: str):
    from .structured_model import StructuredPolicy

    resolved = str(Path(path).expanduser().resolve())
    key = (resolved, str(device))
    template = _STRUCTURED_TEMPLATE_CACHE.get(key)
    if template is None:
        template = StructuredPolicy.load(resolved, device=device, seed=0)
        _STRUCTURED_TEMPLATE_CACHE[key] = template
    return template


def _cached_correction_template(path: str, *, device: str):
    from .correction_model import StructuredCorrectionPolicy

    resolved = str(Path(path).expanduser().resolve())
    key = (resolved, str(device))
    template = _CORRECTION_TEMPLATE_CACHE.get(key)
    if template is None:
        template = StructuredCorrectionPolicy.load(resolved, device=device, seed=0)
        _CORRECTION_TEMPLATE_CACHE[key] = template
    return template


def _cached_experience_prior(path: str):
    from .experience_prior import ExperiencePrior

    resolved = str(Path(path).expanduser().resolve())
    prior = _EXPERIENCE_PRIOR_CACHE.get(resolved)
    if prior is None:
        prior = ExperiencePrior.load(resolved)
        _EXPERIENCE_PRIOR_CACHE[resolved] = prior
    return prior


def _cached_deck_prior(path: str):
    from .deck_prior import DeckPrior

    resolved = str(Path(path).expanduser().resolve())
    prior = _DECK_PRIOR_CACHE.get(resolved)
    if prior is None:
        prior = DeckPrior.load(resolved)
        _DECK_PRIOR_CACHE[resolved] = prior
    return prior


def _find_card(observation: dict[str, Any], slot: Any, *, zone: str) -> dict[str, Any] | None:
    expected = _as_int(slot, -1)
    if zone == "hand":
        cards = (observation.get("self") or {}).get("hand") or []
    else:
        cards = (observation.get("pending") or {}).get("candidates") or []
    for card in cards:
        if _as_int(card.get("slot"), -2) == expected:
            return card
    return None


def _find_pregame_card(observation: dict[str, Any], slot: Any, zone: str) -> dict[str, Any] | None:
    own = observation.get("self") or {}
    cards = own.get(zone) or []
    return _find_by_slot(cards, slot)


def _find_by_slot(cards: Sequence[dict[str, Any]], slot: Any) -> dict[str, Any] | None:
    expected = _as_int(slot, -1)
    return next((card for card in cards if _as_int(card.get("slot"), -2) == expected), None)


def _draft_card_score(card: dict[str, Any] | None) -> float:
    if not card:
        return 0.0
    damage = max(0, _as_int(card.get("base_damage"), 0))
    hits = max(1, _as_int(card.get("base_hits"), 1))
    cost = max(0, _as_int(card.get("cost_e"), 0)) + 1.25 * max(0, _as_int(card.get("cost_m"), 0))
    return min(18.0, damage * hits * 0.25) - min(8.0, cost * 0.2)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
