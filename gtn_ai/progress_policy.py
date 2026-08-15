from __future__ import annotations

from typing import Any, Sequence

from .protocol import Action
from .rollout_search import (
    _choice_progress_indices,
    _exact_choice_submit_index,
    _progress_action_index,
    _public_state_key,
)


class ProgressSafePolicy:
    """Public-information policy wrapper that prevents reversible action loops."""

    def __init__(self, base_policy: Any) -> None:
        self.base_policy = base_policy
        self.name = f"{base_policy.name}+progress-safe"
        self.ruleset_fingerprint = str(
            getattr(base_policy, "ruleset_fingerprint", "") or ""
        )
        self.model_fingerprint = str(
            getattr(base_policy, "model_fingerprint", "") or ""
        )
        self.last_decision_metadata = None
        self._turn_key: tuple[Any, ...] | None = None
        self._seen_actions_by_state: dict[str, set[str]] = {}
        self._decisions = 0
        self._repeated_action_avoids = 0
        self._choice_autocompletions = 0
        self._choice_backtrack_filters = 0
        self._forced_progress_actions = 0

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("progress-safe policy received an empty legal action list")
        self.last_decision_metadata = None
        self._decisions += 1
        self._start_turn(observation)

        submit_index = _exact_choice_submit_index(observation, actions)
        if submit_index is not None:
            self._choice_autocompletions += 1
            return actions[submit_index]

        eligible_indices = _choice_progress_indices(observation, actions)
        if len(eligible_indices) < len(actions):
            self._choice_backtrack_filters += 1
        ranked = self._rank_actions(observation, actions, eligible_indices)
        state_key = _public_state_key(observation)
        seen = self._seen_actions_by_state.setdefault(state_key, set())
        selected_index = ranked[0]
        if actions[selected_index].key in seen:
            unseen = [index for index in ranked if actions[index].key not in seen]
            if unseen:
                selected_index = unseen[0]
            else:
                local_index = _progress_action_index(
                    observation,
                    [actions[index] for index in eligible_indices],
                )
                selected_index = eligible_indices[local_index]
                self._forced_progress_actions += 1
            self._repeated_action_avoids += 1
        seen.add(actions[selected_index].key)
        return actions[selected_index]

    def diagnostics(self) -> dict[str, Any]:
        base_diagnostics = getattr(self.base_policy, "diagnostics", None)
        base = base_diagnostics() if callable(base_diagnostics) else None
        result = {
            "policy": self.name,
            "kind": "progress_safe_v1",
            "decisions": self._decisions,
            "repeated_action_avoids": self._repeated_action_avoids,
            "choice_autocompletions": self._choice_autocompletions,
            "choice_backtrack_filters": self._choice_backtrack_filters,
            "forced_progress_actions": self._forced_progress_actions,
            "base": base,
        }
        if isinstance(base, dict):
            for key in ("gated_decisions", "action_changes"):
                result[key] = int(base.get(key, 0) or 0)
        return result

    def _start_turn(self, observation: dict[str, Any]) -> None:
        turn_key = (
            str(observation.get("phase") or ""),
            observation.get("round"),
            observation.get("current_player"),
            observation.get("decision_player"),
        )
        if turn_key == self._turn_key:
            return
        self._turn_key = turn_key
        self._seen_actions_by_state.clear()

    def _rank_actions(
        self,
        observation: dict[str, Any],
        actions: list[Action],
        eligible_indices: list[int],
    ) -> list[int]:
        evaluator = getattr(self.base_policy, "evaluate_actions", None)
        if callable(evaluator):
            logits, _ = evaluator(observation, actions)
            return sorted(
                eligible_indices,
                key=lambda index: (-float(logits[index]), index),
            )
        eligible = [actions[index] for index in eligible_indices]
        selected = self.base_policy.select_action(observation, eligible)
        selected_index = next(
            index for index in eligible_indices if actions[index].key == selected.key
        )
        return [selected_index, *(
            index for index in eligible_indices if index != selected_index
        )]
