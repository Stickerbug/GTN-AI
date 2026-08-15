from __future__ import annotations

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.belief_sampling import determinize_hidden_cards
from gtn_ai.inference_server import DecisionService
from gtn_ai.neural_model import torch_available
from gtn_ai.progress_policy import ProgressSafePolicy
from gtn_ai.protocol import Action
from gtn_ai.rollout_search import (
    UnsafeFullStateRolloutPolicy,
    UnsafeRolloutConfig,
    _choice_progress_indices,
    parse_unsafe_rollout_spec,
)
from gtn_ai.structured_model import (
    StructuredModelConfig,
    StructuredPolicy,
    StructuredPolicyNetwork,
)


pytestmark = pytest.mark.skipif(
    not torch_available(), reason="PyTorch training extra is not installed"
)


def _policy(env: Garden1v1Env) -> StructuredPolicy:
    config = StructuredModelConfig(
        categorical_buckets=256,
        categorical_slots=16,
        numeric_buckets=16,
        max_state_tokens=64,
        max_history_events=8,
        model_dim=32,
        num_heads=4,
        state_layers=1,
        action_layers=1,
        feedforward_dim=64,
        dropout=0,
    )
    return StructuredPolicy(
        StructuredPolicyNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=17,
    )


def test_full_state_rollout_search_is_legal_and_does_not_mutate_root() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=1901,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    legal = env.legal_actions(actor)
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(candidates=2, rollouts=1, horizon=1),
        seed=19,
    )

    selected = search.select_action_with_env(env, observation, legal, actor)

    assert selected in legal
    assert env.observe(actor) == observation
    diagnostics = search.diagnostics()
    assert diagnostics["searched_decisions"] == 1
    assert diagnostics["candidate_evaluations"] == min(2, len(legal))
    assert diagnostics["rollouts"] == min(2, len(legal))
    assert search.last_search_metadata["selected_action_key"] == selected.key


def test_full_state_rollout_search_cannot_be_served() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=1902,
        include_pregame=False,
    )
    env.reset()
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(candidates=1, rollouts=1, horizon=0),
    )

    with pytest.raises(ValueError, match="offline-only"):
        DecisionService(search)
    with pytest.raises(RuntimeError, match="cannot be used"):
        search.select_action(env.observe(env.decision_player()), env.legal_actions())


def test_rollout_policy_fork_shares_model_weights_but_not_search_state() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=1905,
        include_pregame=False,
    )
    env.reset()
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(candidates=1, rollouts=1, horizon=0),
        seed=1,
    )

    forked = search.fork(seed=2)

    assert forked is not search
    assert forked.base_policy is not search.base_policy
    assert forked.base_policy.model is search.base_policy.model
    assert forked.config is search.config


def test_unsafe_rollout_spec_parses_and_rejects_unknown_options() -> None:
    checkpoint, config = parse_unsafe_rollout_spec(
        "models/model.pt;candidates=4;rollouts=3;horizon=6;exploration=.1;prior=.03;belief=true;deck-prior=models/decks.json"
    )
    assert checkpoint == "models/model.pt"
    assert config == UnsafeRolloutConfig(
        candidates=4,
        rollouts=3,
        horizon=6,
        rollout_exploration=0.1,
        prior_weight=0.03,
        determinize_hidden=True,
        belief_deck_prior_path="models/decks.json",
    )
    with pytest.raises(ValueError, match="unknown unsafe rollout option"):
        parse_unsafe_rollout_spec("models/model.pt;depth=4")
    with pytest.raises(ValueError, match="requires belief"):
        parse_unsafe_rollout_spec("models/model.pt;deck-prior=models/decks.json")


def test_unsafe_rollout_spec_parses_adaptive_sampling_options() -> None:
    checkpoint, config = parse_unsafe_rollout_spec(
        "models/model.pt;rollouts=2;max-rollouts=8;confidence=.075;batch=2"
    )

    assert checkpoint == "models/model.pt"
    assert config.rollouts == 2
    assert config.max_rollouts == 8
    assert config.confidence_margin == pytest.approx(0.075)
    assert config.rollout_batch == 2


def test_rollout_spec_can_disable_common_random_numbers() -> None:
    _, config = parse_unsafe_rollout_spec(
        "models/model.pt;crn=false;avoid-repeats=false;auto-submit=false;avoid-choice-backtracking=false"
    )

    assert config.common_random_numbers is False
    assert config.avoid_repeated_actions is False
    assert config.auto_submit_exact_choices is False
    assert config.avoid_choice_backtracking is False


def test_rollout_spec_can_enable_safe_annotation_execution() -> None:
    _, config = parse_unsafe_rollout_spec(
        "models/model.pt;annotate=true;safe-annotate=true"
    )

    assert config.annotate_only is True
    assert config.safe_annotation_execution is True


def test_rollout_candidates_share_random_seed_by_default() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=19023,
        include_pregame=False,
    )
    env.reset()
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(),
        seed=190231,
    )

    first = search._rollout_seed(action_index=0, rollout_index=2)
    second = search._rollout_seed(action_index=7, rollout_index=2)

    assert first == second


def test_rollout_search_avoids_repeating_an_action_in_the_same_public_state(
    monkeypatch,
) -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=19024,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    legal = env.legal_actions(actor)
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(
            candidates=min(3, len(legal)),
            rollouts=1,
            horizon=0,
        ),
        seed=190241,
    )
    def fixed_rollouts(
        env,
        actions,
        candidate_indices,
        candidate_returns,
        *,
        root_player,
        start_rollout,
        stop_rollout,
    ):
        count = stop_rollout - start_rollout
        for position, values in enumerate(candidate_returns):
            values.extend([1.0 - position] * count)

    monkeypatch.setattr(search, "_append_rollout_batch", fixed_rollouts)

    first = search.select_action_with_env(env, observation, legal, actor)
    second = search.select_action_with_env(env, observation, legal, actor)

    assert first != second
    assert search.diagnostics()["repeated_action_avoids"] == 1


def test_progress_safe_policy_uses_the_best_unseen_action() -> None:
    class FixedPolicy:
        name = "fixed"
        ruleset_fingerprint = "rules"
        model_fingerprint = "model"

        @staticmethod
        def evaluate_actions(observation, actions):
            return [float(len(actions) - index) for index in range(len(actions))], 0.0

    policy = ProgressSafePolicy(FixedPolicy())
    observation = {
        "phase": "action",
        "round": 2,
        "current_player": 0,
        "decision_player": 0,
    }
    actions = [Action("play_card", {"hand_slot": index}) for index in range(3)]

    assert policy.select_action(observation, actions) == actions[0]
    assert policy.select_action(observation, actions) == actions[1]
    assert policy.select_action(observation, actions) == actions[2]
    diagnostics = policy.diagnostics()
    assert diagnostics["repeated_action_avoids"] == 2
    assert diagnostics["forced_progress_actions"] == 0


def test_progress_safe_policy_clears_repeat_memory_on_the_next_turn() -> None:
    class FixedPolicy:
        name = "fixed"

        @staticmethod
        def evaluate_actions(observation, actions):
            return [2.0, 1.0], 0.0

    policy = ProgressSafePolicy(FixedPolicy())
    observation = {
        "phase": "action",
        "round": 2,
        "current_player": 0,
        "decision_player": 0,
    }
    actions = [Action("play_card", {"hand_slot": 0}), Action("end_turn")]

    assert policy.select_action(observation, actions) == actions[0]
    next_turn = {**observation, "round": 3}
    assert policy.select_action(next_turn, actions) == actions[0]


def test_progress_safe_policy_submits_a_completed_fixed_choice() -> None:
    class FixedPolicy:
        name = "fixed"

        @staticmethod
        def select_action(observation, actions):
            return actions[0]

    policy = ProgressSafePolicy(FixedPolicy())
    observation = {
        "phase": "action",
        "round": 2,
        "current_player": 0,
        "decision_player": 0,
        "pending": {
            "selection": {"selected_slots": [3]},
            "constraints": {"min_count": 1, "max_count": 1},
        },
    }
    actions = [
        Action("toggle_choice", {"candidate_slot": 3}),
        Action("submit_choice"),
    ]

    assert policy.select_action(observation, actions).kind == "submit_choice"
    assert policy.diagnostics()["choice_autocompletions"] == 1


def test_rollout_search_submits_a_completed_fixed_size_choice() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=19025,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    observation["pending"] = {
        "kind": "choice",
        "selection": {"selected_slots": [3]},
        "constraints": {"min_count": 1, "max_count": 1},
    }
    legal = [
        Action("toggle_choice", {"candidate_slot": 3}),
        Action("submit_choice"),
        Action("resolve_choice", {"choice": {"cancelled": True}}),
    ]
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(candidates=2, rollouts=1, horizon=0),
        seed=190251,
    )

    selected = search.select_action_with_env(env, observation, legal, actor)

    assert selected.kind == "submit_choice"
    diagnostics = search.diagnostics()
    assert diagnostics["choice_autocompletions"] == 1
    assert diagnostics["searched_decisions"] == 0


def test_choice_progress_keeps_compatible_fixed_choice_and_drops_backtracking() -> None:
    observation = {
        "pending": {
            "choice_type": "choose_cards_from_hand",
            "selection": {"selected_slots": [2]},
            "constraints": {"min_count": 2, "max_count": 2},
        }
    }
    actions = [
        Action("toggle_choice", {"candidate_slot": 2}),
        Action("toggle_choice", {"candidate_slot": 4}),
        Action("resolve_choice", {"choice": {"cancelled": True}}),
    ]

    indices = _choice_progress_indices(observation, actions)

    assert indices == [1, 2]


def test_choice_progress_allows_backtracking_without_a_compatible_candidate() -> None:
    observation = {
        "pending": {
            "choice_type": "choose_cards_from_hand",
            "selection": {"selected_slots": [2]},
            "constraints": {"min_count": 2, "max_count": 2},
        }
    }
    actions = [
        Action("toggle_choice", {"candidate_slot": 2}),
        Action("resolve_choice", {"choice": {"cancelled": True}}),
    ]

    assert _choice_progress_indices(observation, actions) == [0, 1]


def test_choice_progress_drops_reorder_reset_after_ordering_begins() -> None:
    observation = {
        "pending": {
            "choice_type": "reorder_deck",
            "selection": {"selected_slots": [3, 1]},
        }
    }
    actions = [
        Action("append_choice_order", {"candidate_slot": 4}),
        Action("reset_choice_order"),
    ]

    assert _choice_progress_indices(observation, actions) == [0]


def test_adaptive_rollout_expands_uncertain_decision_to_configured_maximum() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=19021,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    legal = env.legal_actions(actor)
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(
            candidates=2,
            rollouts=1,
            max_rollouts=3,
            confidence_margin=100.0,
            rollout_batch=1,
            horizon=0,
        ),
        seed=190211,
    )

    selected = search.select_action_with_env(env, observation, legal, actor)

    assert selected in legal
    assert search.last_search_metadata["rollouts_used"] == 3
    assert search.diagnostics()["rollouts"] == min(2, len(legal)) * 3


def test_rollout_confidence_gate_bypasses_clear_base_decision() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=19022,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    legal = env.legal_actions(actor)
    base = _policy(env)
    logits, _ = base.evaluate_actions(observation, legal)
    ordered = sorted(logits, reverse=True)
    margin = ordered[0] - ordered[1]
    assert margin > 0.0
    search = UnsafeFullStateRolloutPolicy(
        base,
        config=UnsafeRolloutConfig(
            candidates=2,
            rollouts=1,
            horizon=0,
            base_margin_gate=margin / 2,
        ),
        seed=190221,
    )

    selected = search.select_action_with_env(env, observation, legal, actor)

    assert selected in legal
    assert search.diagnostics()["searched_decisions"] == 0
    assert search.diagnostics()["confidence_bypassed_decisions"] == 1


def test_belief_sample_preserves_public_observation_and_changes_private_cards() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=1903,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    opponent = 1 - actor
    before_observation = env.observe(actor)
    before_cards = _private_zone_defs(env, opponent)

    summary = determinize_hidden_cards(env, actor, seed=19031)

    assert env.observe(actor) == before_observation
    assert _private_zone_defs(env, opponent) != before_cards
    assert summary.sampled_cards == sum(len(zone) for zone in before_cards)


def test_belief_sample_is_independent_of_actual_private_card_definitions() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=1904,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    opponent = 1 - actor
    first = env.clone()
    second = env.clone()
    from cards import CardInstance

    for zone_name in ("hand", "deck", "discard", "exile"):
        zone = getattr(second.engine.players[opponent], zone_name)
        for index, card in enumerate(zone):
            zone[index] = CardInstance("Basic", instance_id=card.instance_id)
    assert first.observe(actor) == second.observe(actor)

    first_summary = determinize_hidden_cards(first, actor, seed=19041)
    second_summary = determinize_hidden_cards(second, actor, seed=19041)

    assert first_summary == second_summary
    assert _private_zone_defs(first, opponent) == _private_zone_defs(second, opponent)


def test_belief_sample_retains_recent_publicly_played_card_without_hidden_leak() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=19042,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    opponent = 1 - actor
    first = env.clone()
    second = env.clone()
    from cards import CardInstance

    public_event = {
        "round": 1,
        "player": opponent,
        "kind": "play_card",
        "card_def_id": "Fire",
    }
    first.public_history.append(public_event)
    second.public_history.append(dict(public_event))
    for zone_name in ("hand", "deck", "discard", "exile"):
        zone = getattr(second.engine.players[opponent], zone_name)
        for index, card in enumerate(zone):
            zone[index] = CardInstance("Basic", instance_id=card.instance_id)

    first_summary = determinize_hidden_cards(first, actor, seed=190421)
    second_summary = determinize_hidden_cards(second, actor, seed=190421)

    first_defs = _private_zone_defs(first, opponent)
    assert first_defs == _private_zone_defs(second, opponent)
    assert "Fire" in {def_id for zone in first_defs for def_id in zone}
    assert first_summary.history_constrained_cards >= 1
    assert first_summary == second_summary


def test_belief_rollout_search_uses_samples_without_mutating_root() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=1905,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    legal = env.legal_actions(actor)
    search = UnsafeFullStateRolloutPolicy(
        _policy(env),
        config=UnsafeRolloutConfig(
            candidates=2,
            rollouts=2,
            horizon=1,
            determinize_hidden=True,
        ),
        seed=19051,
    )

    selected = search.select_action_with_env(env, observation, legal, actor)

    assert selected in legal
    assert env.observe(actor) == observation
    diagnostics = search.diagnostics()
    assert diagnostics["belief_samples"] == 2
    assert diagnostics["belief_sampled_cards"] > 0
    assert diagnostics["belief_history_constraints"] >= 0
    assert diagnostics["belief_failures"] == 0
    teacher = search.last_teacher_metadata
    assert teacher["action_key"] == selected.key
    assert len(teacher["logits"]) == len(legal)
    assert legal[max(range(len(legal)), key=teacher["logits"].__getitem__)] == selected
    assert -1.0 <= teacher["value"] <= 1.0


def test_belief_annotator_records_search_target_but_executes_base_action() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=1906,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    legal = env.legal_actions(actor)
    base = _policy(env)
    expected_index = max(
        range(len(legal)),
        key=base.evaluate_actions(observation, legal)[0].__getitem__,
    )
    search = UnsafeFullStateRolloutPolicy(
        base,
        config=UnsafeRolloutConfig(
            candidates=2,
            rollouts=2,
            horizon=1,
            determinize_hidden=True,
            annotate_only=True,
        ),
        seed=19061,
    )

    executed = search.select_action_with_env(env, observation, legal, actor)

    assert executed == legal[expected_index]
    assert search.last_teacher_metadata["action_key"] in {
        action.key for action in legal
    }
    assert search.last_search_metadata["executed_action_key"] == executed.key
    assert env.observe(actor) == observation


def _private_zone_defs(env: Garden1v1Env, player_id: int) -> list[list[str]]:
    player = env.engine.players[player_id]
    return [
        [str(card.def_id) for card in getattr(player, zone_name)]
        for zone_name in ("hand", "deck", "discard", "exile")
    ]
