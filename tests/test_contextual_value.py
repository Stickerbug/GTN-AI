from __future__ import annotations

import json

from gtn_ai import Garden1v1Env
from gtn_ai.contextual_value import (
    card_definition_semantics,
    contextual_value_tokens,
    enrich_card_for_context,
    transition_value_components,
)
from gtn_ai.structured_features import StructuredFeatureConfig, encode_structured_decision


def _vanilla_env(seed: int = 8801) -> Garden1v1Env:
    return Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=seed,
        include_pregame=False,
    )


def test_card_instance_tags_change_semantics_without_exposing_instance_ids() -> None:
    _vanilla_env().reset()
    from cards import CardInstance
    from gtn_ai.observation import _card_view

    ordinary = _card_view(CardInstance("Light"))
    modified_card = CardInstance("Light")
    modified_card.instance_flags.update({"sprout", "symbiosis"})
    modified = _card_view(modified_card)

    ordinary_context = enrich_card_for_context(ordinary)
    modified_context = enrich_card_for_context(modified)

    assert ordinary_context["instance_semantics"]["added_flags"] == []
    assert modified_context["instance_semantics"]["added_flags"] == [
        "sprout",
        "symbiosis",
    ]
    assert "instance_id" not in json.dumps(modified_context, ensure_ascii=False)


def test_definition_semantics_expose_trigger_demand_and_rule_operations() -> None:
    load_env = Garden1v1Env(
        enabled_mods=["Factory Cards Addition.gtnmod"],
        seed=8802,
        include_pregame=False,
    )
    load_env.reset()

    uranium = card_definition_semantics("MagicUranium")

    assert uranium["trigger_cost_m"] == 2
    assert uranium["mod_id"] == "factory"
    assert "on_equipment_trigger" in uranium["events"]
    assert uranium["rule_operations"]


def test_context_reconstructs_base_flags_and_cost_for_serialized_pile_cards() -> None:
    _vanilla_env(8812).reset()
    serialized = {
        "def_id": "Dust",
        "cost_e_override": None,
        "mimic_discount": 0,
        "temporary_swift": 0,
        "temporary_heavy": 0,
        "flags": ["symbiosis"],
        "disabled_flags": [],
        "fission_level": 1,
        "fusion_level": 1,
    }

    enriched = enrich_card_for_context(serialized)

    assert enriched["cost_e"] == 1
    assert "symbiosis" in enriched["flags"]
    assert set(enriched["definition_semantics"]["base_flags"]).issubset(
        set(enriched["flags"])
    )


def test_value_context_keeps_resource_demand_separate_from_supply() -> None:
    Garden1v1Env(
        enabled_mods=["Factory Cards Addition.gtnmod"],
        seed=8804,
        include_pregame=False,
    ).reset()
    from cards import CardInstance
    from gtn_ai.observation import _card_view

    uranium = _card_view(CardInstance("MagicUranium"))
    light = _card_view(CardInstance("Light"))
    observation = {
        "phase": "action",
        "self": {
            "health": 50,
            "max_health": 50,
            "elixir": 4,
            "max_elixir": 10,
            "magic": 0,
            "max_magic": 10,
            "armor": 0,
            "hand_limit": 10,
            "hand": [{"slot": 0, **light}],
            "equipment": [{"card": uranium}],
            "statuses": {},
        },
        "opponent": {"equipment": [], "statuses": {}},
    }

    contexts = dict(contextual_value_tokens(observation))
    equipment = contexts["value_context:equipment"]["self"]
    pool = contexts["value_context:self_pool"]

    assert equipment["trigger_demand"]["magic"] == 2
    assert pool["trigger_demand"]["magic"] == 0
    assert contexts["value_context:resources"]["self"]["magic"] == 0


def test_contextual_encoder_is_opt_in_and_changes_instance_encoding() -> None:
    from cards import CardInstance

    env = _vanilla_env(8803)
    env.reset()
    actor = env.decision_player()
    assert actor is not None
    tagged = CardInstance("Light")
    tagged.instance_flags.update({"sprout", "symbiosis"})
    env.engine.players[actor].hand = [tagged]
    observation = env.observe(actor)
    legal = env.legal_actions(actor)

    ordinary = encode_structured_decision(
        observation,
        legal,
        config=StructuredFeatureConfig(max_state_tokens=256),
    )
    contextual = encode_structured_decision(
        observation,
        legal,
        config=StructuredFeatureConfig(
            max_state_tokens=256,
            contextual_value_features=True,
        ),
    )

    assert len(contextual.state_tokens) > len(ordinary.state_tokens)
    assert contextual.action_tokens != ordinary.action_tokens


def test_transition_components_do_not_collapse_outcomes_to_one_score() -> None:
    before = {
        "round": 4,
        "current_player": 0,
        "game_over": False,
        "winner": -1,
        "self": {
            "health": 40,
            "elixir": 3,
            "magic": 1,
            "armor": 0,
            "hand_count": 4,
            "equipment": [],
            "statuses": {"bleed": 4},
        },
        "opponent": {
            "health": 30,
            "elixir": 2,
            "magic": 0,
            "armor": 0,
            "hand_count": 5,
            "equipment": [{}],
            "statuses": {},
        },
    }
    after = {
        **before,
        "self": {
            **before["self"],
            "health": 36,
            "elixir": 3,
            "hand_count": 3,
        },
        "opponent": {**before["opponent"], "health": 30},
    }

    outcome = transition_value_components(before, after)

    assert outcome["self"]["deltas"]["health"] == -4
    assert outcome["self"]["deltas"]["hand_count"] == -1
    assert outcome["opponent"]["deltas"]["health"] == 0
    assert "score" not in outcome
