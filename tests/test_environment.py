from __future__ import annotations

import json

import pytest

from gtn_ai import Action, Garden1v1Env, IllegalActionError
from gtn_ai.game_imports import load_official_content, official_ruleset_fingerprint


VANILLA = ["Vanilla Cards.gtnmod"]


def make_env(seed: int = 7) -> Garden1v1Env:
    env = Garden1v1Env(enabled_mods=VANILLA, seed=seed, include_pregame=False)
    env.reset()
    return env


def test_official_catalog_excludes_entertainment_mods():
    _, _, names = load_official_content()
    assert len(names) >= 12
    assert "Vanilla Cards.gtnmod" in names
    assert all("DLC" not in name for name in names)
    assert all("Formal Logic" not in name for name in names)


def test_same_seed_reproduces_state_and_actions():
    left = make_env(31)
    right = make_env(31)
    assert left.observe(0) == right.observe(0)
    assert [action.key for action in left.legal_actions()] == [action.key for action in right.legal_actions()]


def test_observation_carries_stable_global_ruleset_fingerprint():
    left = make_env(32)
    right = make_env(33)
    expected = official_ruleset_fingerprint()
    assert len(expected) == 40
    assert left.ruleset_fingerprint == expected == right.ruleset_fingerprint
    assert left.observe(0)["loadout"]["ruleset_fingerprint"] == expected


def test_clone_is_independent_and_deterministic():
    env = make_env(41)
    clone = env.clone()
    action = env.legal_actions()[0]
    env.step(action)
    assert clone.observe(0) != env.observe(0)
    clone_result = clone.step(action)
    assert clone_result.observation == env.observe(clone.decision_player(default=0))


def test_clone_preserves_an_in_progress_choice_builder():
    env = make_env(42)
    actor = env.decision_player()
    card = env.engine.players[actor].hand[0]
    env.engine.pending_choice = {
        "player_id": actor,
        "choice_type": "choose_cards_from_hand",
        "choice_params": {"min_count": 1, "max_count": 2},
        "card": card.to_dict(),
    }
    env._choice_signature = env._pending_choice_signature()
    env._choice_selection = [int(card.instance_id)]
    env._choice_candidate_ids = [int(card.instance_id)]

    clone = env.clone()

    assert clone._choice_signature == clone._pending_choice_signature()
    assert clone._choice_selection == env._choice_selection
    assert clone._choice_candidate_ids == env._choice_candidate_ids


def test_hidden_opponent_card_is_not_in_observation():
    env = make_env(51)
    from cards import CardInstance

    secret = CardInstance("Yggdrasil")
    env.engine.players[1].hand = [secret]
    observation = env.observe(0)
    encoded = json.dumps(observation, ensure_ascii=False)
    assert "Yggdrasil" not in encoded
    assert observation["opponent"]["hand_count"] == 1


def test_protocol_never_exposes_engine_instance_ids_or_private_custom_vars():
    env = make_env(511)
    actor = env.decision_player()
    card = env.engine.players[actor].hand[0]
    card.custom_vars["secret_snapshot"] = 123456789
    card.custom_vars["setup_magic_acceleration_last_instance_id"] = 987654321
    card.custom_vars["display_name_en"] = "Visible Name"

    observation = env.observe(actor)
    actions = [action.to_dict() for action in env.legal_actions(actor)]
    assert not _keys_matching(observation, "instance")
    assert not _keys_matching(actions, "instance")
    encoded = json.dumps(observation, ensure_ascii=False)
    assert "secret_snapshot" not in encoded
    assert "setup_magic_acceleration_last_instance_id" not in encoded
    assert "Visible Name" in encoded


def test_revealed_opponent_card_is_visible():
    env = make_env(52)
    from cards import CardInstance

    secret = CardInstance("Basic")
    secret.instance_flags.add("revealed")
    env.engine.players[1].hand = [secret]
    observation = env.observe(0)
    assert observation["opponent"]["revealed_hand"][0]["def_id"] == "Basic"


def test_zone_multiset_stably_sorts_optional_and_numeric_overrides():
    env = make_env(521)
    from cards import CardInstance

    actor = env.engine.current_player
    ordinary = CardInstance("Basic")
    overridden = CardInstance("Basic")
    overridden.cost_e_override = 0
    env.engine.players[actor].discard = [ordinary, overridden]

    discard = env.observe(actor)["self"]["discard"]
    assert len(discard) == 2
    assert {entry["card"]["cost_e_override"] for entry in discard} == {None, 0}


def test_blindness_masks_information_but_preserves_card_slots():
    env = make_env(53)
    player = env.engine.players[0]
    player.blind = 2
    observation = env.observe(0)
    assert observation["self"]["health"] is None
    assert observation["self"]["elixir"] is None
    assert observation["self"]["hand"]
    assert "slot" in observation["self"]["hand"][0]
    assert "def_id" not in observation["self"]["hand"][0]
    assert "card_type" not in observation["self"]["hand"][0]


def test_illegal_action_is_rejected_before_engine_execution():
    env = make_env(61)
    before = env.observe(0)
    with pytest.raises(IllegalActionError):
        env.step(Action("play_card", {"hand_slot": 999999999}))
    assert env.observe(0) == before


def test_draw_phase_keeps_current_player_as_decision_owner():
    env = make_env(612)
    env.engine.phase = "draw"
    env.engine.current_player = 1

    assert env.decision_player() == 1
    assert env.legal_actions(1)
    assert env.legal_actions(0) == []


def test_mod_runtime_error_is_not_silently_accepted(monkeypatch):
    env = make_env(611)
    from runtime_errors import MOD_RUNTIME_ERROR_MESSAGE
    from gtn_ai import UnsupportedDecisionError

    actor = env.decision_player()
    action = Action("end_turn")

    def broken_execute(_action, _player_id):
        env.engine.log.append(MOD_RUNTIME_ERROR_MESSAGE)
        return {"success": True}

    monkeypatch.setattr(env, "_execute", broken_execute)
    with pytest.raises(UnsupportedDecisionError, match="mod runtime error"):
        env.step(action, actor)


def test_every_initial_legal_action_can_step_on_a_clone():
    env = make_env(71)
    for action in env.legal_actions():
        branch = env.clone()
        branch.step(action)


def test_short_random_episode_terminates():
    from gtn_ai.policies import RandomPolicy
    from gtn_ai.self_play import play_episode

    env = Garden1v1Env(enabled_mods=VANILLA, seed=81)
    episode = play_episode(env, [RandomPolicy(1), RandomPolicy(2)], max_steps=1500)
    assert episode.terminated
    assert not episode.truncated
    assert episode.steps > 0


def test_fusion_picker_only_allows_matching_attacks_and_resolves():
    env = make_env(101)
    from cards import CardInstance

    actor = env.engine.current_player
    player = env.engine.players[actor]
    player.elixir = 20
    player.magic = 20
    with env.runtime.activate():
        fusion = CardInstance("Fusion")
        first = CardInstance("Basic")
        second = CardInstance("Basic")
        different = CardInstance("Bone")
    player.hand = [fusion, first, second, different]

    play = next(
        action for action in env.legal_actions(actor)
        if action.kind == "play_card" and action.payload["hand_slot"] == 0
    )
    env.step(play, actor)
    candidates = env.observe(actor)["pending"]["candidates"]
    first_slot = next(item["slot"] for item in candidates if item.get("def_id") == "Basic")
    env.step(Action("toggle_choice", {"candidate_slot": first_slot}), actor)
    toggle_slots = {
        action.payload["candidate_slot"]
        for action in env.legal_actions(actor)
        if action.kind == "toggle_choice"
    }
    basic_slots = [item["slot"] for item in candidates if item.get("def_id") == "Basic"]
    bone_slot = next(item["slot"] for item in candidates if item.get("def_id") == "Bone")
    second_slot = next(slot for slot in basic_slots if slot != first_slot)
    assert second_slot in toggle_slots
    assert bone_slot not in toggle_slots

    env.step(Action("toggle_choice", {"candidate_slot": second_slot}), actor)
    submit = next(action for action in env.legal_actions(actor) if action.kind == "submit_choice")
    env.step(submit, actor)
    if env.engine.pending_response is not None:
        responder = env.decision_player()
        env.step(Action("respond", {"hand_slot": None}), responder)
    fused = next(card for card in player.hand if card.def_id == "Basic")
    assert fused.fusion_level == 2
    assert len([card for card in player.hand if card.def_id == "Basic"]) == 1


def test_engine_snapshotted_single_choice_candidates_are_authoritative():
    env = make_env(102)
    from cards import CardInstance

    actor = env.engine.current_player
    source = CardInstance("Basic")
    allowed = CardInstance("Bone")
    excluded = CardInstance("Stinger")
    env.engine.players[actor].hand = [allowed, excluded]
    env.engine.pending_choice = {
        "player_id": actor,
        "choice_type": "choose_card_from_hand",
        "choice_params": {"cancellable": False},
        "card": source.to_dict(),
        "target_player_id": actor,
        "hand_cards": [allowed.to_dict()],
    }
    actions = env.legal_actions(actor)
    assert actions == [Action("select_choice", {"candidate_slot": 0})]
    candidate = env.observe(actor)["pending"]["candidates"][0]
    assert candidate["def_id"] == "Bone"


def test_snapshotted_picker_excludes_the_card_currently_being_played():
    env = make_env(1021)
    from cards import CardInstance

    actor = env.engine.current_player
    source = CardInstance("Basic")
    other = CardInstance("Bone")
    env.engine.players[actor].hand = [source, other]
    env.engine.pending_choice = {
        "player_id": actor,
        "choice_type": "choose_card_from_hand",
        "choice_params": {"cancellable": False},
        "card": source.to_dict(),
        "target_player_id": actor,
        "hand_cards": [source.to_dict(), other.to_dict()],
    }

    assert env.legal_actions(actor) == [Action("select_choice", {"candidate_slot": 0})]
    assert env.observe(actor)["pending"]["candidates"][0]["def_id"] == "Bone"


def test_empty_engine_candidate_snapshot_does_not_fall_back_to_live_hand():
    env = make_env(1022)
    from cards import CardInstance

    actor = env.engine.current_player
    source = CardInstance("Basic")
    live_but_ineligible = CardInstance("Bone")
    env.engine.players[actor].hand = [source, live_but_ineligible]
    env.engine.pending_choice = {
        "player_id": actor,
        "choice_type": "choose_card_from_hand",
        "choice_params": {"cancellable": True},
        "card": source.to_dict(),
        "target_player_id": actor,
        "hand_cards": [],
    }

    assert env.legal_actions(actor) == [
        Action("resolve_choice", {"choice": {"cancelled": True}}),
    ]
    assert env.observe(actor)["pending"].get("candidates", []) == []


def test_successive_same_source_choice_windows_refresh_candidate_slots():
    env = make_env(103)
    from cards import CardInstance

    actor = env.engine.current_player
    source = CardInstance("Basic")
    first = CardInstance("Bone")
    second = CardInstance("Stinger")
    env.engine.players[actor].hand = [first, second]

    def pending_for(card):
        return {
            "player_id": actor,
            "choice_type": "choose_card_from_hand",
            "choice_params": {"cancellable": False},
            "card": source.to_dict(),
            "target_player_id": actor,
            "hand_cards": [card.to_dict()],
        }

    env.engine.pending_choice = pending_for(first)
    assert env.observe(actor)["pending"]["candidates"][0]["def_id"] == "Bone"
    env.engine.pending_choice = pending_for(second)
    actions = env.legal_actions(actor)
    assert actions == [Action("select_choice", {"candidate_slot": 0})]
    assert env.observe(actor)["pending"]["candidates"][0]["def_id"] == "Stinger"


def test_target_with_empty_mandatory_followup_picker_is_not_legal():
    env = Garden1v1Env(enabled_mods=["Desert Cards Addition.gtnmod"], seed=104, include_pregame=False)
    env.reset()
    from cards import CardInstance

    actor = env.engine.current_player
    target = 1 - actor
    magnet = CardInstance("Magnet")
    env.engine.players[actor].hand = [magnet]
    env.engine.players[actor].elixir = 20
    env.engine.players[target].hand = []

    play_targets = {
        action.payload.get("choice", {}).get("target_player_id")
        for action in env.legal_actions(actor)
        if action.kind == "play_card" and action.payload["hand_slot"] == 0
    }
    assert target not in play_targets

    env.engine.players[target].hand = [CardInstance("Basic")]
    play_targets = {
        action.payload.get("choice", {}).get("target_player_id")
        for action in env.legal_actions(actor)
        if action.kind == "play_card" and action.payload["hand_slot"] == 0
    }
    assert target in play_targets


def test_empty_optional_picker_is_not_exposed_as_a_semantic_play_action():
    env = make_env(105)
    from cards import CardInstance

    actor = env.engine.current_player
    player = env.engine.players[actor]
    player.hand = [CardInstance("Sewage")]
    player.elixir = 20
    env.engine.players[0].equipment = []
    env.engine.players[1].equipment = []

    assert env.legal_actions(actor) == [Action("end_turn")]


def test_sapphire_picker_can_keep_self_as_target_when_enemy_is_unselectable():
    env = Garden1v1Env(enabled_mods=["Ocean Cards Addition.gtnmod"], seed=106, include_pregame=False)
    env.reset()
    from cards import CardInstance

    actor = env.engine.current_player
    enemy = 1 - actor
    sapphire = CardInstance("Sapphire")
    attack = CardInstance("Basic")
    env.engine.players[actor].hand = [sapphire, attack]
    env.engine.players[enemy].untargetable = True
    env.engine.pending_choice = {
        "player_id": actor,
        "choice_type": "choose_ocean_sapphire",
        "choice_params": {"cancellable": True},
        "card": sapphire.to_dict(),
        "target_player_id": actor,
        "hand_cards": [attack.to_dict()],
    }

    choices = [
        action
        for action in env.legal_actions(actor)
        if action.kind == "select_choice"
    ]
    assert choices == [Action("select_choice", {
        "candidate_slot": 0,
        "target_player_id": actor,
    })]


def test_mimic_picker_excludes_targets_with_unaffordable_special_cost():
    env = make_env(107)
    from cards import CardInstance

    actor = env.engine.current_player
    mimic = CardInstance("Mimic")
    affordable = CardInstance("Basic")
    expensive = CardInstance("Bone")
    expensive.power_value = 4
    env.engine.players[actor].hand = [mimic, affordable, expensive]
    env.engine.players[actor].elixir = 1
    env.engine.pending_choice = {
        "player_id": actor,
        "choice_type": "choose_card_from_hand",
        "choice_params": {"cancellable": True},
        "card": mimic.to_dict(),
        "target_player_id": actor,
        "hand_cards": [affordable.to_dict(), expensive.to_dict()],
    }

    actions = [action for action in env.legal_actions(actor) if action.kind == "select_choice"]
    assert actions == [Action("select_choice", {"candidate_slot": 0})]
    assert env.observe(actor)["pending"]["candidates"][0]["def_id"] == "Basic"


def test_unpaid_mimic_picker_reserves_source_and_special_costs():
    env = make_env(1071)
    from cards import CardInstance

    actor = env.engine.current_player
    mimic = CardInstance("Mimic")
    mimic.temp_heavy_value = 1
    mimic.instance_flags.add("temp_heavy")
    target = CardInstance("Basic")
    target.temp_heavy_value = 1
    target.instance_flags.add("temp_heavy")
    env.engine.players[actor].hand = [mimic, target]
    source_cost = max(0, mimic.cost_e + env.engine._get_extra_e_for_card(actor, mimic))
    special_cost = env.engine._mimic_special_cost_for_card(target)
    assert source_cost > 0 and special_cost > 0
    env.engine.pending_choice = {
        "player_id": actor,
        "choice_type": "choose_card_from_hand",
        "choice_params": {"cancellable": True},
        "card": mimic.to_dict(),
        "target_player_id": actor,
        "hand_cards": [target.to_dict()],
        "already_paid": False,
    }

    env.engine.players[actor].elixir = source_cost + special_cost - 1
    assert not [action for action in env.legal_actions(actor) if action.kind == "select_choice"]

    env.engine.players[actor].elixir += 1
    env._reset_choice_builder()
    assert [action for action in env.legal_actions(actor) if action.kind == "select_choice"] == [
        Action("select_choice", {"candidate_slot": 0})
    ]


def test_dynamic_wide_strike_v2_target_is_resolved_without_policy_target_action():
    env = Garden1v1Env(enabled_mods=None, seed=108, include_pregame=False)
    env.reset()
    from cards import CardInstance

    actor = env.engine.current_player
    target = 1 - actor
    date = CardInstance("Date")
    date.instance_flags.add("wide_strike")
    env.engine.players[actor].hand = [date]
    env.engine.players[actor].elixir = 20
    before = env.engine.players[target].health

    actions = [action for action in env.legal_actions(actor) if action.kind == "play_card"]
    assert actions == [Action("play_card", {"hand_slot": 0})]
    env.step(actions[0], actor)
    assert env.engine.players[target].health < before


def test_v2_ui_actions_use_stable_slots_and_execute_through_production_validator():
    env = make_env(109)
    from cards import CardInstance
    from mod_runtime_v2 import _sanitize_ui_component

    actor = env.decision_player()
    target = 1 - actor
    cards = [CardInstance("Basic"), CardInstance("Bone"), CardInstance("Stinger")]
    env.engine.players[actor].hand = cards
    context = {"source_player": actor, "target_player": target, "vars": {}}
    component = _sanitize_ui_component(env.engine, context, {
        "type": "modal",
        "controls": [
            {
                "id": "mode",
                "type": "select",
                "options": [
                    {"value": "alpha", "label": "Alpha"},
                    {"value": "beta", "label": "Beta"},
                ],
            },
            {
                "id": "cards",
                "type": "multi_card_picker",
                "target": actor,
                "zone": "hand",
                "allowed_instance_ids": [card.instance_id for card in cards],
                "min_select": 2,
                "max_select": 2,
            },
            {
                "id": "target",
                "type": "player_picker",
                "allowed_player_ids": [actor, target],
            },
        ],
        "buttons": [
            {"id": "confirm", "role": "confirm"},
            {"id": "cancel", "role": "cancel"},
        ],
    })
    env.engine.pending_v2_ui = {
        "request_id": "test-v2-ui",
        "player_id": actor,
        "component": component,
        "context": context,
        "remaining_steps": [],
        "on_cancel": [],
    }

    actions = env.legal_actions(actor)
    assert len(actions) == 13
    assert all(action.kind == "v2_ui_response" for action in actions)
    assert "instance_id" not in json.dumps(
        [action.to_dict() for action in actions],
        ensure_ascii=False,
    )
    target_option_slot = next(
        slot for slot, option in enumerate(component["controls"][2]["options"])
        if int(option["value"]) == target
    )
    selected = next(
        action for action in actions
        if action.payload["button_slot"] == 0
        and action.payload["controls"][0]["option_slot"] == 1
        and action.payload["controls"][1]["option_slots"] == [1, 2]
        and action.payload["controls"][2]["option_slot"] == target_option_slot
    )
    observation = env.observe(actor)
    assert observation["pending"]["actor"] == actor
    assert observation["pending"]["component_type"] == "modal"
    assert len(observation["pending"]["controls"]) == 3
    assert not _keys_matching(observation["pending"], "instance")
    assert {candidate.get("def_id") for candidate in observation["pending"]["candidates"]} >= {
        "Basic", "Bone", "Stinger",
    }
    private_view = env.observe(target)["pending"]
    assert private_view == {"kind": "mod_ui", "actor": actor, "private": True}
    from gtn_ai.structured_features import _selected_visible_card
    assert _selected_visible_card(observation, selected)["def_id"] == "Bone"

    result = env.step(selected, actor)

    assert not result.terminated
    assert env.engine.pending_v2_ui is None
    values = context["current_action"]["v2_ui"]["values"]
    assert values == {
        "mode": "beta",
        "cards": [cards[1].instance_id, cards[2].instance_id],
        "target": target,
    }


def test_v2_ui_cancel_has_valid_defaults_for_required_controls():
    env = make_env(110)
    from mod_runtime_v2 import _sanitize_ui_component

    actor = env.decision_player()
    context = {"source_player": actor, "target_player": actor, "vars": {}}
    component = _sanitize_ui_component(env.engine, context, {
        "type": "modal",
        "controls": [{"id": "amount", "type": "number", "min": 2, "max": 4, "default": 3}],
        "buttons": [
            {"id": "confirm", "role": "confirm"},
            {"id": "cancel", "role": "cancel"},
        ],
    })
    env.engine.pending_v2_ui = {
        "request_id": "test-v2-cancel",
        "player_id": actor,
        "component": component,
        "context": context,
        "remaining_steps": [],
        "on_cancel": [],
    }

    cancel = next(action for action in env.legal_actions(actor) if action.payload["button_slot"] == 1)
    assert cancel.payload["controls"] == [{"control_slot": 0, "value": 3}]
    env.step(cancel, actor)
    assert env.engine.pending_v2_ui is None


def _keys_matching(value, fragment: str) -> list[str]:
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if fragment in str(key).lower():
                result.append(str(key))
            result.extend(_keys_matching(item, fragment))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_keys_matching(item, fragment))
    return result
