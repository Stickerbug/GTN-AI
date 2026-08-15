import gzip
import pickle

from gtn_ai.diagnostics import read_jsonl
from gtn_ai.environment import Garden1v1Env
from gtn_ai.live_worker import (
    LiveDecisionService,
    _filter_unambiguous_bad_targets,
    decode_engine,
    encode_engine,
)
from gtn_ai.protocol import Action


def test_live_worker_decides_and_records_with_a_production_snapshot(tmp_path) -> None:
    env = Garden1v1Env(
        seed=31,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    env.reset()
    original_snapshot = encode_engine(env.engine)
    actor = env.decision_player()
    service = LiveDecisionService(
        policy_name="heuristic",
        game_root=env.game_root,
        diagnostics_root=tmp_path,
    )

    result = service.decide({
        "session_id": "live-worker-test",
        "engine_snapshot": original_snapshot,
        "player_id": actor,
        "seed": 7,
        "execute": True,
    })

    assert result["success"] is True
    assert result["action"] in [action.to_dict() for action in env.legal_actions(actor)]
    assert result["decision_id"] == 0
    assert result["executed"] is True
    assert result["engine_snapshot"]
    assert encode_engine(env.engine) == original_snapshot
    finished = service.finish({
        "session_id": "live-worker-test",
        "outcome": {"winner": actor},
    })
    assert finished["export"].endswith(".gtnai.zip")


def test_live_worker_records_an_external_human_decision(tmp_path) -> None:
    env = Garden1v1Env(
        seed=47,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    action = env.legal_actions(actor)[0]
    service = LiveDecisionService(
        policy_name="heuristic",
        game_root=env.game_root,
        diagnostics_root=tmp_path,
    )

    result = service.record({
        "session_id": "live-human-record-test",
        "engine_snapshot": encode_engine(env.engine),
        "player_id": actor,
        "actor_kind": "human",
        "action": action.to_dict(),
    })

    assert result["success"] is True
    assert result["decision_id"] == 0
    assert result["canonicalized"] is True
    assert result["canonical_action"] == action.to_dict()
    finished = service.finish({
        "session_id": "live-human-record-test",
        "outcome": {"reason": "test"},
    })
    assert finished["export"].endswith(".gtnai.zip")


def test_live_worker_snapshot_removes_player_controlled_names(tmp_path) -> None:
    env = Garden1v1Env(
        seed=49,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    env.reset()
    env.engine.player_names = ["Private Human Name", "Phelren"]
    actor = env.decision_player()
    service = LiveDecisionService(
        policy_name="heuristic",
        game_root=env.game_root,
        diagnostics_root=tmp_path,
        export_finished=False,
    )

    service.decide({
        "session_id": "sanitized-name-test",
        "engine_snapshot": encode_engine(env.engine),
        "player_id": actor,
        "seed": 9,
        "execute": False,
    })
    decision = next(read_jsonl(tmp_path / "sanitized-name-test" / "decisions.jsonl"))
    with gzip.open(tmp_path / "sanitized-name-test" / decision["snapshot"], "rb") as handle:
        snapshot = pickle.load(handle)

    assert snapshot.player_names == ["Player 1", "Player 2"]
    assert env.engine.player_names == ["Private Human Name", "Phelren"]
    finished = service.finish({"session_id": "sanitized-name-test"})
    assert finished["export"] is None


def test_live_worker_executes_a_v2_ui_decision(tmp_path) -> None:
    env = Garden1v1Env(
        seed=53,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    env.reset()
    from mod_runtime_v2 import _sanitize_ui_component

    actor = env.decision_player()
    context = {"source_player": actor, "target_player": actor, "vars": {}}
    component = _sanitize_ui_component(env.engine, context, {
        "type": "modal",
        "controls": [{
            "id": "mode",
            "type": "select",
            "options": [
                {"value": "first", "label": "First"},
                {"value": "second", "label": "Second"},
            ],
        }],
        "buttons": [
            {"id": "confirm", "role": "confirm"},
            {"id": "cancel", "role": "cancel"},
        ],
    })
    env.engine.pending_v2_ui = {
        "request_id": "live-worker-v2-test",
        "player_id": actor,
        "component": component,
        "context": context,
        "remaining_steps": [],
        "on_cancel": [],
    }
    service = LiveDecisionService(
        policy_name="heuristic",
        game_root=env.game_root,
        diagnostics_root=tmp_path,
    )

    result = service.decide({
        "session_id": "live-worker-v2-test",
        "engine_snapshot": encode_engine(env.engine),
        "player_id": actor,
        "seed": 11,
        "execute": True,
    })

    assert result["success"] is True
    assert result["action"]["kind"] == "v2_ui_response"
    assert result["action"]["payload"]["button_role"] == "confirm"
    assert result["legal_action_count"] == 3
    assert decode_engine(result["engine_snapshot"]).pending_v2_ui is None


def test_live_worker_canonicalizes_browser_card_instance_payload(tmp_path) -> None:
    env = Garden1v1Env(
        seed=59,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    canonical = next(action for action in env.legal_actions(actor) if action.kind == "play_card")
    card = env.engine.players[actor].hand[canonical.payload["hand_slot"]]
    raw_payload = {
        "card_instance_id": card.instance_id,
        "choice": dict(canonical.payload.get("choice") or {}),
    }
    service = LiveDecisionService(
        policy_name="heuristic",
        game_root=env.game_root,
        diagnostics_root=tmp_path,
    )

    result = service.record({
        "session_id": "live-browser-card-record-test",
        "engine_snapshot": encode_engine(env.engine),
        "player_id": actor,
        "actor_kind": "human",
        "action": {"kind": "play_card", "payload": raw_payload},
    })

    assert result["canonicalized"] is True
    assert result["canonical_action"] == canonical.to_dict()
    assert "instance_id" not in str(result["canonical_action"])


def test_live_worker_canonicalizes_browser_v2_response(tmp_path) -> None:
    env = Garden1v1Env(
        seed=61,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    env.reset()
    from mod_runtime_v2 import _sanitize_ui_component

    actor = env.decision_player()
    context = {"source_player": actor, "target_player": actor, "vars": {}}
    component = _sanitize_ui_component(env.engine, context, {
        "type": "modal",
        "controls": [{
            "id": "mode",
            "type": "select",
            "options": [
                {"value": "first", "label": "First"},
                {"value": "second", "label": "Second"},
            ],
        }],
        "buttons": [
            {"id": "confirm", "role": "confirm"},
            {"id": "cancel", "role": "cancel"},
        ],
    })
    env.engine.pending_v2_ui = {
        "request_id": "live-browser-v2-record",
        "player_id": actor,
        "component": component,
        "context": context,
        "remaining_steps": [],
        "on_cancel": [],
    }
    service = LiveDecisionService(
        policy_name="heuristic",
        game_root=env.game_root,
        diagnostics_root=tmp_path,
    )

    result = service.record({
        "session_id": "live-browser-v2-record-test",
        "engine_snapshot": encode_engine(env.engine),
        "player_id": actor,
        "actor_kind": "human",
        "action": {
            "kind": "v2_ui_response",
            "payload": {"button": "confirm", "values": {"mode": "second"}},
        },
    })

    assert result["canonicalized"] is True
    assert result["canonical_action"]["kind"] == "v2_ui_response"
    assert result["canonical_action"]["payload"]["controls"] == [
        {"control_slot": 0, "option_slot": 1},
    ]


def test_live_worker_filters_only_unambiguous_reversed_equipment_targets() -> None:
    observation = {
        "seat": 0,
        "self": {
            "hand": [
                {"slot": 0, "def_id": "Pincer"},
                {"slot": 1, "def_id": "Powder"},
                {"slot": 2, "def_id": "GoldenLeaf"},
            ],
        },
    }
    actions = [
        Action("play_card", {"hand_slot": slot, "choice": {"target_player_id": target}})
        for slot in range(3)
        for target in (0, 1)
    ]

    filtered = _filter_unambiguous_bad_targets(observation, actions)
    keys = {action.key for action in filtered}

    assert actions[0].key not in keys  # Pincer -> self
    assert actions[1].key in keys      # Pincer -> opponent
    assert actions[2].key in keys      # Powder -> self
    assert actions[3].key not in keys  # Powder -> opponent
    assert actions[4].key in keys      # Other equipment remains unconstrained.
    assert actions[5].key in keys
