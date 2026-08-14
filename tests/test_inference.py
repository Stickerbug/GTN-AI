from __future__ import annotations

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.inference_server import DecisionService, SERVICE_API_VERSION
from gtn_ai.linear_model import HashedLinearPolicy
from gtn_ai.policies import HeuristicPolicy


def test_decision_service_returns_only_supplied_legal_action():
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=91)
    env.reset()
    actor = env.decision_player()
    legal = env.legal_actions(actor)
    response = DecisionService(HeuristicPolicy(seed=1, exploration=0)).decide({
        "api_version": SERVICE_API_VERSION,
        "observation": env.observe(actor),
        "legal_actions": [action.to_dict() for action in legal],
    })
    assert response["action"] in [action.to_dict() for action in legal]


def test_decision_service_rejects_empty_legal_set():
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=92)
    observation = env.reset()
    with pytest.raises(ValueError, match="must not be empty"):
        DecisionService(HeuristicPolicy(exploration=0)).decide({
            "api_version": SERVICE_API_VERSION,
            "observation": observation,
            "legal_actions": [],
        })


def test_linear_model_round_trip_still_selects_legal_action(tmp_path):
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=93)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    policy = HashedLinearPolicy(weights={1: 0.5, 7: -0.25}, buckets=256, seed=2)
    path = tmp_path / "model.json"
    policy.save(path, metadata={"purpose": "test"})
    loaded = HashedLinearPolicy.load(path, seed=2)
    assert loaded.select_action(observation, legal).key in {action.key for action in legal}


def test_linear_model_rejects_a_different_game_ruleset():
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=94)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    policy = HashedLinearPolicy(ruleset_fingerprint="not-the-current-rules")
    with pytest.raises(ValueError, match="ruleset fingerprint"):
        policy.select_action(observation, legal)
