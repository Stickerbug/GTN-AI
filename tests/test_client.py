from __future__ import annotations

from gtn_ai import Garden1v1Env
from gtn_ai.client import InferenceClient
from gtn_ai.policies import HeuristicPolicy


def test_unavailable_sidecar_uses_verified_fallback():
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=131)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    client = InferenceClient(
        "http://127.0.0.1:1/v1/decide",
        timeout_seconds=0.01,
        fallback_policy=HeuristicPolicy(seed=1, exploration=0),
    )
    result = client.decide(observation, legal)
    assert result.source == "fallback"
    assert result.action in legal
    assert result.error
