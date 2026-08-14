from __future__ import annotations

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.neural_model import (
    NeuralModelConfig,
    NeuralEnsemblePolicy,
    NeuralPolicy,
    VariableActionNetwork,
    _apply_progress_prior,
    collate_decisions,
    encode_decision,
    torch_available,
)
from gtn_ai.protocol import Action


pytestmark = pytest.mark.skipif(not torch_available(), reason="PyTorch training extra is not installed")


def _small_config():
    return NeuralModelConfig(
        observation_buckets=256,
        action_buckets=128,
        history_buckets=64,
        observation_embedding_dim=16,
        action_embedding_dim=16,
        history_embedding_dim=8,
        hidden_dim=24,
        max_history_events=8,
        max_history_tokens_per_event=8,
        dropout=0,
    )


def test_variable_action_network_handles_different_legal_set_sizes():
    import torch

    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=801)
    observation = env.reset()
    actor = env.decision_player()
    legal = env.legal_actions(actor)
    first = encode_decision(
        observation,
        legal,
        config=config,
        selected_index=0,
        policy_weight=2.5,
    )
    second = encode_decision(observation, legal[:1], config=config, selected_index=0)
    batch = collate_decisions([first, second])
    model = VariableActionNetwork(config)
    scores, values = model(batch)
    assert scores.shape == (len(legal) + 1,)
    assert values.shape == (2,)
    assert batch["policy_weights"].tolist() == [2.5, 1.0]
    assert torch.isfinite(scores).all()
    assert torch.isfinite(values).all()


def test_neural_checkpoint_round_trip_returns_only_a_legal_action(tmp_path):
    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=802)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    policy = NeuralPolicy(
        VariableActionNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=3,
    )
    path = tmp_path / "policy.pt"
    policy.save(path, metadata={"purpose": "test"})
    loaded = NeuralPolicy.load(path, device="cpu", seed=3)
    assert loaded.select_action(observation, legal).key in {action.key for action in legal}
    assert -1.0 <= loaded.estimate_value(observation, legal) <= 1.0


def test_neural_policy_rejects_a_ruleset_mismatch():
    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=803)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    policy = NeuralPolicy(
        VariableActionNetwork(config),
        config=config,
        ruleset_fingerprint="different-rules",
        device="cpu",
    )
    with pytest.raises(ValueError, match="ruleset fingerprint"):
        policy.select_action(observation, legal)


def test_neural_ensemble_returns_a_legal_action():
    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=804)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    members = [
        NeuralPolicy(
            VariableActionNetwork(config),
            config=config,
            ruleset_fingerprint=env.ruleset_fingerprint,
            device="cpu",
            seed=seed,
        )
        for seed in (1, 2)
    ]
    ensemble = NeuralEnsemblePolicy(members, seed=3)
    assert ensemble.select_action(observation, legal) in legal
    assert -1.0 <= ensemble.estimate_value(observation, legal) <= 1.0


def test_progress_prior_penalizes_only_deselecting_an_existing_choice():
    observation = {
        "pending": {"selection": {"selected_slots": [2]}},
        "self": {},
    }
    actions = [
        Action("toggle_choice", {"candidate_slot": 1}),
        Action("toggle_choice", {"candidate_slot": 2}),
        Action("submit_choice"),
    ]
    assert _apply_progress_prior(observation, actions, [1.0, 2.0, 3.0]) == [1.0, -6.0, 3.0]


def test_on_policy_sampling_records_finite_behavior_statistics():
    import math

    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=805)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    policy = NeuralPolicy(
        VariableActionNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=4,
        temperature=0.8,
        record_behavior=True,
    )
    action = policy.select_action(observation, legal)
    metadata = policy.last_decision_metadata
    assert action in legal
    assert metadata is not None
    assert metadata["action_key"] == action.key
    assert all(math.isfinite(float(metadata[key])) for key in (
        "log_prob", "value", "entropy", "temperature"
    ))
    assert metadata["log_prob"] <= 0
    assert metadata["temperature"] == 0.8
