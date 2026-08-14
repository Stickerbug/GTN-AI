from __future__ import annotations

import json

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.inference_server import DecisionService, SERVICE_API_VERSION
from gtn_ai.neural_model import NeuralModelConfig, NeuralPolicy, torch_available
from gtn_ai.neural_training import (
    inspect_trajectory_files,
    iter_encoded_decisions,
    train_neural_behavior_policy,
)
from gtn_ai.policies import RandomPolicy
from gtn_ai.self_play import play_episode


pytestmark = pytest.mark.skipif(not torch_available(), reason="PyTorch training extra is not installed")


def _small_config() -> NeuralModelConfig:
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


def test_streaming_neural_training_checkpoint_and_service_round_trip(tmp_path):
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=901)
    episode = play_episode(
        env,
        [RandomPolicy(11), RandomPolicy(12)],
        max_steps=1500,
        record_decisions=True,
    )
    assert episode.terminated
    trajectory = tmp_path / "episodes.jsonl"
    trajectory.write_text(
        json.dumps(episode.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    inspected = inspect_trajectory_files([trajectory])
    assert inspected["usable_decisions"] == len([
        item for item in episode.decisions if not item.forced_fallback
    ])
    weighted = list(iter_encoded_decisions(
        [trajectory],
        config=_small_config(),
        validation_fraction=0,
        winner_policy_weight=2.0,
        loser_policy_weight=0.1,
        draw_policy_weight=0.5,
    ))
    expected_weights = {0.5} if episode.winner not in (0, 1) else {0.1, 2.0}
    assert {item.policy_weight for item in weighted} == expected_weights
    policy, metrics = train_neural_behavior_policy(
        [trajectory],
        config=_small_config(),
        epochs=1,
        batch_size=32,
        shuffle_buffer=64,
        validation_fraction=0,
        device="cpu",
        seed=13,
    )
    assert metrics["epoch_metrics"][0]["train"]["examples"] == inspected["usable_decisions"]
    assert metrics["parameter_count"] > 0

    checkpoint = tmp_path / "policy.pt"
    policy.save(checkpoint, metadata=metrics)
    loaded = NeuralPolicy.load(checkpoint, device="cpu", seed=13)
    continued, continued_metrics = train_neural_behavior_policy(
        [trajectory],
        config=None,
        epochs=0,
        device="cpu",
        initial_checkpoint=checkpoint,
    )
    assert continued.config == loaded.config
    assert continued_metrics["initial_checkpoint"] == str(checkpoint.resolve())
    fresh = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=902)
    observation = fresh.reset()
    legal = fresh.legal_actions(fresh.decision_player())
    response = DecisionService(loaded).decide({
        "api_version": SERVICE_API_VERSION,
        "observation": observation,
        "legal_actions": [action.to_dict() for action in legal],
    })
    assert response["action"] in [action.to_dict() for action in legal]
