from __future__ import annotations

import json

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.neural_model import torch_available
from gtn_ai.self_play import play_episode
from gtn_ai.structured_actor_critic_training import (
    train_structured_actor_critic_policy,
)
from gtn_ai.structured_model import (
    StructuredModelConfig,
    StructuredPolicy,
    StructuredPolicyNetwork,
)


pytestmark = pytest.mark.skipif(
    not torch_available(), reason="PyTorch training extra is not installed"
)


def _small_config() -> StructuredModelConfig:
    return StructuredModelConfig(
        categorical_buckets=256,
        categorical_slots=8,
        numeric_buckets=8,
        max_state_tokens=64,
        max_history_events=8,
        model_dim=24,
        num_heads=4,
        state_layers=1,
        action_layers=1,
        feedforward_dim=48,
        dropout=0,
    )


def test_structured_actor_critic_accepts_exact_on_policy_trajectory(tmp_path):
    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1961)
    env.reset()
    template = StructuredPolicy(
        StructuredPolicyNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=1,
    )
    checkpoint = tmp_path / "structured-initial.pt"
    template.save(checkpoint)
    episode = play_episode(
        env,
        [
            template.fork(seed=2, temperature=0.8, record_behavior=True),
            template.fork(seed=3, temperature=0.8, record_behavior=True),
        ],
        max_steps=1500,
        record_decisions=True,
        max_state_repeats=0,
        max_turn_decisions=0,
    )
    assert episode.terminated
    assert episode.loop_recoveries == 0
    trajectory = tmp_path / "structured-on-policy.jsonl"
    trajectory.write_text(
        json.dumps(episode.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    candidate, metrics = train_structured_actor_critic_policy(
        [trajectory],
        initial_checkpoint=checkpoint,
        epochs=1,
        batch_size=16,
        shuffle_buffer=32,
        learning_rate=1e-5,
        target_kl=0,
        device="cpu",
        seed=4,
        show_progress=False,
    )
    assert metrics["dataset"]["decisions"] == len(episode.decisions)
    assert metrics["epoch_metrics"][0]["examples"] == len(episode.decisions)
    assert metrics["initial_policy_fingerprint"] == template.model_fingerprint
    assert candidate.ruleset_fingerprint == env.ruleset_fingerprint
