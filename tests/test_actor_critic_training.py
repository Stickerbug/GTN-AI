from __future__ import annotations

import json

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.actor_critic_training import (
    generalized_advantage_targets,
    inspect_on_policy_trajectories,
    train_actor_critic_policy,
)
from gtn_ai.neural_model import (
    NeuralModelConfig,
    NeuralPolicy,
    VariableActionNetwork,
    torch_available,
)
from gtn_ai.self_play import play_episode


pytestmark = pytest.mark.skipif(not torch_available(), reason="PyTorch training extra is not installed")


def test_generalized_advantage_lambda_one_matches_terminal_return():
    monte_carlo = generalized_advantage_targets(
        [0.2, 0.4],
        1.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert monte_carlo == pytest.approx([(0.8, 1.0), (0.6, 1.0)])

    shaped = generalized_advantage_targets(
        [0.2, 0.4],
        1.0,
        gamma=1.0,
        gae_lambda=0.5,
    )
    assert shaped == pytest.approx([(0.5, 0.7), (0.6, 1.0)])


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


def test_actor_critic_training_accepts_exact_on_policy_trajectory(tmp_path):
    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=951)
    env.reset()
    template = NeuralPolicy(
        VariableActionNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=1,
    )
    checkpoint = tmp_path / "initial.pt"
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
    trajectory = tmp_path / "on-policy.jsonl"
    trajectory.write_text(
        json.dumps(episode.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    inspected = inspect_on_policy_trajectories(
        [trajectory], expected_policy_fingerprint=template.model_fingerprint
    )
    assert inspected["decisions"] == len(episode.decisions)
    candidate, metrics = train_actor_critic_policy(
        [trajectory],
        initial_checkpoint=checkpoint,
        epochs=1,
        batch_size=32,
        shuffle_buffer=64,
        learning_rate=1e-5,
        target_kl=0,
        device="cpu",
        seed=4,
    )
    assert metrics["epoch_metrics"][0]["examples"] == len(episode.decisions)
    assert metrics["initial_policy_fingerprint"] == template.model_fingerprint
    assert candidate.ruleset_fingerprint == env.ruleset_fingerprint


def test_actor_critic_rejects_a_different_policy_fingerprint(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_on_policy_trajectories([path], expected_policy_fingerprint="wrong")


def test_actor_critic_ignores_an_unrecorded_opponent_policy(tmp_path):
    config = _small_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=952)
    env.reset()
    actor = NeuralPolicy(
        VariableActionNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=5,
    )
    from gtn_ai.policies import RandomPolicy

    episode = play_episode(
        env,
        [actor.fork(seed=6, temperature=0.8, record_behavior=True), RandomPolicy(7)],
        max_steps=1500,
        record_decisions=True,
        max_state_repeats=0,
        max_turn_decisions=0,
    )
    trajectory = tmp_path / "mixed.jsonl"
    trajectory.write_text(
        json.dumps(episode.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    inspected = inspect_on_policy_trajectories(
        [trajectory], expected_policy_fingerprint=actor.model_fingerprint
    )
    actor_decisions = sum(item.player == 0 for item in episode.decisions)
    assert inspected["decisions"] == actor_decisions
    assert inspected["ignored_opponent_decisions"] == len(episode.decisions) - actor_decisions
