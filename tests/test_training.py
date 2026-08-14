from __future__ import annotations

from gtn_ai import Garden1v1Env
from gtn_ai.linear_model import train_behavior_cloning, train_monte_carlo
from gtn_ai.policies import RandomPolicy
from gtn_ai.protocol import Action
from gtn_ai.self_play import _sample_policy_pair, _timeout_fallback_action, play_episode


def test_league_policy_pair_sampling_is_reproducible():
    pool = ("heuristic", "random", "heuristic")
    assert _sample_policy_pair(pool, 44) == _sample_policy_pair(pool, 44)
    assert all(name in pool for name in _sample_policy_pair(pool, 45))


def test_recorded_episode_trains_a_ruleset_locked_policy():
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=121)
    episode = play_episode(
        env,
        [RandomPolicy(1), RandomPolicy(2)],
        max_steps=1500,
        record_decisions=True,
    )
    assert episode.terminated
    assert episode.ruleset_fingerprint == env.ruleset_fingerprint
    assert episode.decisions
    assert "teacher" not in episode.to_dict()["decisions"][0]

    policy, metrics = train_monte_carlo(
        [episode.to_dict()],
        buckets=1024,
        epochs=1,
        seed=3,
    )
    trainable_decisions = [decision for decision in episode.decisions if not decision.forced_fallback]
    assert metrics["examples"] == len(trainable_decisions)
    assert policy.ruleset_fingerprint == env.ruleset_fingerprint

    actor = env.decision_player(default=0)
    observation = env.observe(actor)
    if not env.engine.game_over:
        legal = env.legal_actions(actor)
        assert policy.select_action(observation, legal) in legal


def test_timeout_fallback_prefers_progress_over_cancelling():
    legal = [
        Action("resolve_choice", {"choice": {"cancelled": True}}),
        Action("select_choice", {"candidate_slot": 0}),
    ]
    observation = {"pending": {"kind": "choice", "selection": {"selected_slots": []}}}
    assert _timeout_fallback_action(observation, legal) == legal[1]


def test_behavior_cloning_uses_complete_legal_action_sets():
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=122)
    episode = play_episode(
        env,
        [RandomPolicy(4), RandomPolicy(5)],
        max_steps=1500,
        record_decisions=True,
    )
    policy, metrics = train_behavior_cloning(
        [episode.to_dict()],
        buckets=1024,
        epochs=1,
        seed=6,
    )
    assert metrics["objective"] == "behavior_cloning"
    assert metrics["examples"] > 0
    assert metrics["phase_examples"]["pregame"] == episode.pregame_steps
    assert metrics["phase_examples"]["combat"] == episode.combat_steps
    assert set(metrics["phase_accuracies"]) == {"pregame", "combat"}
    assert policy.name == "hashed-linear-bc-v1"


def test_training_rejects_mixed_rulesets():
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=123)
    episode = play_episode(
        env,
        [RandomPolicy(7), RandomPolicy(8)],
        max_steps=1500,
        record_decisions=True,
    ).to_dict()
    other = dict(episode)
    other["ruleset_fingerprint"] = "different-rules"
    import pytest

    with pytest.raises(ValueError, match="mixes different game rulesets"):
        train_behavior_cloning([episode, other], buckets=256, epochs=1)
