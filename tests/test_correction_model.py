from __future__ import annotations

import json

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.correction_model import (
    CorrectionModelConfig,
    StructuredCorrectionNetwork,
    StructuredCorrectionPolicy,
)
from gtn_ai.correction_training import train_structured_correction
from gtn_ai.neural_model import torch_available
from gtn_ai.policies import policy_from_name
from gtn_ai.structured_cache import build_recorded_teacher_cache
from gtn_ai.structured_model import (
    StructuredModelConfig,
    StructuredPolicy,
    StructuredPolicyNetwork,
)


pytestmark = pytest.mark.skipif(
    not torch_available(), reason="PyTorch training extra is not installed"
)


def _config() -> StructuredModelConfig:
    return StructuredModelConfig(
        categorical_buckets=256,
        categorical_slots=16,
        numeric_buckets=16,
        max_state_tokens=64,
        max_history_events=8,
        model_dim=32,
        num_heads=4,
        state_layers=1,
        action_layers=1,
        feedforward_dim=64,
        dropout=0,
    )


def test_correction_checkpoint_round_trip_and_policy_parser(tmp_path):
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=2101)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    config = _config()
    correction_config = CorrectionModelConfig(
        hidden_dim=24,
        dropout=0,
        top_k=3,
        gate_threshold=0.5,
    )
    policy = StructuredCorrectionPolicy(
        StructuredPolicyNetwork(config),
        StructuredCorrectionNetwork(config, correction_config),
        base_config=config,
        correction_config=correction_config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=2,
    )
    path = tmp_path / "correction.pt"
    policy.save(path, metadata={"purpose": "test"})

    loaded = StructuredCorrectionPolicy.load(path, device="cpu", seed=2)
    action = loaded.select_action(observation, legal)
    assert action in legal
    assert len(loaded.evaluate_actions(observation, legal)[0]) == len(legal)
    assert len(loaded.estimate_values([(observation, legal), (observation, legal[:1])])) == 2
    assert loaded.diagnostics()["decisions"] == 1

    parsed = policy_from_name(f"correction-cpu:{path}", seed=3)
    assert parsed.select_action(observation, legal) in legal


def test_correction_training_builds_standalone_policy(tmp_path):
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=2102)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    assert len(legal) >= 2
    config = _config()
    base = StructuredPolicy(
        StructuredPolicyNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=3,
    )
    base_path = tmp_path / "base.pt"
    base.save(base_path)

    decisions = []
    for index in range(20):
        selected_index = index % min(2, len(legal))
        logits = [-1.0] * len(legal)
        logits[selected_index] = 0.2
        decisions.append({
            "step": index,
            "player": env.decision_player(),
            "observation": observation,
            "legal_actions": [action.to_dict() for action in legal],
            "action": legal[selected_index].to_dict(),
            "forced_fallback": False,
            "teacher": {
                "kind": "test_preference",
                "action_key": legal[selected_index].key,
                "logits": logits,
                "value": 0.0,
                "search_margin": 0.2,
            },
        })
    trajectory = tmp_path / "teacher.jsonl"
    trajectory.write_text(json.dumps({
        "schema_version": 4,
        "seed": 2102,
        "official_mods": ["Vanilla Cards.gtnmod"],
        "loadout_fingerprint": env.loadout_fingerprint,
        "ruleset_fingerprint": env.ruleset_fingerprint,
        "policies": ["teacher", "teacher"],
        "winner": 0,
        "terminated": True,
        "truncated": False,
        "steps": len(decisions),
        "rounds": 0,
        "loop_recoveries": 0,
        "decisions": decisions,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    cache = tmp_path / "cache"
    build_recorded_teacher_cache(
        [trajectory],
        output_dir=cache,
        config=config,
        shard_size=5,
    )
    output = tmp_path / "trained-correction.pt"
    trained, metrics = train_structured_correction(
        cache,
        base_checkpoint=base_path,
        output_checkpoint=output,
        correction_config=CorrectionModelConfig(
            hidden_dim=24,
            dropout=0,
            top_k=2,
        ),
        epochs=1,
        batch_size=4,
        shuffle_buffer=8,
        validation_fraction=0.2,
        device="cpu",
        seed=4,
        show_progress=False,
    )
    assert output.is_file()
    assert metrics["objective"] == "frozen_base_preference_correction"
    assert metrics["parameter_count"] < sum(
        parameter.numel() for parameter in trained.base_model.parameters()
    )
    assert metrics["final_validation"]["examples"] > 0
    assert trained.select_action(observation, legal) in legal
