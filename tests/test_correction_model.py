from __future__ import annotations

import json
from dataclasses import replace

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.correction_model import (
    CorrectionModelConfig,
    StructuredCorrectionNetwork,
    StructuredCorrectionPolicy,
)
from gtn_ai.correction_training import _correction_loss, train_structured_correction
from gtn_ai.neural_model import torch, torch_available
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
    assert loaded.diagnostics()["decisions"] == 2

    parsed = policy_from_name(f"correction-cpu:{path}", seed=3)
    assert parsed.select_action(observation, legal) in legal
    safe = policy_from_name(f"safe-correction-cpu:{path}", seed=3)
    assert safe.select_action(observation, legal) in legal
    assert safe.diagnostics()["policy"].endswith("+progress-safe")
    annotator = policy_from_name(
        f"unsafe-rollout-correction-cpu:{path};"
        "candidates=2;rollouts=1;horizon=1;belief=false;annotate=true",
        seed=3,
    )
    assert isinstance(annotator.base_policy, StructuredCorrectionPolicy)
    assert annotator.config.annotate_only is True


def test_contextual_correction_starts_as_exact_base_policy(tmp_path):
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=2111)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    base_config = _config()
    context_config = replace(base_config, contextual_value_features=True)
    base_model = StructuredPolicyNetwork(base_config)
    base_policy = StructuredPolicy(
        base_model,
        config=base_config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=1,
    )
    context_model = StructuredPolicyNetwork(context_config)
    correction_config = CorrectionModelConfig(
        hidden_dim=24,
        dropout=0,
        top_k=3,
        gate_threshold=0.2,
        contextual_value_features=True,
    )
    correction = StructuredCorrectionPolicy(
        base_model,
        StructuredCorrectionNetwork(
            base_config, correction_config, context_config
        ),
        base_config=base_config,
        correction_config=correction_config,
        context_model=context_model,
        context_config=context_config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=1,
    )

    base_logits, base_value = base_policy.evaluate_actions(observation, legal)
    corrected_logits, corrected_value = correction.evaluate_actions(
        observation, legal
    )
    assert corrected_logits == pytest.approx(base_logits, abs=0.0)
    assert corrected_value == pytest.approx(base_value, abs=0.0)
    assert correction.diagnostics()["action_changes"] == 0

    path = tmp_path / "contextual-correction.pt"
    correction.save(path)
    loaded = StructuredCorrectionPolicy.load(path, device="cpu", seed=1)
    loaded_logits, loaded_value = loaded.evaluate_actions(observation, legal)
    assert loaded_logits == pytest.approx(base_logits, abs=0.0)
    assert loaded_value == pytest.approx(base_value, abs=0.0)
    assert loaded.context_config == context_config


def test_low_margin_teacher_disagreement_trains_toward_base_action():
    correction_logits = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    action_mask = torch.tensor([[True, True]])
    loss, metrics = _correction_loss(
        torch.zeros((1, 2), dtype=torch.float32),
        correction_logits,
        action_mask,
        torch.tensor([0], dtype=torch.long),
        {
            "action_mask": action_mask,
            # The teacher narrowly prefers action 1, below the confidence gate.
            "teacher_logits": torch.tensor([[0.0, 0.01]], dtype=torch.float32),
        },
        top_k=2,
        rank_loss_weight=1.0,
        gate_loss_weight=0.0,
        pair_loss_weight=0.0,
        anchor_loss_weight=0.0,
        residual_loss_weight=0.0,
        minimum_correction_margin=0.05,
    )
    loss.backward()

    assert metrics["correction_fraction"] == pytest.approx(0.0)
    assert correction_logits.grad[0, 0].item() < 0.0
    assert correction_logits.grad[0, 1].item() > 0.0


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
    assert metrics["objective"] == "frozen_base_topk_correction_v2"
    assert metrics["parameter_count"] < sum(
        parameter.numel() for parameter in trained.base_model.parameters()
    )
    assert metrics["final_validation"]["examples"] > 0
    assert trained.select_action(observation, legal) in legal

    context_config = replace(config, contextual_value_features=True)
    context_cache = tmp_path / "context-cache"
    context_manifest = build_recorded_teacher_cache(
        [trajectory],
        output_dir=context_cache,
        config=context_config,
        shard_size=5,
    )
    assert context_manifest["paired_base_features"] is True
    context_validation_cache = tmp_path / "context-validation-cache"
    build_recorded_teacher_cache(
        [trajectory],
        output_dir=context_validation_cache,
        config=context_config,
        shard_size=5,
    )
    context_checkpoint = tmp_path / "context.pt"
    StructuredPolicy(
        StructuredPolicyNetwork(context_config),
        config=context_config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=5,
    ).save(context_checkpoint)
    contextual_output = tmp_path / "trained-contextual-correction.pt"
    contextual, contextual_metrics = train_structured_correction(
        context_cache,
        base_checkpoint=base_path,
        validation_cache_dir=context_validation_cache,
        context_checkpoint=context_checkpoint,
        output_checkpoint=contextual_output,
        correction_config=CorrectionModelConfig(
            hidden_dim=24,
            dropout=0,
            top_k=2,
            contextual_value_features=True,
        ),
        epochs=0,
        batch_size=4,
        validation_fraction=0.2,
        device="cpu",
        seed=6,
        show_progress=False,
    )
    assert contextual_output.is_file()
    assert contextual_metrics["objective"] == (
        "frozen_base_contextual_topk_correction_v1"
    )
    assert contextual_metrics["hyperparameters"]["external_validation_cache"] is True
    assert contextual_metrics["final_validation"]["examples"] == len(decisions)
    assert contextual.context_model is not None
    assert contextual.select_action(observation, legal) in legal
