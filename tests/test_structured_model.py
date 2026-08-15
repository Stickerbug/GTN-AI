from __future__ import annotations

import json
from dataclasses import replace

import pytest

from gtn_ai import Garden1v1Env
from gtn_ai.deck_prior import DeckPrior
from gtn_ai.neural_model import (
    NeuralModelConfig,
    NeuralPolicy,
    VariableActionNetwork,
    torch,
    torch_available,
)
from gtn_ai.policies import policy_from_name
from gtn_ai.structured_cache import (
    build_distillation_cache,
    build_recorded_teacher_cache,
    iter_cached_examples,
)
from gtn_ai.structured_distillation import (
    _teacher_margin_weights,
    train_structured_distillation,
)
from gtn_ai.structured_features import TokenType, encode_structured_decision
from gtn_ai.structured_model import (
    StructuredEnsemblePolicy,
    StructuredModelConfig,
    StructuredPolicy,
    StructuredPolicyNetwork,
    collate_structured_decisions,
    expand_structured_model,
)


pytestmark = pytest.mark.skipif(
    not torch_available(), reason="PyTorch training extra is not installed"
)


def _structured_config() -> StructuredModelConfig:
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


def _teacher_config() -> NeuralModelConfig:
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


def test_teacher_margin_weights_favor_decisive_labels() -> None:
    logits = torch.tensor([[0.0, -0.1, -2.0], [0.0, -2.0, -3.0]])
    mask = torch.ones_like(logits, dtype=torch.bool)

    weights = _teacher_margin_weights(
        logits,
        mask,
        power=1.0,
        floor=0.25,
        reference=1.0,
    )

    assert weights[0].item() == pytest.approx(0.325)
    assert weights[1].item() == pytest.approx(1.0)


def test_structured_encoder_keeps_cards_players_and_actions_as_separate_tokens():
    config = _structured_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1101)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    encoded = encode_structured_decision(
        observation, legal, config=config.feature_config
    )
    token_types = [token.token_type for token in encoded.state_tokens]
    assert token_types[0] == int(TokenType.CLS)
    assert token_types.count(int(TokenType.PLAYER)) == 2
    assert int(TokenType.EVENT) in token_types
    assert len(encoded.action_tokens) == len(legal)
    assert all(token.token_type == int(TokenType.ACTION) for token in encoded.action_tokens)


def test_structured_network_handles_different_legal_action_counts():
    import torch

    config = _structured_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1102)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    examples = [
        encode_structured_decision(observation, legal, config=config.feature_config),
        encode_structured_decision(observation, legal[:1], config=config.feature_config),
    ]
    batch = collate_structured_decisions(examples, config=config)
    scores, values = StructuredPolicyNetwork(config)(batch)
    assert scores.shape == (2, len(legal))
    assert values.shape == (2,)
    assert batch["action_mask"][1].sum().item() == 1
    assert torch.isfinite(scores[batch["action_mask"]]).all()
    assert torch.isfinite(values).all()


def test_structured_checkpoint_round_trip_selects_only_a_legal_action(tmp_path):
    config = _structured_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1103)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    policy = StructuredPolicy(
        StructuredPolicyNetwork(config),
        config=config,
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=3,
    )
    path = tmp_path / "structured.pt"
    policy.save(path, metadata={"purpose": "test"})
    loaded = StructuredPolicy.load(path, device="cpu", seed=3)
    assert loaded.select_action(observation, legal) in legal
    assert -1.0 <= loaded.estimate_value(observation, legal) <= 1.0
    logits, value = loaded.evaluate_actions(observation, legal)
    assert len(logits) == len(legal)
    assert -1.0 <= value <= 1.0
    batch_values = loaded.estimate_values([
        (observation, legal),
        (observation, legal[:1]),
    ])
    assert len(batch_values) == 2
    assert batch_values[0] == pytest.approx(value, abs=1e-6)
    ensemble = StructuredEnsemblePolicy([loaded, loaded.fork(seed=4)], seed=5)
    assert ensemble.select_action(observation, legal) in legal
    assert ensemble.estimate_value(observation, legal) == pytest.approx(value, abs=1e-6)
    prior_path = tmp_path / "deck-prior.json"
    DeckPrior({}).save(prior_path)
    belief_policy = policy_from_name(
        f"structured-belief-cpu:{path}|{prior_path}", seed=6
    )
    assert belief_policy.select_action(observation, legal) in legal
    assert belief_policy.diagnostics()["deck_belief_fingerprint"]


def test_structured_model_expansion_preserves_initial_outputs() -> None:
    import torch

    source_config = _structured_config()
    target_config = StructuredModelConfig(
        **{
            **source_config.__dict__,
            "state_layers": source_config.state_layers + 1,
            "action_layers": source_config.action_layers + 1,
        }
    )
    source = StructuredPolicyNetwork(source_config)
    source.eval()
    expanded, report = expand_structured_model(
        source,
        source_config=source_config,
        target_config=target_config,
    )
    expanded.eval()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1107)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    decision = encode_structured_decision(
        observation,
        legal,
        config=source_config.feature_config,
    )
    batch = collate_structured_decisions([decision], config=source_config)

    with torch.no_grad():
        source_scores, source_values = source(batch)
        expanded_scores, expanded_values = expanded(batch)

    assert torch.allclose(source_scores, expanded_scores, atol=1e-6, rtol=1e-6)
    assert torch.allclose(source_values, expanded_values, atol=1e-6, rtol=1e-6)
    assert report["identity_state_layers"] == 1
    assert report["identity_action_layers"] == 1


def test_structured_model_can_enable_context_features_from_old_weights() -> None:
    source_config = _structured_config()
    target_config = StructuredModelConfig(
        **{
            **source_config.__dict__,
            "contextual_value_features": True,
        }
    )
    source = StructuredPolicyNetwork(source_config)
    expanded, report = expand_structured_model(
        source,
        source_config=source_config,
        target_config=target_config,
    )

    assert expanded.config.contextual_value_features is True
    assert report["contextual_value_features_changed"] is True
    assert set(expanded.state_dict()) == set(source.state_dict())


def test_versioned_cache_and_distillation_round_trip(tmp_path):
    config = _structured_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1104)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    teacher = NeuralPolicy(
        VariableActionNetwork(_teacher_config()),
        config=_teacher_config(),
        ruleset_fingerprint=env.ruleset_fingerprint,
        device="cpu",
        seed=1,
    )
    teacher_path = tmp_path / "teacher.pt"
    teacher.save(teacher_path)
    trajectory_path = tmp_path / "trajectory.jsonl"
    trajectory_path.write_text(json.dumps({
        "schema_version": 4,
        "seed": 1104,
        "official_mods": ["Vanilla Cards.gtnmod"],
        "loadout_fingerprint": env.loadout_fingerprint,
        "ruleset_fingerprint": env.ruleset_fingerprint,
        "policies": ["teacher", "teacher"],
        "winner": 0,
        "terminated": True,
        "truncated": False,
        "steps": 1,
        "rounds": 0,
        "loop_recoveries": 0,
        "decisions": [{
            "step": 0,
            "player": env.decision_player(),
            "observation": observation,
            "legal_actions": [action.to_dict() for action in legal],
            "action": legal[0].to_dict(),
            "forced_fallback": False,
        }],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    manifest = build_distillation_cache(
        [trajectory_path],
        teacher_checkpoint=teacher_path,
        output_dir=cache_dir,
        config=config,
        teacher_batch_size=1,
        shard_size=1,
        device="cpu",
    )
    cached = list(iter_cached_examples(cache_dir))
    assert manifest["counters"]["examples"] == 1
    assert len(cached) == 1
    assert len(cached[0].teacher_logits) == len(legal)
    output = tmp_path / "student.pt"
    student, metrics = train_structured_distillation(
        cache_dir,
        output_checkpoint=output,
        epochs=1,
        batch_size=1,
        shuffle_buffer=1,
        validation_fraction=0,
        replay_cache_dir=cache_dir,
        replay_ratio=1,
        trainable_scope="combat-policy-head",
        device="cpu",
        seed=2,
    )
    assert output.is_file()
    assert metrics["examples"] == 1
    assert metrics["replay_cache"]["examples_per_epoch"] == 1
    assert metrics["hyperparameters"]["replay_ratio"] == pytest.approx(1.0)
    assert metrics["trainable_parameter_count"] < metrics["parameter_count"]
    assert student.select_action(observation, legal) in legal


def test_recorded_search_teacher_cache_round_trip(tmp_path):
    config = _structured_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1105)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    decision_player = env.decision_player()
    selected_index = min(1, len(legal) - 1)
    executed_index = 0
    logits = [-2.0] * len(legal)
    logits[selected_index] = 0.0
    trajectory_path = tmp_path / "search.jsonl"
    trajectory_path.write_text(json.dumps({
        "schema_version": 4,
        "seed": 1105,
        "official_mods": ["Vanilla Cards.gtnmod"],
        "loadout_fingerprint": env.loadout_fingerprint,
        "ruleset_fingerprint": env.ruleset_fingerprint,
        "policies": ["search", "heuristic"],
        "winner": decision_player,
        "terminated": True,
        "truncated": False,
        "steps": 1,
        "rounds": 0,
        "loop_recoveries": 0,
        "decisions": [{
            "step": 0,
            "player": decision_player,
            "observation": observation,
            "legal_actions": [action.to_dict() for action in legal],
            "action": legal[executed_index].to_dict(),
            "forced_fallback": False,
            "diagnostic": {"hard_example_weight": 2.0},
            "teacher": {
                "kind": "belief_rollout_v1",
                "action_key": legal[selected_index].key,
                "logits": logits,
                "value": 0.25,
                "search_margin": 0.1,
            },
        }],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    cache_dir = tmp_path / "search-cache"
    prior_path = tmp_path / "deck-prior.json"
    prior = DeckPrior({})
    prior.save(prior_path)
    baseline_tokens = len(encode_structured_decision(
        observation, legal, config=config.feature_config
    ).state_tokens)

    manifest = build_recorded_teacher_cache(
        [trajectory_path],
        output_dir=cache_dir,
        config=config,
        deck_prior_path=prior_path,
        dynamic_deck_belief=True,
        shard_size=1,
    )
    examples = list(iter_cached_examples(cache_dir))

    assert manifest["source_kind"] == "recorded_offline_teacher"
    assert manifest["teacher_name"] == "belief_rollout_v1"
    assert manifest["deck_prior_fingerprint"] == prior.fingerprint
    assert manifest["deck_belief_schema_version"] == 2
    assert manifest["counters"]["examples"] == 1
    assert len(examples) == 1
    assert examples[0].teacher_logits == pytest.approx(logits)
    assert examples[0].teacher_value == pytest.approx(0.25)
    assert examples[0].example_weight == pytest.approx(2.0)
    assert examples[0].decision.selected_index == selected_index
    assert len(examples[0].decision.state_tokens) > baseline_tokens
    assert manifest["counters"]["teacher_behavior_disagreements"] == int(
        selected_index != executed_index
    )
    assert manifest["counters"]["teacher_behavior_agreements"] == int(
        selected_index == executed_index
    )

    contextual_cache = tmp_path / "contextual-search-cache"
    contextual_manifest = build_recorded_teacher_cache(
        [trajectory_path],
        output_dir=contextual_cache,
        config=replace(config, contextual_value_features=True),
        shard_size=1,
    )
    contextual_example = next(iter_cached_examples(contextual_cache))
    assert contextual_manifest["paired_base_features"] is True
    assert contextual_example.base_decision is not None
    assert contextual_example.example_weight == pytest.approx(2.0)
    assert contextual_example.base_decision.action_count == len(legal)
    assert contextual_example.decision.state_tokens != (
        contextual_example.base_decision.state_tokens
    )

    disagreement_cache = tmp_path / "teacher-disagreement-cache"
    disagreement_manifest = build_recorded_teacher_cache(
        [trajectory_path],
        output_dir=disagreement_cache,
        config=config,
        shard_size=1,
        only_teacher_disagreements=True,
    )
    assert disagreement_manifest["only_teacher_disagreements"] is True
    assert disagreement_manifest["counters"]["examples"] == 1
    assert disagreement_manifest["counters"]["decisions_filtered_teacher_agreement"] == 0

    agreeing_trajectory = tmp_path / "search-agreement.jsonl"
    agreeing_episode = json.loads(trajectory_path.read_text(encoding="utf-8"))
    agreeing_decision = agreeing_episode["decisions"][0]
    agreeing_logits = [-2.0] * len(legal)
    agreeing_logits[executed_index] = 0.0
    agreeing_decision["teacher"]["action_key"] = legal[executed_index].key
    agreeing_decision["teacher"]["logits"] = agreeing_logits
    agreeing_trajectory.write_text(
        json.dumps(agreeing_episode, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no recorded teacher decisions"):
        build_recorded_teacher_cache(
            [agreeing_trajectory],
            output_dir=tmp_path / "teacher-agreement-cache",
            config=config,
            shard_size=1,
            only_teacher_disagreements=True,
        )

    with pytest.raises(ValueError, match="no recorded teacher decisions"):
        build_recorded_teacher_cache(
            [trajectory_path],
            output_dir=tmp_path / "filtered-search-cache",
            config=config,
            shard_size=1,
            min_teacher_margin=0.2,
        )

    preserved_cache = tmp_path / "winner-preserved-cache"
    preserved = build_recorded_teacher_cache(
        [trajectory_path],
        output_dir=preserved_cache,
        config=config,
        shard_size=1,
        preserve_winner_actions=True,
    )
    preserved_example = next(iter_cached_examples(preserved_cache))
    assert preserved_example.decision.selected_index == executed_index
    assert max(
        range(len(preserved_example.teacher_logits)),
        key=preserved_example.teacher_logits.__getitem__,
    ) == executed_index
    assert preserved["counters"]["winner_actions_preserved"] == 1
    assert preserved["counters"]["winner_teacher_disagreements"] == int(
        selected_index != executed_index
    )

    losing_trajectory = tmp_path / "search-loss.jsonl"
    losing_episode = json.loads(trajectory_path.read_text(encoding="utf-8"))
    losing_episode["winner"] = 1 - decision_player
    losing_trajectory.write_text(
        json.dumps(losing_episode, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    corrected_cache = tmp_path / "loser-corrected-cache"
    corrected = build_recorded_teacher_cache(
        [losing_trajectory],
        output_dir=corrected_cache,
        config=config,
        shard_size=1,
        preserve_winner_actions=True,
    )
    corrected_example = next(iter_cached_examples(corrected_cache))
    assert corrected_example.decision.selected_index == selected_index
    assert corrected_example.teacher_logits == pytest.approx(logits)
    assert corrected["counters"]["loser_teacher_actions"] == 1

    shard_path = next(cache_dir.glob("shard-*.pt"))
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    shard.pop("example_weights")
    torch.save(shard, shard_path)
    legacy_example = next(iter_cached_examples(cache_dir))
    assert legacy_example.example_weight == pytest.approx(1.0)


def test_cache_split_is_stable_when_shards_are_shuffled(tmp_path):
    config = _structured_config()
    env = Garden1v1Env(enabled_mods=["Vanilla Cards.gtnmod"], seed=1106)
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    decisions = []
    for index in range(24):
        selected = index % len(legal)
        logits = [-1.0] * len(legal)
        logits[selected] = float(index + 1)
        decisions.append({
            "step": index,
            "player": env.decision_player(),
            "observation": observation,
            "legal_actions": [action.to_dict() for action in legal],
            "action": legal[selected].to_dict(),
            "forced_fallback": False,
            "teacher": {
                "kind": "split_test",
                "action_key": legal[selected].key,
                "logits": logits,
                "value": 0.0,
                "search_margin": 0.2,
            },
        })
    source = tmp_path / "split.jsonl"
    source.write_text(json.dumps({
        "schema_version": 4,
        "seed": 1106,
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
    cache = tmp_path / "stable-split-cache"
    build_recorded_teacher_cache(
        [source], output_dir=cache, config=config, shard_size=4
    )

    def signatures(seed):
        return {
            example.teacher_logits
            for example in iter_cached_examples(
                cache,
                split="validation",
                validation_fraction=0.25,
                shuffle_shards=True,
                seed=seed,
            )
        }

    assert signatures(1) == signatures(2)
