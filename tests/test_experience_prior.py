from __future__ import annotations

import json

from gtn_ai.experience_prior import (
    ExperiencePrior,
    ExperiencePriorConfig,
    ExperiencePriorPolicy,
    build_experience_prior,
)
from gtn_ai.protocol import Action


def _observation():
    return {
        "schema_version": 3,
        "phase": "playing",
        "round": 3,
        "seat": 0,
        "loadout": {"ruleset_fingerprint": "rules-v1"},
        "self": {
            "hand": [{"slot": 0, "def_id": "vanilla:thorn"}],
        },
        "opponent": {"health": 20},
    }


def test_strict_exposure_prior_prefers_empirically_selected_action(tmp_path):
    path = tmp_path / "strict.jsonl"
    play = Action("play_card", {"hand_slot": 0, "choice": {"target_player_id": 1}})
    end = Action("end_turn")
    rows = []
    for index in range(20):
        selected = play if index < 16 else end
        rows.append({
            "schema_version": 1,
            "ruleset_fingerprint": "rules-v1",
            "outcome": 1.0 if selected == play else -1.0,
            "observation": _observation(),
            "legal_actions": [play.to_dict(), end.to_dict()],
            "selected_action": selected.to_dict(),
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    prior, report = build_experience_prior(
        strict_datasets=[path],
        config=ExperiencePriorConfig(min_strict_exposures=1),
    )
    assert report["retained_entries"] > 0
    assert prior.action_bonus(_observation(), play) > prior.action_bonus(_observation(), end)

    output = tmp_path / "prior.json"
    prior.save(output)
    loaded = ExperiencePrior.load(output)
    assert loaded.fingerprint == prior.fingerprint
    assert loaded.action_bonus(_observation(), play) == prior.action_bonus(_observation(), play)


def test_experience_prior_policy_only_reranks_base_logits(tmp_path):
    play = Action("play_card", {"hand_slot": 0, "choice": {"target_player_id": 1}})
    end = Action("end_turn")
    path = tmp_path / "strict.jsonl"
    rows = [{
        "schema_version": 1,
        "ruleset_fingerprint": "rules-v1",
        "outcome": 1.0,
        "observation": _observation(),
        "legal_actions": [play.to_dict(), end.to_dict()],
        "selected_action": play.to_dict(),
    } for _ in range(20)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    prior, _ = build_experience_prior(
        strict_datasets=[path],
        config=ExperiencePriorConfig(min_strict_exposures=1),
    )

    class Base:
        name = "base"
        ruleset_fingerprint = "rules-v1"
        model_fingerprint = "base-fingerprint"

        def evaluate_actions(self, observation, actions):
            return [0.0 for _ in actions], 0.25

        def estimate_value(self, observation, actions):
            return 0.25

        def estimate_values(self, decisions):
            return [0.25 for _ in decisions]

        def fork(self, **kwargs):
            return self

    policy = ExperiencePriorPolicy(Base(), prior, seed=3)
    logits, value = policy.evaluate_actions(_observation(), [play, end])
    assert logits[0] > logits[1]
    assert value == 0.25
    assert policy.select_action(_observation(), [play, end]) == play
    assert policy.diagnostics()["prior_adjusted_decisions"] == 2
