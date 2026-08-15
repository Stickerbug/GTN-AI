from __future__ import annotations

import gzip
import json

from gtn_ai.environment import Garden1v1Env
from gtn_ai.hard_examples import curate_hard_examples


class _BasePolicy:
    def evaluate_actions(self, observation, legal_actions):
        return [0.0] + [-1.0] * (len(legal_actions) - 1), 0.0


def _decision(observation, legal, *, margin=0.5, marker=False):
    logits = [-2.0] * len(legal)
    logits[-1] = 0.0
    if len(logits) > 1:
        logits[-2] = -float(margin)
    return {
        "step": 0,
        "player": observation["decision_player"],
        "observation": observation,
        "legal_actions": [action.to_dict() for action in legal],
        "action": legal[0].to_dict(),
        "forced_fallback": False,
        "teacher": {
            "kind": "test_teacher",
            "action_key": legal[-1].key,
            "logits": logits,
            "value": 0.0,
            "search_margin": margin,
        },
        "diagnostic": {
            "marked": marker,
            "marker_window": marker,
            "marker_distance": 1 if marker else None,
        },
    }


def _episode(env, observation, decision, *, session, mods):
    return {
        "schema_version": 4,
        "seed": 1,
        "official_mods": list(mods),
        "loadout_fingerprint": env.loadout_fingerprint,
        "ruleset_fingerprint": env.ruleset_fingerprint,
        "policies": ["human", "ai"],
        "winner": 0,
        "terminated": True,
        "truncated": False,
        "steps": 1,
        "rounds": 1,
        "loop_recoveries": 0,
        "decisions": [decision],
        "diagnostic_session": session,
    }


def _write(path, episodes):
    path.write_text(
        "".join(
            json.dumps(episode, ensure_ascii=False, separators=(",", ":")) + "\n"
            for episode in episodes
        ),
        encoding="utf-8",
    )


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_curate_hard_examples_splits_whole_sessions_by_mod_stratum(tmp_path):
    env = Garden1v1Env(
        seed=81,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    episodes = []
    for stratum in ("Vanilla", "Garden"):
        for index in range(2):
            episodes.append(_episode(
                env,
                observation,
                _decision(observation, legal),
                session=f"{stratum}-{index}",
                mods=[stratum],
            ))
    source = tmp_path / "labels.jsonl"
    _write(source, episodes)
    train = tmp_path / "train.jsonl.gz"
    validation = tmp_path / "validation.jsonl.gz"

    report = curate_hard_examples(
        [source],
        train_output=train,
        validation_output=validation,
        base_policy=_BasePolicy(),
        base_policy_name="test-base",
        validation_fraction=0.5,
        anchor_ratio=0.0,
        seed=7,
    )

    assert report["train_sessions"] == 2
    assert report["validation_sessions"] == 2
    train_episodes = _read(train)
    validation_episodes = _read(validation)
    train_ids = {episode["diagnostic_session"] for episode in train_episodes}
    validation_ids = {episode["diagnostic_session"] for episode in validation_episodes}
    assert train_ids.isdisjoint(validation_ids)
    assert {tuple(episode["official_mods"]) for episode in train_episodes} == {
        ("Vanilla",),
        ("Garden",),
    }
    assert {tuple(episode["official_mods"]) for episode in validation_episodes} == {
        ("Vanilla",),
        ("Garden",),
    }
    assert all(
        episode["decisions"][0]["diagnostic"]["hard_example_role"]
        == "correction_or_review"
        for episode in (*train_episodes, *validation_episodes)
    )


def test_marker_window_survives_low_teacher_margin_as_review_sample(tmp_path):
    env = Garden1v1Env(
        seed=82,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    source = tmp_path / "marked.jsonl"
    _write(source, [_episode(
        env,
        observation,
        _decision(observation, legal, margin=0.01, marker=True),
        session="marked-session",
        mods=["Vanilla"],
    )])
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"

    report = curate_hard_examples(
        [source],
        train_output=train,
        validation_output=validation,
        base_policy=_BasePolicy(),
        base_policy_name="test-base",
        minimum_teacher_margin=0.05,
        validation_fraction=0.0,
        anchor_ratio=0.0,
    )

    assert report["train_examples"] == 1
    diagnostic = _read(train)[0]["decisions"][0]["diagnostic"]
    assert diagnostic["hard_example_reasons"] == ["marker_window"]
    assert diagnostic["teacher_margin"] == 0.01
