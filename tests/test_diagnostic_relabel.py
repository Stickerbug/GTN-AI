import gzip
import json

import pytest

from gtn_ai.diagnostic_relabel import relabel_diagnostic_sessions
from gtn_ai.diagnostics import DiagnosticSessionRecorder
from gtn_ai.environment import Garden1v1Env


class _TeacherPolicy:
    name = "test-teacher"

    def __init__(self) -> None:
        self.last_teacher_metadata = None
        self.last_search_metadata = None

    def select_action_with_env(self, env, observation, legal_actions, player_id):
        selected = legal_actions[-1]
        logits = [-2.0] * len(legal_actions)
        logits[-1] = 0.0
        self.last_teacher_metadata = {
            "kind": "test_teacher_v1",
            "action_key": selected.key,
            "logits": logits,
            "value": 0.25,
            "search_margin": 0.5,
            "rollouts": 2,
        }
        self.last_search_metadata = {"seconds": 0.01}
        return legal_actions[0]


def _diagnostic_session(tmp_path, *, marked: bool = False):
    env = Garden1v1Env(
        seed=71,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    observation = env.reset()
    actor = env.decision_player()
    legal = env.legal_actions(actor)
    recorder = DiagnosticSessionRecorder(
        tmp_path,
        session_id="relabel-test",
        metadata={"ruleset_fingerprint": env.ruleset_fingerprint},
    )
    recorder.record_decision(
        actor=actor,
        actor_kind="human",
        observation=observation,
        legal_actions=legal,
        action=legal[0],
        engine=env.engine,
    )
    if marked:
        recorder.mark(0, label="review")
    recorder.finish(outcome={"winner": actor, "reason": "game_over"})
    return env, recorder, actor, legal


def _read_episodes(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_relabel_requires_explicit_trust_for_private_pickles(tmp_path) -> None:
    _, recorder, _, _ = _diagnostic_session(tmp_path / "sessions")

    with pytest.raises(ValueError, match="Pickle"):
        relabel_diagnostic_sessions(
            [recorder.path],
            output=tmp_path / "labels.jsonl.gz",
            policy=_TeacherPolicy(),
        )


def test_relabel_writes_teacher_trajectory_from_session_directory(tmp_path) -> None:
    env, recorder, actor, legal = _diagnostic_session(tmp_path / "sessions", marked=True)
    output = tmp_path / "labels.jsonl.gz"

    report = relabel_diagnostic_sessions(
        [recorder.path],
        output=output,
        game_root=env.game_root,
        policy_name="test-teacher",
        policy=_TeacherPolicy(),
        only_marked=True,
        trust_private_snapshots=True,
        show_progress=False,
    )

    assert report["episodes"] == 1
    assert report["decisions_labeled"] == 1
    episode = _read_episodes(output)[0]
    assert episode["terminated"] is True
    assert episode["winner"] == actor
    decision = episode["decisions"][0]
    assert decision["action"] == legal[0].to_dict()
    assert decision["teacher"]["action_key"] == legal[-1].key
    assert decision["diagnostic"]["marked"] is True
    assert decision["diagnostic"]["marker_window"] is True
    assert decision["diagnostic"]["marker_distance"] == 0
    assert decision["diagnostic"]["recorded_action_matches_teacher"] == (
        legal[0].key == legal[-1].key
    )


def test_relabel_marker_window_includes_delayed_previous_decisions(tmp_path) -> None:
    env = Garden1v1Env(
        seed=73,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    observation = env.reset()
    actor = env.decision_player()
    legal = env.legal_actions(actor)
    recorder = DiagnosticSessionRecorder(
        tmp_path / "sessions",
        session_id="delayed-marker-test",
        metadata={"ruleset_fingerprint": env.ruleset_fingerprint},
    )
    for _ in range(3):
        recorder.record_decision(
            actor=actor,
            actor_kind="ai",
            observation=observation,
            legal_actions=legal,
            action=legal[0],
            engine=env.engine,
        )
    recorder.mark(2, label="late-review")
    recorder.finish(outcome={"winner": actor, "reason": "game_over"})
    output = tmp_path / "window.jsonl"

    report = relabel_diagnostic_sessions(
        [recorder.path],
        output=output,
        game_root=env.game_root,
        policy_name="test-teacher",
        policy=_TeacherPolicy(),
        only_marked=True,
        marker_lookback=2,
        trust_private_snapshots=True,
        show_progress=False,
    )

    assert report["decisions_labeled"] == 3
    decisions = _read_episodes(output)[0]["decisions"]
    assert [item["diagnostic"]["marker_distance"] for item in decisions] == [2, 1, 0]
    assert all(item["diagnostic"]["marker_labels"] == ["late-review"] for item in decisions)


def test_relabel_candidate_mode_selects_search_changes_and_low_margins(tmp_path) -> None:
    env = Garden1v1Env(
        seed=79,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    observation = env.reset()
    actor = env.decision_player()
    legal = env.legal_actions(actor)
    recorder = DiagnosticSessionRecorder(
        tmp_path / "sessions",
        session_id="automatic-candidate-test",
        metadata={"ruleset_fingerprint": env.ruleset_fingerprint},
    )
    recorder.record_decision(
        actor=actor,
        actor_kind="ai",
        observation=observation,
        legal_actions=legal,
        action=legal[0],
        policy_metadata={"changed": True, "score_margin": 0.4},
        engine=env.engine,
    )
    recorder.record_decision(
        actor=actor,
        actor_kind="ai",
        observation=observation,
        legal_actions=legal,
        action=legal[0],
        policy_metadata={"changed": False, "score_margin": 0.01},
        engine=env.engine,
    )
    recorder.record_decision(
        actor=actor,
        actor_kind="human",
        observation=observation,
        legal_actions=legal,
        action=legal[0],
        engine=env.engine,
    )
    recorder.finish(outcome={"winner": actor, "reason": "game_over"})
    output = tmp_path / "candidates.jsonl"

    report = relabel_diagnostic_sessions(
        [recorder.path],
        output=output,
        game_root=env.game_root,
        policy_name="test-teacher",
        policy=_TeacherPolicy(),
        only_candidates=True,
        anchor_rate=0,
        uncertainty_margin=0.05,
        trust_private_snapshots=True,
        show_progress=False,
    )

    assert report["decisions_labeled"] == 2
    assert report["rejection_counts"] == {"non_candidate_decision": 1}
    decisions = _read_episodes(output)[0]["decisions"]
    assert decisions[0]["diagnostic"]["candidate_reasons"] == [
        "live_search_changed_action",
    ]
    assert decisions[1]["diagnostic"]["candidate_reasons"] == [
        "low_live_search_margin",
    ]


def test_relabel_candidate_mode_always_keeps_marker_windows(tmp_path) -> None:
    env, recorder, _, _ = _diagnostic_session(tmp_path / "sessions", marked=True)
    output = tmp_path / "marked-candidate.jsonl"

    report = relabel_diagnostic_sessions(
        [recorder.path],
        output=output,
        game_root=env.game_root,
        policy_name="test-teacher",
        policy=_TeacherPolicy(),
        only_candidates=True,
        anchor_rate=0,
        trust_private_snapshots=True,
        show_progress=False,
    )

    assert report["decisions_labeled"] == 1
    decision = _read_episodes(output)[0]["decisions"][0]
    assert decision["diagnostic"]["candidate_reasons"] == ["marker_window"]


def test_relabel_reads_export_zip_without_extracting_it(tmp_path) -> None:
    env, recorder, _, _ = _diagnostic_session(tmp_path / "sessions")
    exported = recorder.export(tmp_path / "session.gtnai.zip")
    output = tmp_path / "zip-labels.jsonl"

    report = relabel_diagnostic_sessions(
        [exported],
        output=output,
        game_root=env.game_root,
        policy_name="test-teacher",
        policy=_TeacherPolicy(),
        trust_private_snapshots=True,
        show_progress=False,
    )

    assert report["decisions_labeled"] == 1
    assert len(_read_episodes(output)) == 1


def test_relabel_deduplicates_session_directory_and_export(tmp_path) -> None:
    env, recorder, _, _ = _diagnostic_session(tmp_path / "sessions")
    recorder.export()
    output = tmp_path / "deduplicated.jsonl"

    report = relabel_diagnostic_sessions(
        [tmp_path / "sessions"],
        output=output,
        game_root=env.game_root,
        policy_name="test-teacher",
        policy=_TeacherPolicy(),
        trust_private_snapshots=True,
        show_progress=False,
    )

    assert report["decisions_labeled"] == 1
    assert report["rejection_counts"] == {"duplicate_session": 1}
    assert len(_read_episodes(output)) == 1


def test_relabel_rejects_changed_legal_action_protocol(tmp_path) -> None:
    env, recorder, _, _ = _diagnostic_session(tmp_path / "sessions")
    decisions_path = recorder.decisions_path
    row = json.loads(decisions_path.read_text(encoding="utf-8"))
    row["legal_actions"] = row["legal_actions"][:-1]
    decisions_path.write_text(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "drifted.jsonl.gz"

    report = relabel_diagnostic_sessions(
        [recorder.path],
        output=output,
        game_root=env.game_root,
        policy_name="test-teacher",
        policy=_TeacherPolicy(),
        trust_private_snapshots=True,
        show_progress=False,
    )

    assert report["decisions_labeled"] == 0
    assert report["rejection_counts"] == {"legal_actions_changed": 1}
    assert _read_episodes(output) == []
