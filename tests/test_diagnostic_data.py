from __future__ import annotations

import gzip
import json

from gtn_ai.diagnostic_data import import_diagnostic_sessions
from gtn_ai.diagnostics import DiagnosticSessionRecorder
from gtn_ai.environment import Garden1v1Env
from gtn_ai.protocol import Action


def _finished_bundle(tmp_path):
    env = Garden1v1Env(
        seed=71,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    observation = env.observe(actor)
    legal = env.legal_actions(actor)
    selected = legal[0]
    recorder = DiagnosticSessionRecorder(tmp_path / "sessions", session_id="diagnostic-import-test")
    recorder.record_decision(
        actor=actor,
        actor_kind="human",
        observation=observation,
        legal_actions=legal,
        action=selected,
        engine=env.engine,
    )
    recorder.record_decision(
        actor=actor,
        actor_kind="ai",
        observation=observation,
        legal_actions=legal,
        action=selected,
        engine=env.engine,
    )
    marked = recorder.record_decision(
        actor=actor,
        actor_kind="human",
        observation=observation,
        legal_actions=legal,
        action=selected,
        engine=env.engine,
    )
    recorder.mark(marked.decision_id, label="review")
    recorder.record_decision(
        actor=actor,
        actor_kind="human",
        observation=observation,
        legal_actions=legal,
        action=Action("play_card", {"hand_slot": 999}),
        engine=env.engine,
    )
    recorder.finish(outcome={"winner": actor})
    return recorder, selected


def test_import_diagnostics_defaults_to_unmarked_human_legal_actions(tmp_path):
    recorder, selected = _finished_bundle(tmp_path)
    output = tmp_path / "human.jsonl.gz"

    report = import_diagnostic_sessions([recorder.path], output=output)

    assert report["samples"] == 1
    assert report["rejection_counts"] == {
        "actor_kind_filtered": 1,
        "marked_decision": 1,
        "selected_action_not_legal": 1,
    }
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows[0]["selected_action"] == selected.to_dict()
    assert rows[0]["outcome"] == 1.0
    assert rows[0]["actor_kind"] == "human"


def test_import_diagnostics_reads_export_and_can_include_ai_and_marked(tmp_path):
    recorder, _ = _finished_bundle(tmp_path)
    exported = recorder.export(tmp_path / "session.gtnai.zip")
    output = tmp_path / "all.jsonl"

    report = import_diagnostic_sessions(
        [exported],
        output=output,
        actor_kinds=("human", "ai"),
        include_marked=True,
    )

    assert report["samples"] == 3
    assert report["rejection_counts"] == {"selected_action_not_legal": 1}
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["actor_kind"] for row in rows] == ["human", "ai", "human"]
