import json
import zipfile

from gtn_ai.diagnostics import (
    DiagnosticSessionRecorder,
    cleanup_diagnostic_storage,
    read_jsonl,
)
from gtn_ai.environment import Garden1v1Env
from gtn_ai.protocol import Action


def test_environment_can_attach_to_live_engine_without_mutating_it() -> None:
    source = Garden1v1Env(seed=17, enabled_mods=["Vanilla Cards.gtnmod"])
    source.reset()
    attached = Garden1v1Env.from_engine(
        source.engine,
        seed=19,
        enabled_mods=source.mod_filenames,
    )

    assert attached.engine is source.engine
    assert attached.ruleset_fingerprint == source.ruleset_fingerprint
    assert attached.observe(0)["seat"] == 0

    detached = Garden1v1Env.from_engine(source.engine, seed=20, copy_engine=True)
    assert detached.engine is not source.engine
    assert detached.engine.get_public_state(0) == source.engine.get_public_state(0)


def test_diagnostic_session_records_marks_and_exports(tmp_path) -> None:
    env = Garden1v1Env(
        seed=23,
        enabled_mods=["Vanilla Cards.gtnmod"],
        include_pregame=False,
    )
    observation = env.reset()
    actor = env.decision_player()
    legal = env.legal_actions(actor)
    action = legal[0]
    recorder = DiagnosticSessionRecorder(
        tmp_path,
        session_id="test-session",
        metadata={"ruleset_fingerprint": env.ruleset_fingerprint},
    )

    record = recorder.record_decision(
        actor=actor,
        actor_kind="ai",
        observation=observation,
        legal_actions=legal,
        action=action,
        elapsed_ms=12.3456,
        policy_metadata={"selected_action_key": action.key},
        engine=env.engine,
    )
    marker = recorder.mark(label="bad-target", note="Human review")
    recorder.finish(outcome={"winner": 0})
    exported = recorder.export()

    assert record.decision_id == 0
    assert marker["decision_id"] == 0
    rows = list(read_jsonl(recorder.decisions_path))
    assert rows[0]["action"] == action.to_dict()
    assert rows[0]["elapsed_ms"] == 12.346
    assert (recorder.path / rows[0]["snapshot"]).is_file()
    manifest = json.loads((recorder.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "finished"
    assert manifest["decision_count"] == 1
    assert manifest["marker_count"] == 1
    with zipfile.ZipFile(exported) as archive:
        assert "manifest.json" in archive.namelist()
        assert "decisions.jsonl" in archive.namelist()
        assert rows[0]["snapshot"] in archive.namelist()


def test_diagnostic_session_can_omit_private_snapshots(tmp_path) -> None:
    recorder = DiagnosticSessionRecorder(
        tmp_path,
        session_id="public-only",
        save_private_snapshots=False,
    )
    record = recorder.record_decision(
        actor=0,
        actor_kind="human",
        observation={"phase": "action"},
        legal_actions=[Action("end_turn")],
        action=Action("end_turn"),
    )

    assert record.snapshot is None
    assert not recorder.snapshot_path.exists()


def test_diagnostic_cleanup_expires_old_data_but_preserves_active_sessions(tmp_path) -> None:
    old = DiagnosticSessionRecorder(tmp_path, session_id="old-session", save_private_snapshots=False)
    protected = DiagnosticSessionRecorder(
        tmp_path,
        session_id="protected-session",
        save_private_snapshots=False,
    )
    old.finish(outcome={"reason": "test"})
    protected.finish(outcome={"reason": "test"})
    old_export = old.export()
    protected_export = protected.export()
    for recorder in (old, protected):
        manifest_path = recorder.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["created_at"] = "2020-01-01T00:00:00+00:00"
        manifest["finished_at"] = "2020-01-01T00:01:00+00:00"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = cleanup_diagnostic_storage(
        tmp_path,
        retention_days=1,
        max_bytes=0,
        protected_session_ids={"protected-session"},
        now=1_800_000_000,
    )

    assert report["removed_entries"] == 2
    assert not old.path.exists()
    assert not old_export.exists()
    assert protected.path.exists()
    assert protected_export.exists()


def test_diagnostic_cleanup_enforces_oldest_first_byte_budget(tmp_path) -> None:
    first = DiagnosticSessionRecorder(tmp_path, session_id="first", save_private_snapshots=False)
    second = DiagnosticSessionRecorder(tmp_path, session_id="second", save_private_snapshots=False)
    (first.path / "payload.bin").write_bytes(b"a" * 200)
    (second.path / "payload.bin").write_bytes(b"b" * 200)
    first.finish()
    second.finish()
    first_manifest = json.loads((first.path / "manifest.json").read_text(encoding="utf-8"))
    first_manifest["finished_at"] = "2025-01-01T00:00:00+00:00"
    (first.path / "manifest.json").write_text(json.dumps(first_manifest), encoding="utf-8")

    report = cleanup_diagnostic_storage(tmp_path, retention_days=0, max_bytes=350)

    assert report["remaining_bytes"] <= 350
    assert not first.path.exists()
