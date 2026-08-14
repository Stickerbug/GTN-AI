from __future__ import annotations

import json
import struct
import zlib

from gtn_ai.player_data import extract_strict_player_decisions, import_player_replays
from gtn_ai.protocol import Action
from gtn_ai.replay_audit import REPLAY_DOWNLOAD_MAGIC


def _write_replay(path, ai_decision):
    header = {
        "format": "gtn-replay",
        "format_version": 1,
        "encoding": "zlib-json",
        "replay_id": 17,
        "mode": "1v1",
        "players": ["must-not", "leak"],
    }
    replay = {
        "meta": {"mode": "1v1", "winner_index": 0},
        "actions": [{
            "type": "play_card",
            "actor": 0,
            "ai_decision": ai_decision,
        }],
    }
    header_raw = json.dumps(header).encode("utf-8")
    path.write_bytes(
        REPLAY_DOWNLOAD_MAGIC
        + struct.pack(">I", len(header_raw))
        + header_raw
        + zlib.compress(json.dumps(replay).encode("utf-8"))
    )


def _snapshot():
    selected = Action("end_turn")
    return {
        "observation": {
            "schema_version": 3,
            "seat": 0,
            "decision_player": 0,
            "phase": "playing",
            "loadout": {"ruleset_fingerprint": "rules-v1"},
            "self": {"health": 30, "hand": []},
            "opponent": {"health": 20, "hand_count": 2},
        },
        "legal_actions": [selected.to_dict(), Action("concede").to_dict()],
        "selected_action": selected.to_dict(),
        "actor_group_hash": "playerhash_0123456789abcdef",
    }


def test_strict_player_data_accepts_complete_anonymous_snapshot(tmp_path):
    replay = tmp_path / "strict.gtnreplay"
    _write_replay(replay, _snapshot())
    report = extract_strict_player_decisions(replay)
    assert report["strict_ready"]
    assert report["accepted_samples"] == 1
    sample = report["samples"][0]
    assert sample["player_group_quality"] == "cross_replay"
    assert sample["outcome"] == 1.0
    assert "must-not" not in json.dumps(sample)


def test_strict_player_data_rejects_illegal_or_sensitive_snapshot(tmp_path):
    replay = tmp_path / "bad.gtnreplay"
    snapshot = _snapshot()
    snapshot["observation"]["player_name"] = "secret"
    snapshot["selected_action"] = Action("play_card", {"slot": 99}).to_dict()
    _write_replay(replay, snapshot)
    report = extract_strict_player_decisions(replay)
    assert not report["strict_ready"]
    assert report["accepted_samples"] == 0
    assert report["rejection_counts"] == {"sensitive_observation_fields": 1}


def test_import_player_replays_writes_anonymous_jsonl(tmp_path):
    replay = tmp_path / "strict.gtnreplay"
    output = tmp_path / "players.jsonl"
    _write_replay(replay, _snapshot())
    report = import_player_replays([replay], output=output)
    assert report["samples"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["selected_action"]["kind"] == "end_turn"
    assert "players" not in row
