from __future__ import annotations

import json
import struct
import zlib

from gtn_ai.replay_audit import REPLAY_DOWNLOAD_MAGIC, audit_replay


def test_old_style_replay_is_statistics_only(tmp_path):
    header = {
        "format": "gtn-replay",
        "format_version": 1,
        "encoding": "zlib-json",
        "replay_id": 42,
        "mode": "1v1",
        "players": ["A", "B"],
    }
    replay = {
        "version": 2,
        "meta": {"mode": "1v1", "winner_index": 0},
        "rules": {"game_version": "test"},
        "keyframes": [{"state": {"phase": "playing"}}],
        "actions": [{"type": "play_card", "actor": 0, "payload": {}}],
    }
    header_raw = json.dumps(header).encode("utf-8")
    payload = (
        REPLAY_DOWNLOAD_MAGIC
        + struct.pack(">I", len(header_raw))
        + header_raw
        + zlib.compress(json.dumps(replay).encode("utf-8"))
    )
    path = tmp_path / "old.gtnreplay"
    path.write_bytes(payload)
    report = audit_replay(path)
    assert not report["strict_behavior_cloning_ready"]
    assert report["usable_for"] == ["outcome_and_action_statistics"]
    assert report["decision_actions"] == 1
