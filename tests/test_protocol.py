from __future__ import annotations

import pytest

from gtn_ai.protocol import ACTION_SCHEMA_VERSION, Action


def test_action_round_trip_and_stable_key():
    action = Action("play_card", {"choice": {"target_player_id": 1}, "hand_slot": 7})
    assert Action.from_dict(action.to_dict()) == action
    assert action.key == Action("play_card", {"hand_slot": 7, "choice": {"target_player_id": 1}}).key


def test_action_rejects_unknown_schema():
    with pytest.raises(ValueError, match="unsupported action schema"):
        Action.from_dict({"schema_version": ACTION_SCHEMA_VERSION + 1, "kind": "end_turn"})
