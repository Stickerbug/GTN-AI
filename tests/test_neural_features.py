from __future__ import annotations

from gtn_ai.features import (
    hashed_action_only_features,
    hashed_observation_features,
    history_event_token_ids,
)
from gtn_ai.protocol import Action


def _observation():
    return {
        "phase": "action",
        "seat": 0,
        "first_player": 1,
        "round": 3,
        "loadout": {"official_mods": ["Vanilla Cards.gtnmod"]},
        "self": {
            "health": 50,
            "max_health": 60,
            "elixir": 4,
            "magic": 2,
            "hand": [{"slot": 0, "def_id": "Basic", "card_type": "thorn"}],
        },
        "opponent": {"health": 17, "max_health": 50},
        "public_history": [
            {"kind": "play_card", "player": 0, "target_player": 1, "card_def_id": "Basic", "round": 2},
            {"kind": "play_card", "player": 1, "target_player": 0, "card_def_id": "Bone", "round": 3},
        ],
    }


def test_neural_feature_encoders_are_deterministic_and_separate_state_from_action():
    observation = _observation()
    action = Action("play_card", {"hand_slot": 0, "choice": {"target_player_id": 1}})
    state = hashed_observation_features(observation, buckets=256)
    encoded_action = hashed_action_only_features(observation, action, buckets=128)
    assert state == hashed_observation_features(observation, buckets=256)
    assert encoded_action == hashed_action_only_features(observation, action, buckets=128)
    assert state
    assert encoded_action
    assert all(0 <= index < 256 for index in state)
    assert all(0 <= index < 128 for index in encoded_action)


def test_history_encoding_uses_relative_players_and_respects_limits():
    player_zero = _observation()
    player_one = {**player_zero, "seat": 1}
    zero_tokens = history_event_token_ids(player_zero, buckets=64, max_events=1)
    one_tokens = history_event_token_ids(player_one, buckets=64, max_events=1)
    assert len(zero_tokens) == 1
    assert len(one_tokens) == 1
    assert zero_tokens != one_tokens
    assert all(1 <= token <= 64 for token in zero_tokens[0])


def test_history_encoding_never_reads_private_opponent_cards():
    observation = _observation()
    baseline = history_event_token_ids(observation, buckets=64)
    observation["opponent"]["hand"] = [{"def_id": "SecretCard"}]
    assert history_event_token_ids(observation, buckets=64) == baseline
