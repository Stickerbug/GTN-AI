from __future__ import annotations

import json
import sqlite3
import zlib
from copy import deepcopy

from gtn_ai.experience_prior import ExperiencePriorConfig, build_experience_prior
from gtn_ai.historical_aggregate import aggregate_replay_database
from gtn_ai.protocol import Action


def _database(tmp_path):
    path = tmp_path / "gtn.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE matches (
            id INTEGER PRIMARY KEY,
            mode TEXT,
            result TEXT,
            summary_json TEXT
        );
        CREATE TABLE match_replays (
            id INTEGER PRIMARY KEY,
            match_id INTEGER,
            created_at TEXT,
            mode TEXT,
            winner_index INTEGER,
            round_num INTEGER,
            replay_size INTEGER,
            replay_blob BLOB,
            replay_blob_path TEXT,
            mod_source TEXT,
            community_mod_name TEXT
        );
        """
    )
    summary = {
        "mode": "1v1",
        "result": "win",
        "winner_index": 0,
        "rounds": 6,
        "valid_for_ranking": True,
        "valid_action_counts_by_side": [20, 18],
        "player_ids": [101, 202],
        "players": ["private-a", "private-b"],
        "gr_result": {
            "repeat_factor": 1.0,
            "before": {
                "101": {"total_gr": 1000.0},
                "202": {"total_gr": 1000.0},
            },
        },
        "community_mods": [],
        "entertainment_mods": [],
        "draft_card_ids_by_player": [
            ["vanilla:thorn", "vanilla:bloom"],
            ["vanilla:root", "vanilla:guard"],
        ],
    }
    initial_state = {
        "phase": "playing",
        "perspectives": [{
            "v2_load_order": ["vanilla"],
            "spectate_players": [
                {
                    "health": 30,
                    "max_health": 30,
                    "elixir": 3,
                    "magic": 0,
                    "hand": [{
                        "instance_id": "card-1",
                        "def_id": "vanilla:thorn",
                    }],
                },
                {
                    "health": 24,
                    "max_health": 30,
                    "elixir": 2,
                    "magic": 1,
                    "hand": [],
                },
            ],
        }],
    }
    after_play = deepcopy(initial_state)
    after_play["perspectives"][0]["spectate_players"][0]["hand"] = []
    replay = {
        "meta": {"mode": "1v1", "winner_index": 0},
        "keyframes": [{"seq": 0, "t": 0, "state": initial_state}],
        "actions": [
            {
                "seq": 1,
                "type": "play_card",
                "actor": 0,
                "round": 4,
                "payload": {
                    "def_id": "vanilla:thorn",
                    "card_instance_id": "card-1",
                    "target_player_id": 1,
                },
                "state": after_play,
            },
            {
                "seq": 2,
                "type": "end_turn",
                "actor": 1,
                "round": 4,
                "payload": {},
                "state": after_play,
            },
        ],
    }
    blob = zlib.compress(json.dumps(replay).encode("utf-8"))
    connection.execute(
        "INSERT INTO matches VALUES (1, '1v1', 'win', ?)",
        (json.dumps(summary),),
    )
    connection.execute(
        "INSERT INTO match_replays VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            1,
            "2099-01-01T00:00:00Z",
            "1v1",
            0,
            6,
            len(blob),
            blob,
            "",
            "official",
            "",
        ),
    )
    connection.commit()
    connection.close()
    return path


def test_server_aggregate_is_identity_free_and_builds_prior(tmp_path):
    aggregate = aggregate_replay_database(
        _database(tmp_path),
        since_days=36500,
        minimum_actions_per_player=1,
    )
    rendered = json.dumps(aggregate)
    assert aggregate["quality"]["accepted_replays"] == 1
    assert aggregate["quality"]["accepted_actions"] == 2
    assert aggregate["quality"]["context_decisions"] == 2
    assert aggregate["global"]["context_raw_selections"] == 10
    assert aggregate["quality"]["deck_prior_replays"] == 1
    assert aggregate["deck_statistics"]["*"]["raw_decks"] == 2
    assert aggregate["deck_statistics"]["[\"vanilla\"]"]["raw_decks"] == 2
    assert "private-a" not in rendered
    assert "101" not in rendered

    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    prior, report = build_experience_prior(
        historical_aggregates=[aggregate_path],
        config=ExperiencePriorConfig(min_historical_uses=0),
    )
    observation = {
        "phase": "playing",
        "round": 4,
        "seat": 0,
        "loadout": {},
        "self": {"hand": [{"slot": 0, "def_id": "vanilla:thorn"}]},
    }
    action = Action("play_card", {"hand_slot": 0, "choice": {"target_player_id": 1}})
    assert report["retained_entries"] > 0
    assert prior.action_bonus(observation, action) > 0
