from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence


HISTORICAL_AGGREGATE_SCHEMA_VERSION = 1
MAX_COMPRESSED_BYTES = 8 * 1024 * 1024
MAX_DECODED_BYTES = 64 * 1024 * 1024


def aggregate_replay_database(
    database: str | Path,
    *,
    blob_root: str | Path | None = None,
    since_days: int = 45,
    half_life_days: float = 14.0,
    minimum_rounds: int = 4,
    minimum_actions_per_player: int = 8,
    include_action_statistics: bool = True,
    until_days_ago: int = 0,
) -> dict[str, Any]:
    """Aggregate replay outcomes without exporting identity or raw state."""

    database_path = Path(database).resolve()
    root = Path(blob_root).resolve() if blob_root else database_path.parent / "replay-blobs"
    window_days = max(1, int(since_days))
    end_offset = max(0, int(until_days_ago))
    if end_offset >= window_days:
        raise ValueError("until_days_ago must be smaller than since_days")
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)
    connection = sqlite3.connect(
        f"file:{database_path.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    query = """
        SELECT r.id, r.created_at, r.mode, r.winner_index, r.round_num,
               r.replay_size, r.replay_blob, r.replay_blob_path,
               r.mod_source, r.community_mod_name,
               m.result, m.summary_json
        FROM match_replays AS r
        LEFT JOIN matches AS m ON m.id = r.match_id
        WHERE r.mode = '1v1' AND r.created_at >= ?
    """
    parameters = [cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")]
    if end_offset:
        query += " AND r.created_at < ?"
        upper = now - timedelta(days=end_offset)
        parameters.append(
            upper.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    query += " ORDER BY r.id ASC"
    rows = connection.execute(query, tuple(parameters))
    statistics: dict[str, Counter[str]] = defaultdict(Counter)
    deck_statistics: dict[str, dict[str, Any]] = {}
    global_stats: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    try:
        for row in rows:
            quality["rows"] += 1
            summary = _json_object(row["summary_json"])
            rejection = _metadata_rejection(
                row,
                summary,
                minimum_rounds=minimum_rounds,
                minimum_actions_per_player=minimum_actions_per_player,
            )
            if rejection:
                quality[f"rejected_{rejection}"] += 1
                continue
            try:
                replay = _load_replay(row, root)
            except (OSError, ValueError, zlib.error, json.JSONDecodeError):
                quality["rejected_decode"] += 1
                continue
            action_rejection = _action_stream_rejection(
                replay, minimum_rounds=minimum_rounds
            )
            if action_rejection:
                quality[f"rejected_{action_rejection}"] += 1
                continue
            winner = _winner_index(row, summary, replay)
            if winner not in (-1, 0, 1):
                quality["rejected_winner"] += 1
                continue
            recency_weight = _recency_weight(
                row["created_at"], half_life_days=half_life_days
            )
            repeat_weight = max(
                0.1,
                min(1.0, _as_float((summary.get("gr_result") or {}).get("repeat_factor"), 1.0)),
            )
            match_weight = recency_weight * repeat_weight
            expected = _expected_scores(summary)
            skill_weights = _skill_weights(summary)
            actions_used = 0
            for raw_action in replay.get("actions") or []:
                if not isinstance(raw_action, dict):
                    continue
                actor = _as_int(raw_action.get("actor"), -1)
                if actor not in (0, 1):
                    continue
                keys = historical_action_keys(raw_action, actor=actor)
                if not keys:
                    continue
                actual = 0.5 if winner < 0 else (1.0 if winner == actor else 0.0)
                adjusted_score = max(
                    0.0,
                    min(1.0, 0.5 + actual - expected[actor]),
                )
                if include_action_statistics:
                    for key in keys:
                        statistics[key]["historical_uses"] += match_weight
                        statistics[key]["historical_outcome_sum"] += (
                            adjusted_score * match_weight
                        )
                        statistics[key]["historical_raw_uses"] += 1
                        global_stats["historical_uses"] += match_weight
                        global_stats["historical_outcome_sum"] += (
                            adjusted_score * match_weight
                        )
                        global_stats["historical_raw_uses"] += 1
                actions_used += 1
            if actions_used:
                if _accumulate_drafted_decks(
                    replay,
                    summary=summary,
                    match_weight=match_weight,
                    skill_weights=skill_weights,
                    deck_statistics=deck_statistics,
                ):
                    quality["deck_prior_replays"] += 1
                else:
                    quality["deck_prior_missing"] += 1
                if include_action_statistics:
                    _accumulate_contextual_choices(
                        replay,
                        summary=summary,
                        winner=winner,
                        match_weight=match_weight,
                        expected=expected,
                        skill_weights=skill_weights,
                        statistics=statistics,
                        global_stats=global_stats,
                        quality=quality,
                    )
                quality["accepted_replays"] += 1
                quality["accepted_actions"] += actions_used
                global_stats["effective_replays"] += match_weight
            else:
                quality["rejected_no_supported_actions"] += 1
    finally:
        connection.close()

    return {
        "schema_version": HISTORICAL_AGGREGATE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "parameters": {
            "since_days": max(1, int(since_days)),
            "until_days_ago": end_offset,
            "half_life_days": float(half_life_days),
            "minimum_rounds": int(minimum_rounds),
            "minimum_actions_per_player": int(minimum_actions_per_player),
            "include_action_statistics": bool(include_action_statistics),
            "quality_filters": [
                "formal_ranked_1v1",
                "official_mods_only",
                "no_disconnect",
                "no_early_surrender",
                "minimum_actions",
                "elo_expected_score_residual",
                "repeat_match_downweight",
                "recency_decay",
            ],
        },
        "quality": _rounded_counter(quality),
        "global": _rounded_counter(global_stats),
        "deck_statistics": _render_deck_statistics(deck_statistics),
        "statistics": {
            key: _rounded_counter(value)
            for key, value in sorted(statistics.items())
            if (
                float(value.get("context_exposures", 0.0) or 0.0) >= 5.0
                or float(value.get("historical_uses", 0.0) or 0.0) > 0.0
            )
        },
    }


def _accumulate_drafted_decks(
    replay: dict[str, Any],
    *,
    summary: dict[str, Any],
    match_weight: float,
    skill_weights: tuple[float, float],
    deck_statistics: dict[str, dict[str, Any]],
) -> bool:
    drafts = summary.get("draft_card_ids_by_player")
    if not isinstance(drafts, list) or len(drafts) != 2:
        return False
    mod_ids = _replay_mod_ids(replay)
    if not mod_ids:
        return False
    exact_key = json.dumps(mod_ids, ensure_ascii=False, separators=(",", ":"))
    accepted = 0
    for player_id, raw_cards in enumerate(drafts):
        if not isinstance(raw_cards, list):
            continue
        cards = [str(value) for value in raw_cards if str(value)]
        if not cards:
            continue
        weight = float(match_weight) * float(skill_weights[player_id])
        for key in ("*", exact_key):
            bucket = deck_statistics.setdefault(key, {
                "effective_decks": 0.0,
                "raw_decks": 0,
                "card_weights": Counter(),
                "raw_card_counts": Counter(),
            })
            bucket["effective_decks"] += weight
            bucket["raw_decks"] += 1
            for def_id in cards:
                bucket["card_weights"][def_id] += weight
                bucket["raw_card_counts"][def_id] += 1
        accepted += 1
    return accepted == 2


def _replay_mod_ids(replay: dict[str, Any]) -> tuple[str, ...]:
    current_state: dict[str, Any] = {}
    for _, frame in _ordered_replay_frames(replay):
        frame_state = frame.get("state")
        if not isinstance(frame_state, dict) or not frame_state:
            continue
        current_state = _merge_replay_state(current_state, frame_state)
        perspectives = current_state.get("perspectives")
        if not isinstance(perspectives, list):
            continue
        for perspective in perspectives:
            if not isinstance(perspective, dict):
                continue
            load_order = perspective.get("v2_load_order")
            if isinstance(load_order, list) and load_order:
                return tuple(sorted({
                    str(value) for value in load_order if str(value)
                }))
    return ()


def _render_deck_statistics(
    deck_statistics: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, bucket in sorted(deck_statistics.items()):
        card_weights = bucket.get("card_weights") or {}
        raw_counts = bucket.get("raw_card_counts") or {}
        result[key] = {
            "effective_decks": round(float(bucket.get("effective_decks", 0.0)), 6),
            "raw_decks": int(bucket.get("raw_decks", 0) or 0),
            "card_weights": _rounded_counter(Counter(card_weights)),
            "raw_card_counts": {
                str(card_id): int(value)
                for card_id, value in sorted(raw_counts.items())
                if int(value) > 0
            },
        }
    return result


def historical_action_keys(
    raw_action: dict[str, Any], *, actor: int
) -> tuple[str, ...]:
    action_type = str(raw_action.get("type") or "")
    kind = {
        "draft_pick": "draft_pick",
        "end_turn": "end_turn",
        "play_card": "play_card",
    }.get(action_type)
    if kind is None:
        return ()
    payload = raw_action.get("payload") if isinstance(raw_action.get("payload"), dict) else {}
    card_id = str(payload.get("def_id") or "")
    if kind == "play_card" and not card_id:
        return ()
    target_id = payload.get("target_player_id")
    choice = payload.get("choice") if isinstance(payload.get("choice"), dict) else {}
    if target_id is None:
        target_id = choice.get("target_player_id", choice.get("target_player"))
    target = _target_relation(target_id, seat=actor)
    phase = "pregame" if kind == "draft_pick" else "combat"
    round_bucket = _round_bucket(raw_action.get("round"), phase=phase)
    return _hierarchical_keys(phase, kind, card_id, target, round_bucket)


def contextual_action_keys_from_values(
    *,
    kind: str,
    card_id: str,
    round_value: Any,
    own_health: Any,
    own_max_health: Any,
    opponent_health: Any,
    opponent_max_health: Any,
    elixir: Any,
    magic: Any,
    hand_count: Any,
) -> tuple[str, ...]:
    identity = str(card_id or "__end_turn__")
    round_bucket = _round_bucket(round_value, phase="combat")
    own_health_bucket = _health_bucket(own_health, own_max_health)
    opponent_health_bucket = _health_bucket(opponent_health, opponent_max_health)
    elixir_bucket = _resource_bucket(elixir)
    magic_bucket = _resource_bucket(magic)
    hand_bucket = _hand_bucket(hand_count)
    levels = (
        (
            "context",
            kind,
            identity,
            round_bucket,
            own_health_bucket,
            opponent_health_bucket,
            elixir_bucket,
            magic_bucket,
            hand_bucket,
        ),
        (
            "context",
            kind,
            identity,
            round_bucket,
            own_health_bucket,
            opponent_health_bucket,
            elixir_bucket,
            magic_bucket,
            "*",
        ),
        (
            "context",
            kind,
            identity,
            round_bucket,
            own_health_bucket,
            opponent_health_bucket,
            "*",
            "*",
            "*",
        ),
        (
            "context",
            kind,
            identity,
            round_bucket,
            "*",
            "*",
            "*",
            "*",
            "*",
        ),
        ("context", kind, identity, "*", "*", "*", "*", "*", "*"),
    )
    return tuple(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        for value in levels
    )


def _accumulate_contextual_choices(
    replay: dict[str, Any],
    *,
    summary: dict[str, Any],
    winner: int,
    match_weight: float,
    expected: tuple[float, float],
    skill_weights: tuple[float, float],
    statistics,
    global_stats: Counter[str],
    quality: Counter[str],
) -> None:
    current_state: dict[str, Any] = {}
    for frame_kind, frame in _ordered_replay_frames(replay):
        before_state = current_state
        if frame_kind == "action":
            action_type = str(frame.get("type") or "")
            actor = _as_int(frame.get("actor"), -1)
            if action_type in {"play_card", "end_turn"} and actor in (0, 1):
                context = _private_actor_context(before_state, actor)
                if context is None:
                    quality["context_missing_prestate"] += 1
                else:
                    actual = 0.5 if winner < 0 else (1.0 if winner == actor else 0.0)
                    adjusted_score = max(0.0, min(1.0, 0.5 + actual - expected[actor]))
                    behavior_weight = (
                        match_weight
                        * skill_weights[actor]
                        * (0.75 + 0.5 * adjusted_score)
                    )
                    _accumulate_contextual_decision(
                        frame,
                        actor=actor,
                        context=context,
                        weight=behavior_weight,
                        statistics=statistics,
                        global_stats=global_stats,
                        quality=quality,
                    )
        frame_state = frame.get("state")
        if isinstance(frame_state, dict) and frame_state:
            current_state = _merge_replay_state(current_state, frame_state)


def _accumulate_contextual_decision(
    frame: dict[str, Any],
    *,
    actor: int,
    context: dict[str, Any],
    weight: float,
    statistics,
    global_stats: Counter[str],
    quality: Counter[str],
) -> None:
    action_type = str(frame.get("type") or "")
    hand = context["hand"]
    candidates = [
        ("play_card", str(card.get("def_id") or ""))
        for card in hand
        if isinstance(card, dict) and str(card.get("def_id") or "")
    ]
    candidates.append(("end_turn", ""))
    payload = frame.get("payload") if isinstance(frame.get("payload"), dict) else {}
    if action_type == "end_turn":
        selected = ("end_turn", "")
    else:
        selected_def_id = str(payload.get("def_id") or "")
        selected_instance_id = str(payload.get("card_instance_id") or "")
        if selected_instance_id:
            matching = next((
                card for card in hand
                if isinstance(card, dict)
                and str(card.get("instance_id") or "") == selected_instance_id
            ), None)
            if matching is not None:
                selected_def_id = str(matching.get("def_id") or selected_def_id)
        if not selected_def_id or not any(
            kind == "play_card" and card_id == selected_def_id
            for kind, card_id in candidates
        ):
            quality["context_selected_card_missing"] += 1
            return
        selected = ("play_card", selected_def_id)

    for kind, card_id in candidates:
        keys = contextual_action_keys_from_values(
            kind=kind,
            card_id=card_id,
            round_value=frame.get("round"),
            own_health=context["own"].get("health"),
            own_max_health=context["own"].get("max_health"),
            opponent_health=context["opponent"].get("health"),
            opponent_max_health=context["opponent"].get("max_health"),
            elixir=context["own"].get("elixir"),
            magic=context["own"].get("magic"),
            hand_count=len(hand),
        )
        for key in keys:
            statistics[key]["context_exposures"] += weight
            statistics[key]["context_raw_exposures"] += 1
            global_stats["context_exposures"] += weight
            global_stats["context_raw_exposures"] += 1
    selected_keys = contextual_action_keys_from_values(
        kind=selected[0],
        card_id=selected[1],
        round_value=frame.get("round"),
        own_health=context["own"].get("health"),
        own_max_health=context["own"].get("max_health"),
        opponent_health=context["opponent"].get("health"),
        opponent_max_health=context["opponent"].get("max_health"),
        elixir=context["own"].get("elixir"),
        magic=context["own"].get("magic"),
        hand_count=len(hand),
    )
    for key in selected_keys:
        statistics[key]["context_selections"] += weight
        statistics[key]["context_raw_selections"] += 1
        global_stats["context_selections"] += weight
        global_stats["context_raw_selections"] += 1
    quality["context_decisions"] += 1


def _ordered_replay_frames(replay: dict[str, Any]):
    entries = []
    order = 0
    for kind, frames in (
        ("keyframe", replay.get("keyframes") or []),
        ("action", replay.get("actions") or []),
    ):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            entries.append((
                frame.get("seq"),
                _as_int(frame.get("t"), 0),
                order,
                kind,
                frame,
            ))
            order += 1
    has_sequence = any(value[0] is not None for value in entries)
    if has_sequence:
        entries.sort(key=lambda value: (
            _as_int(value[0], 10**12 + value[2]) if value[0] is not None else 10**12 + value[2],
            value[2],
        ))
    else:
        entries.sort(key=lambda value: (value[1], value[2]))
    return ((kind, frame) for _, _, _, kind, frame in entries)


def _merge_replay_state(previous: Any, frame_state: dict[str, Any]):
    if frame_state.get("compact") and isinstance(previous, dict):
        merged = dict(previous)
        merged.update(frame_state)
        return merged
    if frame_state.get("delta") is True:
        return _apply_replay_delta(previous or {}, frame_state.get("patch") or {})
    return frame_state


def _apply_replay_delta(previous: Any, patch: Any):
    if not isinstance(patch, dict):
        return previous
    if "$set" in patch:
        return patch.get("$set")
    if "$list" in patch:
        spec = patch.get("$list") if isinstance(patch.get("$list"), dict) else {}
        base = list(previous) if isinstance(previous, list) else []
        start = max(0, min(len(base), _as_int(spec.get("start"), 0)))
        items = spec.get("items") if isinstance(spec.get("items"), list) else []
        return base[:start] + items
    if "$items" in patch:
        base = list(previous) if isinstance(previous, list) else []
        changes = patch.get("$items") if isinstance(patch.get("$items"), dict) else {}
        for raw_index, child in changes.items():
            index = _as_int(raw_index, -1)
            if 0 <= index < len(base):
                base[index] = _apply_replay_delta(base[index], child)
        return base
    if "$dict" in patch:
        base = dict(previous) if isinstance(previous, dict) else {}
        for key in patch.get("$remove") or []:
            base.pop(key, None)
        changes = patch.get("$dict") if isinstance(patch.get("$dict"), dict) else {}
        for key, child in changes.items():
            base[key] = _apply_replay_delta(base.get(key), child)
        return base
    return previous


def _private_actor_context(state: dict[str, Any], actor: int):
    perspectives = state.get("perspectives") if isinstance(state, dict) else None
    perspective = perspectives[0] if isinstance(perspectives, list) and perspectives else None
    players = perspective.get("spectate_players") if isinstance(perspective, dict) else None
    if not isinstance(players, list) or len(players) != 2:
        return None
    own = players[actor]
    opponent = players[1 - actor]
    if not isinstance(own, dict) or not isinstance(opponent, dict):
        return None
    hand = own.get("hand")
    if not isinstance(hand, list):
        return None
    return {"own": own, "opponent": opponent, "hand": hand}


def _skill_weights(summary: dict[str, Any]) -> tuple[float, float]:
    player_ids = summary.get("player_ids")
    before = (summary.get("gr_result") or {}).get("before")
    if not isinstance(player_ids, list) or len(player_ids) != 2 or not isinstance(before, dict):
        return 1.0, 1.0
    result = []
    for player_id in player_ids:
        value = before.get(str(player_id))
        rating = _as_float((value or {}).get("total_gr"), 1000.0) if isinstance(value, dict) else 1000.0
        result.append(max(0.75, min(1.25, 1.0 + (rating - 1000.0) / 800.0)))
    return float(result[0]), float(result[1])


def _metadata_rejection(
    row,
    summary: dict[str, Any],
    *,
    minimum_rounds: int,
    minimum_actions_per_player: int,
) -> str | None:
    if str(row["mode"] or "") != "1v1":
        return "mode"
    if str(row["mod_source"] or "official") not in {"", "official"}:
        return "non_official_mod"
    if str(row["community_mod_name"] or ""):
        return "community_mod"
    if summary.get("valid_for_ranking") is not True:
        return "not_ranked"
    if summary.get("community_mods") or summary.get("entertainment_mods"):
        return "non_official_mod"
    result = str(row["result"] or summary.get("result") or "").lower()
    if result not in {"win", "draw", "finished"}:
        return "result"
    rounds = max(_as_int(row["round_num"], 0), _as_int(summary.get("rounds"), 0))
    if rounds < int(minimum_rounds):
        return "too_short"
    action_counts = summary.get("valid_action_counts_by_side")
    if not isinstance(action_counts, list) or len(action_counts) != 2:
        return "missing_action_counts"
    if min((_as_int(value, 0) for value in action_counts), default=0) < int(
        minimum_actions_per_player
    ):
        return "too_few_actions"
    return None


def _action_stream_rejection(
    replay: dict[str, Any], *, minimum_rounds: int
) -> str | None:
    meta = replay.get("meta") if isinstance(replay.get("meta"), dict) else {}
    if bool(meta.get("truncated") or replay.get("truncated")):
        return "truncated"
    actions = [value for value in replay.get("actions") or [] if isinstance(value, dict)]
    action_types = {str(value.get("type") or "") for value in actions}
    if action_types & {"disconnect_timeout", "both_disconnected_result", "player_exit"}:
        return "disconnect"
    maximum_round = max((_as_int(value.get("round"), 0) for value in actions), default=0)
    if "surrender" in action_types and maximum_round < int(minimum_rounds):
        return "early_surrender"
    return None


def _load_replay(row, root: Path) -> dict[str, Any]:
    compressed_size = _as_int(row["replay_size"], 0)
    if compressed_size <= 0 or compressed_size > MAX_COMPRESSED_BYTES:
        raise ValueError("compressed_size")
    relative = str(row["replay_blob_path"] or "").replace("\\", "/").strip("/")
    if relative:
        parts = [part for part in relative.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError("blob_path")
        candidate = (root.joinpath(*parts)).resolve()
        if os.path.commonpath((root, candidate)) != str(root):
            raise ValueError("blob_path")
        compressed = candidate.read_bytes()
    else:
        compressed = bytes(row["replay_blob"] or b"")
    if len(compressed) != compressed_size:
        raise ValueError("blob_size")
    inflater = zlib.decompressobj()
    decoded = inflater.decompress(compressed, MAX_DECODED_BYTES + 1)
    if len(decoded) > MAX_DECODED_BYTES or inflater.unconsumed_tail:
        raise ValueError("decoded_size")
    decoded += inflater.flush(MAX_DECODED_BYTES + 1 - len(decoded))
    if not inflater.eof or len(decoded) > MAX_DECODED_BYTES:
        raise ValueError("decoded_truncated")
    replay = json.loads(decoded.decode("utf-8"))
    if not isinstance(replay, dict):
        raise ValueError("payload")
    return replay


def _expected_scores(summary: dict[str, Any]) -> tuple[float, float]:
    player_ids = summary.get("player_ids")
    before = (summary.get("gr_result") or {}).get("before")
    if not isinstance(player_ids, list) or len(player_ids) != 2 or not isinstance(before, dict):
        return 0.5, 0.5
    ratings = []
    for player_id in player_ids:
        value = before.get(str(player_id))
        if not isinstance(value, dict):
            return 0.5, 0.5
        ratings.append(_as_float(value.get("total_gr"), 1000.0))
    expected_zero = 1.0 / (1.0 + 10.0 ** ((ratings[1] - ratings[0]) / 400.0))
    return expected_zero, 1.0 - expected_zero


def _winner_index(row, summary: dict[str, Any], replay: dict[str, Any]) -> int:
    meta = replay.get("meta") if isinstance(replay.get("meta"), dict) else {}
    for value in (summary.get("winner_index"), row["winner_index"], meta.get("winner_index")):
        parsed = _as_int(value, -2)
        if parsed in (-1, 0, 1):
            return parsed
    if str(summary.get("result") or row["result"] or "").lower() == "draw":
        return -1
    return -2


def _recency_weight(created_at: Any, *, half_life_days: float) -> float:
    try:
        parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        age_days = float(half_life_days)
    return 0.5 ** (age_days / max(0.1, float(half_life_days)))


def _json_object(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _target_relation(target: Any, *, seat: int) -> str:
    target_id = _as_int(target, -1)
    if target_id < 0:
        return "none"
    return "self" if target_id == seat else "opponent"


def _hierarchical_keys(
    phase: str, kind: str, card_id: str, target: str, round_bucket: str
) -> tuple[str, ...]:
    values = (
        (phase, kind, card_id, target, round_bucket),
        (phase, kind, card_id, target, "*"),
        (phase, kind, card_id, "*", "*"),
    )
    return tuple(json.dumps(value, ensure_ascii=False, separators=(",", ":")) for value in values)


def _round_bucket(value: Any, *, phase: str) -> str:
    if phase == "pregame":
        return "pregame"
    round_number = _as_int(value, 0)
    if round_number <= 2:
        return "1-2"
    if round_number <= 5:
        return "3-5"
    if round_number <= 9:
        return "6-9"
    return "10+"


def _health_bucket(health: Any, maximum: Any) -> str:
    max_value = max(1.0, _as_float(maximum, 1.0))
    ratio = _as_float(health, 0.0) / max_value
    if ratio <= 0.25:
        return "0-25%"
    if ratio <= 0.5:
        return "26-50%"
    if ratio <= 0.75:
        return "51-75%"
    return "76%+"


def _resource_bucket(value: Any) -> str:
    amount = _as_int(value, 0)
    if amount <= 0:
        return "0"
    if amount <= 2:
        return "1-2"
    if amount <= 4:
        return "3-4"
    return "5+"


def _hand_bucket(value: Any) -> str:
    count = _as_int(value, 0)
    if count <= 3:
        return "0-3"
    if count <= 6:
        return "4-6"
    return "7+"


def _rounded_counter(values: Counter[str]) -> dict[str, int | float]:
    result = {}
    for key, value in sorted(values.items()):
        numeric = float(value)
        result[key] = int(numeric) if numeric.is_integer() else round(numeric, 6)
    return result


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".json", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an identity-free historical action aggregate from the GTN database"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--blob-root")
    parser.add_argument("--output", required=True, help="Output JSON path or - for stdout")
    parser.add_argument("--since-days", type=int, default=45)
    parser.add_argument("--until-days-ago", type=int, default=0)
    parser.add_argument("--half-life-days", type=float, default=14.0)
    parser.add_argument("--minimum-rounds", type=int, default=4)
    parser.add_argument("--minimum-actions", type=int, default=8)
    parser.add_argument(
        "--deck-only",
        action="store_true",
        help="Skip action/context statistics and emit only replay quality plus deck data",
    )
    args = parser.parse_args(argv)
    aggregate = aggregate_replay_database(
        args.db,
        blob_root=args.blob_root,
        since_days=args.since_days,
        half_life_days=args.half_life_days,
        minimum_rounds=args.minimum_rounds,
        minimum_actions_per_player=args.minimum_actions,
        include_action_statistics=not args.deck_only,
        until_days_ago=args.until_days_ago,
    )
    if args.output == "-":
        json.dump(aggregate, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        sys.stdout.write("\n")
    else:
        _write_json(args.output, aggregate)
        print(json.dumps({
            "output": str(Path(args.output).resolve()),
            "quality": aggregate["quality"],
            "statistics": len(aggregate["statistics"]),
            "deck_buckets": len(aggregate["deck_statistics"]),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
