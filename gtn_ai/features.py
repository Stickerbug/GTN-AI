from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable

from .protocol import Action


FEATURE_SCHEMA_VERSION = 4
NEURAL_FEATURE_SCHEMA_VERSION = 1
DEFAULT_HISTORY_EVENTS = 32
DEFAULT_HISTORY_TOKENS_PER_EVENT = 12


def hashed_action_features(
    observation: dict[str, Any],
    action: Action,
    *,
    buckets: int = 1 << 16,
) -> dict[int, float]:
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    features: dict[int, float] = defaultdict(float)
    _add(features, buckets, "bias", 1.0)
    _populate_action_features(features, buckets, observation, action)
    _populate_observation_features(features, buckets, observation, include_bias=False)
    return dict(features)


def hashed_observation_features(
    observation: dict[str, Any],
    *,
    buckets: int = 1 << 15,
) -> dict[int, float]:
    """Encode only information available to the acting seat."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    features: dict[int, float] = defaultdict(float)
    _populate_observation_features(features, buckets, observation)
    return dict(features)


def hashed_action_only_features(
    observation: dict[str, Any],
    action: Action,
    *,
    buckets: int = 1 << 14,
) -> dict[int, float]:
    """Encode one legal action without duplicating the whole observation."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    features: dict[int, float] = defaultdict(float)
    _add(features, buckets, "bias", 1.0)
    _populate_action_features(features, buckets, observation, action)
    _add(features, buckets, f"action_phase:{observation.get('phase')}:{action.kind}", 1.0)
    target = action.payload.get("target_player_id")
    if target is None and isinstance(action.payload.get("choice"), dict):
        target = action.payload["choice"].get("target_player_id")
    if target is not None:
        relation = "self" if _same_int(target, observation.get("seat")) else "opponent"
        _add(features, buckets, f"action_target:{relation}", 1.0)
    return dict(features)


def history_event_token_ids(
    observation: dict[str, Any],
    *,
    buckets: int = 1 << 13,
    max_events: int = DEFAULT_HISTORY_EVENTS,
    max_tokens_per_event: int = DEFAULT_HISTORY_TOKENS_PER_EVENT,
) -> list[list[int]]:
    """Return a short public-history sequence; zero remains reserved for padding."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    events = observation.get("public_history") or []
    if not isinstance(events, list):
        return []
    seat = observation.get("seat")
    encoded: list[list[int]] = []
    for raw_event in events[-max(0, int(max_events)):]:
        if not isinstance(raw_event, dict):
            continue
        tokens = [
            "history:bias",
            f"history:kind:{raw_event.get('kind', '?')}",
        ]
        if raw_event.get("player") is not None:
            relation = "self" if _same_int(raw_event.get("player"), seat) else "opponent"
            tokens.append(f"history:actor:{relation}")
        if raw_event.get("target_player") is not None:
            relation = "self" if _same_int(raw_event.get("target_player"), seat) else "opponent"
            tokens.append(f"history:target:{relation}")
        card_def_id = raw_event.get("card_def_id")
        if card_def_id:
            tokens.append(f"history:card:{card_def_id}")
        if raw_event.get("round") is not None:
            round_bucket = _log_bucket(raw_event.get("round"))
            tokens.append(f"history:round:{round_bucket}")
        for key, value in sorted(raw_event.items()):
            if key in {"kind", "player", "target_player", "card_def_id", "round"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                tokens.append(f"history:{key}:{value}")
            elif isinstance(value, (dict, list, tuple)):
                canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                tokens.append(f"history:{key}:{canonical}")
            if len(tokens) >= max(1, int(max_tokens_per_event)):
                break
        encoded.append([_token_id(token, buckets) for token in tokens[:max_tokens_per_event]])
    return encoded


def _populate_action_features(
    features: dict[int, float],
    buckets: int,
    observation: dict[str, Any],
    action: Action,
) -> None:
    _add(features, buckets, f"action:{action.kind}", 1.0)
    _walk(features, buckets, "a", action.payload, depth=0, max_depth=5)
    selected_card = _selected_visible_card(observation, action)
    if selected_card is not None:
        _walk(features, buckets, "selected_card", selected_card, depth=0, max_depth=4)


def _populate_observation_features(
    features: dict[int, float],
    buckets: int,
    observation: dict[str, Any],
    *,
    include_bias: bool = True,
) -> None:
    if include_bias:
        _add(features, buckets, "bias", 1.0)
    _add(features, buckets, f"phase:{observation.get('phase')}", 1.0)
    _add(features, buckets, f"seat:{observation.get('seat')}", 1.0)
    _add(features, buckets, f"first:{observation.get('first_player')}", 1.0)
    for mod_name in (observation.get("loadout") or {}).get("official_mods") or []:
        _add(features, buckets, f"official_mod:{mod_name}", 1.0)
    _add_numeric(features, buckets, "round", observation.get("round"), scale=20.0)
    for side in ("self", "opponent"):
        player = observation.get(side) or {}
        for key, scale in (
            ("health", 100.0),
            ("max_health", 100.0),
            ("elixir", 15.0),
            ("magic", 15.0),
            ("armor", 20.0),
            ("hand_count", 15.0),
            ("deck_count", 30.0),
            ("discard_count", 30.0),
            ("exile_count", 30.0),
        ):
            _add_numeric(features, buckets, f"{side}:{key}", player.get(key), scale=scale)
        for status, value in sorted((player.get("statuses") or {}).items()):
            _add(features, buckets, f"{side}:status:{status}", _bounded_number(value, 20.0))
        for equipment in player.get("equipment") or []:
            card = equipment.get("card") or {}
            _add(features, buckets, f"{side}:equipment:{card.get('def_id')}", 1.0)
            _add_numeric(features, buckets, f"{side}:equipment_armor", equipment.get("armor"), scale=10.0)
    for card in ((observation.get("self") or {}).get("hand") or []):
        _add(features, buckets, f"hand:{card.get('def_id', '?')}", 1.0)
        _add(features, buckets, f"hand_type:{card.get('card_type', '?')}", 1.0)
    own = observation.get("self") or {}
    opponent = observation.get("opponent") or {}
    if observation.get("phase") == "pregame":
        _add(features, buckets, f"pregame:{own.get('status')}", 1.0)
        _add(features, buckets, f"pregame_type:{own.get('draft_card_type', '')}", 1.0)
        _add_numeric(features, buckets, "pregame:rerolls", own.get("rerolls"), scale=3.0)
        _add_numeric(features, buckets, "pregame:draft_count", own.get("draft_count"), scale=15.0)
        _add_numeric(features, buckets, "pregame:opponent_draft_count", opponent.get("draft_count"), scale=15.0)
        own_event = own.get("opening_event") or {}
        opponent_event = opponent.get("opening_event") or {}
        _add(features, buckets, f"pregame:own_event:{own_event.get('id', '?')}", 1.0)
        if opponent_event:
            _add(features, buckets, f"pregame:opponent_event:{opponent_event.get('id', '?')}", 1.0)
        for card in own.get("draft_picks") or []:
            _add(features, buckets, f"pregame:pick:{card.get('def_id', '?')}", 1.0)
            _add(features, buckets, f"pregame:pick_type:{card.get('card_type', '?')}", 1.0)
    pending = observation.get("pending") or {}
    if pending:
        _add(features, buckets, f"pending:{pending.get('kind')}:{pending.get('choice_type', '')}", 1.0)


def _selected_visible_card(observation: dict[str, Any], action: Action) -> dict[str, Any] | None:
    if action.kind in {"play_card", "respond"}:
        slot = action.payload.get("hand_slot")
        cards = (observation.get("self") or {}).get("hand") or []
    elif action.kind == "use_trigger":
        slot = action.payload.get("equipment_slot")
        equipment = (observation.get("self") or {}).get("equipment") or []
        item = _find_slot(equipment, slot)
        return (item.get("card") or {}) if isinstance(item, dict) else None
    elif action.kind in {"select_choice", "toggle_choice", "append_choice_order"}:
        slot = action.payload.get("candidate_slot")
        cards = (observation.get("pending") or {}).get("candidates") or []
    elif action.kind == "draft_pick":
        slot = action.payload.get("candidate_slot")
        cards = (observation.get("self") or {}).get("draft_options") or []
    elif action.kind in {"select_pregame_choice", "toggle_pregame_choice", "append_pregame_order"}:
        slot = action.payload.get("candidate_slot")
        cards = ((observation.get("self") or {}).get("sub_choice") or {}).get("candidates") or []
    elif action.kind == "select_opening_event":
        slot = action.payload.get("option_slot")
        cards = (observation.get("self") or {}).get("opening_event_options") or []
    elif action.kind == "v2_ui_response":
        return _v2_selected_visible_card(observation, action)
    else:
        return None
    return _find_slot(cards, slot)


def _v2_selected_visible_card(
    observation: dict[str, Any],
    action: Action,
) -> dict[str, Any] | None:
    candidates = (observation.get("pending") or {}).get("candidates") or []
    controls = action.payload.get("controls")
    if not isinstance(controls, list):
        return None
    for control in controls:
        if not isinstance(control, dict):
            continue
        option_slots = control.get("option_slots")
        if not isinstance(option_slots, list):
            option_slots = [control.get("option_slot")]
        for option_slot in option_slots:
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if not _same_int(candidate.get("control_slot"), control.get("control_slot")):
                    continue
                if not _same_int(candidate.get("option_slot"), option_slot):
                    continue
                if candidate.get("def_id") or candidate.get("card_type"):
                    return candidate
    return None


def _find_slot(items: Iterable[Any], slot: Any) -> dict[str, Any] | None:
    try:
        expected = int(slot)
    except (TypeError, ValueError):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("slot", -1)) == expected:
                return item
        except (TypeError, ValueError):
            continue
    return None


def dot(weights: dict[int, float], features: dict[int, float]) -> float:
    return sum(float(weights.get(index, 0.0)) * value for index, value in features.items())


def _walk(
    output: dict[int, float],
    buckets: int,
    prefix: str,
    value: Any,
    *,
    depth: int,
    max_depth: int,
) -> None:
    if depth > max_depth:
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _walk(output, buckets, f"{prefix}.{key}", value[key], depth=depth + 1, max_depth=max_depth)
    elif isinstance(value, (list, tuple)):
        _add(output, buckets, f"{prefix}.length", min(len(value), 32) / 8.0)
        for item in value[:16]:
            _walk(output, buckets, f"{prefix}[]", item, depth=depth + 1, max_depth=max_depth)
    elif isinstance(value, bool):
        _add(output, buckets, f"{prefix}:{value}", 1.0)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _add(output, buckets, prefix, _bounded_number(value, 20.0))
        _add(output, buckets, f"{prefix}={value}", 1.0)
    elif value is not None:
        _add(output, buckets, f"{prefix}:{value}", 1.0)


def _add_numeric(output: dict[int, float], buckets: int, key: str, value: Any, *, scale: float) -> None:
    if value is None:
        _add(output, buckets, f"{key}:unknown", 1.0)
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    _add(output, buckets, key, max(-4.0, min(4.0, number / max(1e-6, scale))))
    bucketed = int(math.copysign(math.log2(abs(number) + 1), number)) if number else 0
    _add(output, buckets, f"{key}:bucket:{bucketed}", 1.0)


def _bounded_number(value: Any, scale: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-4.0, min(4.0, number / max(1e-6, scale)))


def _add(output: dict[int, float], buckets: int, key: str, value: float) -> None:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"GTNAIF01").digest()
    index = int.from_bytes(digest, "little") % buckets
    output[index] += float(value)


def _token_id(token: str, buckets: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"GTNAIH01").digest()
    return int.from_bytes(digest, "little") % buckets + 1


def _same_int(left: Any, right: Any) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return False


def _log_bucket(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return int(math.copysign(math.log2(abs(number) + 1), number)) if number else 0
