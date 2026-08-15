from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Iterable, Sequence

from .contextual_value import contextual_value_tokens, enrich_card_for_context
from .protocol import Action


STRUCTURED_FEATURE_SCHEMA_VERSION = 1


class TokenType(IntEnum):
    PAD = 0
    CLS = 1
    GLOBAL = 2
    MOD = 3
    PLAYER = 4
    STATUS = 5
    EQUIPMENT = 6
    CARD = 7
    PENDING = 8
    HISTORY = 9
    REVEAL = 10
    EVENT = 11
    ACTION = 12


@dataclass(frozen=True)
class StructuredFeatureConfig:
    categorical_buckets: int = 1 << 15
    categorical_slots: int = 32
    numeric_buckets: int = 32
    max_state_tokens: int = 192
    max_history_events: int = 32
    contextual_value_features: bool = False

    def __post_init__(self) -> None:
        for name in (
            "categorical_buckets",
            "categorical_slots",
            "numeric_buckets",
            "max_state_tokens",
            "max_history_events",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class EntityToken:
    token_type: int
    categorical_ids: tuple[int, ...]
    numeric_values: tuple[float, ...]


@dataclass(frozen=True)
class StructuredDecision:
    state_tokens: tuple[EntityToken, ...]
    action_tokens: tuple[EntityToken, ...]
    phase: int
    selected_index: int = -1
    action_logit_biases: tuple[float, ...] = ()

    @property
    def action_count(self) -> int:
        return len(self.action_tokens)


_PRIORITY_KEYS = (
    "def_id",
    "definition_semantics",
    "instance_semantics",
    "rule_operations",
    "rule_numbers",
    "added_flags",
    "removed_flags",
    "card_type",
    "quality",
    "id",
    "kind",
    "choice_type",
    "status",
    "event_id",
    "flags",
    "disabled_flags",
    "setup_modifiers",
    "target_player_id",
    "target_player",
    "owner",
    "effect_target",
    "slot",
)

_IGNORED_TEXT_KEYS = frozenset({
    "name",
    "name_cn",
    "name_en",
    "description",
    "desc",
    "display_name_cn",
    "display_name_en",
    "display_effect_text_cn",
    "display_effect_text_en",
})

_NUMERIC_SCALES = {
    "health": 100.0,
    "max_health": 100.0,
    "elixir": 10.0,
    "max_elixir": 10.0,
    "magic": 10.0,
    "max_magic": 10.0,
    "armor": 20.0,
    "hand_limit": 10.0,
    "hand_count": 10.0,
    "deck_count": 20.0,
    "discard_count": 20.0,
    "exile_count": 20.0,
    "round": 20.0,
    "cost_e": 10.0,
    "cost_m": 10.0,
    "base_damage": 25.0,
    "base_hits": 5.0,
    "bonus_damage": 20.0,
    "power": 20.0,
    "fission_level": 5.0,
    "fusion_level": 5.0,
    "extra_hits": 5.0,
    "count": 10.0,
    "slot": 10.0,
    "position": 10.0,
    "probability": 1.0,
    "expected_count": 1.0,
    "card_count": 20.0,
    "definition_count": 20.0,
    "instance_variant_count": 20.0,
    "hand_free_slots": 10.0,
    "status_count": 10.0,
    "equipment_count": 10.0,
    "trigger_cost_e": 10.0,
    "trigger_cost_m": 10.0,
    "cost_delta_e": 10.0,
    "cost_delta_m": 10.0,
    "fission_delta": 5.0,
    "fusion_delta": 5.0,
}


def encode_structured_decision(
    observation: dict[str, Any],
    legal_actions: Sequence[Action],
    *,
    config: StructuredFeatureConfig,
    selected_index: int = -1,
) -> StructuredDecision:
    actions = list(legal_actions)
    if not actions:
        raise ValueError("cannot encode a decision without legal actions")
    if selected_index < -1 or selected_index >= len(actions):
        raise ValueError("selected action index is outside the legal action set")
    return StructuredDecision(
        state_tokens=tuple(encode_state_tokens(observation, config=config)),
        action_tokens=tuple(
            encode_action_token(observation, action, config=config)
            for action in actions
        ),
        phase=0 if observation.get("phase") == "pregame" else 1,
        selected_index=int(selected_index),
        action_logit_biases=tuple(_progress_prior_biases(observation, actions)),
    )


def encode_state_tokens(
    observation: dict[str, Any],
    *,
    config: StructuredFeatureConfig,
) -> list[EntityToken]:
    essential: list[EntityToken] = [
        _entity_token(
            TokenType.CLS,
            "cls",
            {"phase": observation.get("phase")},
            config=config,
        ),
        _entity_token(
            TokenType.GLOBAL,
            "global",
            {
                key: observation.get(key)
                for key in (
                    "ruleset",
                    "phase",
                    "pregame_phase",
                    "round",
                    "seat",
                    "current_player",
                    "first_player",
                    "decision_player",
                    "winner",
                    "opening_events",
                )
            },
            config=config,
        ),
    ]
    cards: list[EntityToken] = []
    secondary: list[EntityToken] = []
    history: list[EntityToken] = []

    if config.contextual_value_features:
        for namespace, value in contextual_value_tokens(observation):
            essential.append(_entity_token(
                TokenType.GLOBAL,
                namespace,
                value,
                config=config,
            ))

    loadout = observation.get("loadout") or {}
    for position, mod_name in enumerate(loadout.get("official_mods") or []):
        secondary.append(_entity_token(
            TokenType.MOD,
            "mod",
            {"id": mod_name, "position": position},
            config=config,
        ))

    for relation in ("self", "opponent"):
        player = observation.get(relation) or {}
        essential.append(_entity_token(
            TokenType.PLAYER,
            f"player:{relation}",
            _without_collections(player),
            config=config,
        ))
        for status_id, value in sorted((player.get("statuses") or {}).items()):
            essential.append(_entity_token(
                TokenType.STATUS,
                f"status:{relation}",
                {"id": status_id, "value": value},
                config=config,
            ))
        for position, equipment in enumerate(player.get("equipment") or []):
            essential.append(_entity_token(
                TokenType.EQUIPMENT,
                f"equipment:{relation}",
                {"position": position, **equipment},
                config=config,
            ))

        if relation == "self":
            for position, card in enumerate(player.get("hand") or []):
                cards.append(_card_token(
                    card,
                    relation=relation,
                    zone="hand",
                    position=position,
                    config=config,
                ))
            _append_visible_pile_tokens(cards, player, relation=relation, config=config)
        else:
            for position, card in enumerate(player.get("revealed_hand") or []):
                cards.append(_card_token(
                    card,
                    relation=relation,
                    zone="revealed_hand",
                    position=position,
                    config=config,
                ))
            for zone in ("deck_ordered", "discard_ordered"):
                for position, card in enumerate(player.get(zone) or []):
                    cards.append(_card_token(
                        card,
                        relation=relation,
                        zone=zone,
                        position=position,
                        config=config,
                    ))

    own = observation.get("self") or {}
    for zone in ("draft_options", "draft_picks", "opening_event_options"):
        token_type = TokenType.EVENT if zone == "opening_event_options" else TokenType.CARD
        for position, value in enumerate(own.get(zone) or []):
            secondary.append(_entity_token(
                token_type,
                f"pregame:{zone}",
                {"position": position, **value},
                config=config,
            ))
    if isinstance(own.get("opening_event"), dict):
        secondary.append(_entity_token(
            TokenType.EVENT,
            "pregame:opening_event",
            own["opening_event"],
            config=config,
        ))
    sub_choice = own.get("sub_choice") or {}
    if sub_choice:
        essential.append(_entity_token(
            TokenType.PENDING,
            "pregame:sub_choice",
            {key: value for key, value in sub_choice.items() if key != "candidates"},
            config=config,
        ))
        for position, card in enumerate(sub_choice.get("candidates") or []):
            essential.append(_card_token(
                card,
                relation="self",
                zone="pregame_candidate",
                position=position,
                config=config,
            ))

    pending = observation.get("pending") or {}
    if pending:
        essential.append(_entity_token(
            TokenType.PENDING,
            "pending",
            {key: value for key, value in pending.items() if key != "candidates"},
            config=config,
        ))
        source_card = pending.get("card")
        if isinstance(source_card, dict):
            essential.append(_card_token(
                source_card,
                relation="pending",
                zone="source",
                position=0,
                config=config,
            ))
        for position, candidate in enumerate(pending.get("candidates") or []):
            if isinstance(candidate, dict) and _looks_like_card(candidate):
                token = _card_token(
                    candidate,
                    relation="pending",
                    zone="candidate",
                    position=position,
                    config=config,
                )
            else:
                token = _entity_token(
                    TokenType.PENDING,
                    "pending:candidate",
                    {"position": position, "value": candidate},
                    config=config,
                )
            essential.append(token)

    for position, reveal in enumerate(observation.get("temporary_reveals") or []):
        secondary.append(_entity_token(
            TokenType.REVEAL,
            "temporary_reveal",
            {"position": position, "value": reveal},
            config=config,
        ))

    for position, card in enumerate(observation.get("opponent_deck_belief") or []):
        if isinstance(card, dict) and card.get("def_id"):
            secondary.append(_card_token(
                card,
                relation="opponent",
                zone="belief",
                position=position,
                config=config,
            ))

    events = observation.get("public_history") or []
    if isinstance(events, list):
        visible_events = events[-config.max_history_events:]
        offset = max(0, len(events) - len(visible_events))
        for position, event in enumerate(visible_events, start=offset):
            history.append(_entity_token(
                TokenType.HISTORY,
                "history",
                {"position": position, "event": event},
                config=config,
            ))

    # Keep immediate state and choices first. Pile summaries and older public
    # history are useful, but are the first information dropped at the cap.
    tokens = essential + cards + secondary + history
    return tokens[:config.max_state_tokens]


def encode_action_token(
    observation: dict[str, Any],
    action: Action,
    *,
    config: StructuredFeatureConfig,
) -> EntityToken:
    payload: dict[str, Any] = {"kind": action.kind, "payload": action.payload}
    target = action.payload.get("target_player_id")
    choice = action.payload.get("choice")
    if target is None and isinstance(choice, dict):
        target = choice.get("target_player_id", choice.get("target_player"))
    if target is not None:
        payload["target_relation"] = (
            "self" if _same_int(target, observation.get("seat")) else "opponent"
        )
    selected = _selected_visible_card(observation, action)
    if selected is not None:
        payload["selected_card"] = (
            enrich_card_for_context(selected)
            if config.contextual_value_features
            else selected
        )
    return _entity_token(
        TokenType.ACTION,
        f"action:{action.kind}",
        payload,
        config=config,
    )


def _append_visible_pile_tokens(
    output: list[EntityToken],
    player: dict[str, Any],
    *,
    relation: str,
    config: StructuredFeatureConfig,
) -> None:
    for zone in ("deck", "discard", "exile"):
        ordered_key = f"{zone}_ordered"
        if isinstance(player.get(ordered_key), list):
            values = player[ordered_key]
            actual_zone = ordered_key
        else:
            values = player.get(zone) or []
            actual_zone = zone
        for position, raw in enumerate(values):
            if not isinstance(raw, dict):
                continue
            card = raw.get("card") if isinstance(raw.get("card"), dict) else raw
            count = raw.get("count", 1)
            output.append(_card_token(
                card,
                relation=relation,
                zone=actual_zone,
                position=position,
                count=count,
                config=config,
            ))


def _card_token(
    card: dict[str, Any],
    *,
    relation: str,
    zone: str,
    position: int,
    config: StructuredFeatureConfig,
    count: Any = 1,
) -> EntityToken:
    visible_card = (
        enrich_card_for_context(card)
        if config.contextual_value_features
        else dict(card)
    )
    return _entity_token(
        TokenType.CARD,
        f"card:{relation}:{zone}",
        {"position": position, "count": count, **visible_card},
        config=config,
    )


def _entity_token(
    token_type: TokenType,
    namespace: str,
    value: Any,
    *,
    config: StructuredFeatureConfig,
) -> EntityToken:
    categorical = [
        _categorical_id(f"type:{token_type.name.lower()}", config.categorical_buckets),
        _categorical_id(f"namespace:{namespace}", config.categorical_buckets),
    ]
    numeric = [0.0] * config.numeric_buckets
    _walk_entity(
        categorical,
        numeric,
        namespace,
        value,
        config=config,
        depth=0,
        max_depth=5,
    )
    unique: list[int] = []
    seen: set[int] = set()
    for token_id in categorical:
        if token_id not in seen:
            seen.add(token_id)
            unique.append(token_id)
        if len(unique) >= config.categorical_slots:
            break
    return EntityToken(
        token_type=int(token_type),
        categorical_ids=tuple(unique),
        numeric_values=tuple(numeric),
    )


def _walk_entity(
    categorical: list[int],
    numeric: list[float],
    path: str,
    value: Any,
    *,
    config: StructuredFeatureConfig,
    depth: int,
    max_depth: int,
) -> None:
    if depth > max_depth:
        return
    if isinstance(value, dict):
        keys = [key for key in _PRIORITY_KEYS if key in value]
        keys.extend(sorted(key for key in value if key not in keys))
        for key in keys:
            if str(key) in _IGNORED_TEXT_KEYS:
                continue
            _walk_entity(
                categorical,
                numeric,
                f"{path}.{key}",
                value[key],
                config=config,
                depth=depth + 1,
                max_depth=max_depth,
            )
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        _add_numeric(numeric, f"{path}.length", len(values), config.numeric_buckets)
        for item in values[:24]:
            _walk_entity(
                categorical,
                numeric,
                f"{path}[]",
                item,
                config=config,
                depth=depth + 1,
                max_depth=max_depth,
            )
    elif isinstance(value, bool):
        categorical.append(_categorical_id(f"{path}:{value}", config.categorical_buckets))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _add_numeric(numeric, path, value, config.numeric_buckets)
        categorical.append(_categorical_id(
            f"{path}:bucket:{_log_bucket(value)}",
            config.categorical_buckets,
        ))
    elif value is not None:
        text = str(value)
        if len(text) <= 160:
            categorical.append(_categorical_id(f"{path}:{text}", config.categorical_buckets))
        elif isinstance(value, (dict, list, tuple)):
            canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            categorical.append(_categorical_id(f"{path}:{canonical[:160]}", config.categorical_buckets))


def _add_numeric(output: list[float], path: str, value: Any, buckets: int) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(number):
        return
    first, second, sign, scale = _numeric_projection_spec(path, buckets)
    normalized = max(-4.0, min(4.0, number / scale))
    output[first] += normalized
    output[second] += sign * math.copysign(math.log1p(abs(number)) / 4.0, number)


@lru_cache(maxsize=4096)
def _numeric_projection_spec(path: str, buckets: int) -> tuple[int, int, float, float]:
    field_name = path.rsplit(".", 1)[-1]
    scale = _NUMERIC_SCALES.get(field_name, 10.0)
    digest = hashlib.blake2b(
        path.encode("utf-8"), digest_size=8, person=b"GTNAISN1"
    ).digest()
    first = int.from_bytes(digest[:4], "little") % buckets
    second = int.from_bytes(digest[4:], "little") % buckets
    sign = -1.0 if digest[0] & 1 else 1.0
    return first, second, sign, scale


@lru_cache(maxsize=262144)
def _categorical_id(value: str, buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=b"GTNAISC1").digest()
    return int.from_bytes(digest, "little") % buckets + 1


def _without_collections(player: dict[str, Any]) -> dict[str, Any]:
    hidden = {
        "hand",
        "deck",
        "discard",
        "exile",
        "deck_ordered",
        "discard_ordered",
        "revealed_hand",
        "equipment",
        "statuses",
        "draft_options",
        "draft_picks",
        "opening_event_options",
        "opening_event",
        "sub_choice",
    }
    return {key: value for key, value in player.items() if key not in hidden}


def _looks_like_card(value: dict[str, Any]) -> bool:
    return "def_id" in value or "card_type" in value


def _selected_visible_card(
    observation: dict[str, Any],
    action: Action,
) -> dict[str, Any] | None:
    if action.kind in {"play_card", "respond"}:
        return _find_slot((observation.get("self") or {}).get("hand") or [], action.payload.get("hand_slot"))
    if action.kind == "use_trigger":
        item = _find_slot(
            (observation.get("self") or {}).get("equipment") or [],
            action.payload.get("equipment_slot"),
        )
        return (item.get("card") or {}) if isinstance(item, dict) else None
    if action.kind in {"select_choice", "toggle_choice", "append_choice_order"}:
        return _find_slot(
            (observation.get("pending") or {}).get("candidates") or [],
            action.payload.get("candidate_slot"),
        )
    if action.kind == "v2_ui_response":
        return _v2_selected_visible_card(observation, action)
    own = observation.get("self") or {}
    if action.kind == "draft_pick":
        return _find_slot(own.get("draft_options") or [], action.payload.get("candidate_slot"))
    if action.kind in {"select_pregame_choice", "toggle_pregame_choice", "append_pregame_order"}:
        return _find_slot(
            (own.get("sub_choice") or {}).get("candidates") or [],
            action.payload.get("candidate_slot"),
        )
    if action.kind == "select_opening_event":
        return _find_slot(own.get("opening_event_options") or [], action.payload.get("option_slot"))
    return None


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
                if _looks_like_card(candidate):
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


def _progress_prior_biases(
    observation: dict[str, Any],
    actions: Sequence[Action],
) -> list[float]:
    biases = [0.0] * len(actions)
    pending = observation.get("pending") or {}
    own = observation.get("self") or {}
    pregame = own.get("sub_choice") or {}
    for index, action in enumerate(actions):
        if action.kind == "toggle_choice":
            selection = pending.get("selection") or {}
        elif action.kind == "toggle_pregame_choice":
            selection = pregame.get("selection") or {}
        else:
            continue
        selected = {_slot_key(slot) for slot in selection.get("selected_slots") or []}
        if _slot_key(action.payload.get("candidate_slot")) in selected:
            biases[index] -= 8.0
    return biases


def _slot_key(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _same_int(left: Any, right: Any) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return False


def _log_bucket(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or number == 0:
        return 0
    return int(math.copysign(math.log2(abs(number) + 1), number))
