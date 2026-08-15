from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from .protocol import OBSERVATION_SCHEMA_VERSION


PLAYER_STATUS_FIELDS = (
    "poison",
    "fire",
    "toxic",
    "triangle_stacks",
    "dodge",
    "nazar_active",
    "nazar_big_hits",
    "equipment_protection",
    "invincible",
    "skip_turn",
    "forced_skip_turn",
    "attack_blocked",
    "untargetable",
    "sluggish",
    "enemy_draw_reduction",
    "enemy_e_reduction",
    "overload",
    "foresight",
    "fracture",
    "stagnation",
    "blind",
    "heal_block",
    "weakness",
    "bleed",
    "fragment_stacks",
    "attack_only",
    "honey_control_turns",
    "negate_next_skill",
    "bandage_active",
    "bandage_death_pending",
)

# These are card/equipment values the production client renders on the face or
# next to the equipment icon. All other custom_vars are engine bookkeeping and
# stay private unless deliberately reviewed and added here.
PUBLIC_CARD_CUSTOM_FIELDS = frozenset({
    "display_name_cn",
    "display_name_en",
    "display_effect_text_cn",
    "display_effect_text_en",
    "jungle_dianthus_power",
    "sewers_toilet_paper_power",
})
PUBLIC_EQUIPMENT_CUSTOM_FIELDS = frozenset({
    "layers",
    "layer",
    "durability",
    "charges",
    "charge",
    "jungle_root_layers",
    "sewers_sealed",
})


def build_observation(env, viewer_id: int) -> dict[str, Any]:
    """Return only information a normal client at this seat may know."""

    engine = env.engine
    if engine is None:
        raise RuntimeError("environment has not been reset")
    if viewer_id not in (0, 1):
        raise ValueError("viewer_id must be 0 or 1")

    if env._in_pregame():
        return _build_pregame_observation(env, viewer_id)

    public = engine.get_public_state(viewer_id)
    own = engine.players[viewer_id]
    blind_level = max(0, int(getattr(own, "blind", 0) or 0))
    opponent_id = 1 - viewer_id
    opponent = engine.players[opponent_id]
    public_own = public.get("you") or {}
    public_opponent = public.get("opponent") or {}

    own_view = _player_view(own, include_cards=True)
    own_view["hand"] = [
        {"slot": slot, **_visible_own_hand_card(card, blind_level)}
        for slot, card in enumerate(own.hand)
    ]
    own_view["hand_count"] = len(own.hand)
    own_view["deck_count"] = len(own.deck)
    own_view["discard_count"] = len(own.discard)
    own_view["exile_count"] = len(own.exile)
    if blind_level < 1:
        own_view["deck"] = _zone_multiset(own.deck)
    else:
        own_view["deck_unknown"] = True
    if blind_level < 2:
        own_view["discard"] = _zone_multiset(own.discard)
        own_view["exile"] = _zone_multiset(own.exile)
    else:
        own_view["discard_unknown"] = True
        own_view["exile_unknown"] = True
        for key in ("health", "elixir", "magic"):
            own_view[key] = None
    if blind_level >= 3:
        for key in ("hand_count", "deck_count", "discard_count", "exile_count"):
            own_view[key] = None
    for zone in ("deck", "discard"):
        ordered = public_own.get(f"{zone}_ordered")
        if blind_level < (1 if zone == "deck" else 2) and isinstance(ordered, list):
            own_view[f"{zone}_ordered"] = [
                {"slot": slot, **_card_dict_view(card)}
                for slot, card in enumerate(ordered)
            ]

    opponent_view = _player_view(opponent, include_cards=False)
    opponent_view.update(
        hand_count=int(public_opponent.get("hand_count", len(opponent.hand)) or 0),
        deck_count=int(public_opponent.get("deck_count", len(opponent.deck)) or 0),
        discard_count=int(public_opponent.get("discard_count", len(opponent.discard)) or 0),
        exile_count=int(public_opponent.get("exile_count", len(opponent.exile)) or 0),
    )
    revealed = _merge_revealed_cards(
        public_opponent.get("revealed_hand"),
        public_opponent.get("revealed_tag_cards"),
    )
    if revealed:
        opponent_view["revealed_hand"] = revealed
    for zone in ("deck", "discard"):
        ordered = public_opponent.get(f"{zone}_ordered")
        if blind_level < 3 and isinstance(ordered, list):
            opponent_view[f"{zone}_ordered"] = [
                {"slot": slot, **_card_dict_view(card)}
                for slot, card in enumerate(ordered)
            ]
    if blind_level >= 3:
        for key in ("health", "elixir", "magic", "hand_count", "deck_count", "discard_count", "exile_count"):
            opponent_view[key] = None
        opponent_view.pop("revealed_hand", None)
        opponent_view.pop("deck_ordered", None)
        opponent_view.pop("discard_ordered", None)

    history = list(env.public_history)
    if blind_level >= 2:
        history = [{"masked": True, "event_count": len(history)}]

    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "ruleset": "formal_1v1",
        "seat": viewer_id,
        "loadout": {
            "official_mods": list(env.mod_filenames),
            "fingerprint": env.loadout_fingerprint,
            "ruleset_fingerprint": env.ruleset_fingerprint,
        },
        "phase": str(engine.phase),
        "round": int(engine.round_num),
        "current_player": int(engine.current_player),
        "first_player": int(engine.first_player),
        "opening_events": list(engine.opening_event_picks),
        "decision_player": env.decision_player(),
        "game_over": bool(engine.game_over),
        "winner": int(engine.winner),
        "self": own_view,
        "opponent": opponent_view,
        "pending": _pending_view(env, viewer_id, public),
        "public_history": history,
        "temporary_reveals": _temporary_reveals(public, blind_level),
    }


def _build_pregame_observation(env, viewer_id: int) -> dict[str, Any]:
    engine = env.engine
    opponent_id = 1 - viewer_id
    status = env.pregame_status(viewer_id)
    own_event = engine.opening_event_picks[viewer_id]
    opponent_event = engine.opening_event_picks[opponent_id]
    all_events_selected = all(pick is not None for pick in engine.opening_event_picks)
    own: dict[str, Any] = {
        "status": status,
        "event_selected": own_event is not None,
        "draft_started": bool(engine.player_draft_started[viewer_id]),
        "draft_count": len(engine.draft_picks[viewer_id]),
        "draft_target": int(engine.draft_target_count(viewer_id)),
        "rerolls": int(engine.draft_rerolls[viewer_id]),
        "ready": bool(engine.player_ready[viewer_id]),
    }
    opponent = {
        "status": env.pregame_status(opponent_id),
        "event_selected": opponent_event is not None,
        "draft_started": bool(engine.player_draft_started[opponent_id]),
        "draft_count": len(engine.draft_picks[opponent_id]),
        "ready": bool(engine.player_ready[opponent_id]),
    }
    if own_event is not None:
        own["opening_event"] = _opening_event_view(
            next((
                option for option in engine.opening_event_options[viewer_id]
                if option and str(option.get("id")) == str(own_event)
            ), {"id": own_event})
        )
    if all_events_selected and opponent_event is not None:
        opponent["draft_target"] = int(engine.draft_target_count(opponent_id))
        opponent["opening_event"] = _opening_event_view(
            next((
                option for option in engine.opening_event_options[opponent_id]
                if option and str(option.get("id")) == str(opponent_event)
            ), {"id": opponent_event})
        )
    if status == "event_select":
        own["opening_event_options"] = [
            {"slot": slot, **_opening_event_view(option)}
            for slot, option in enumerate(engine.opening_event_options[viewer_id] or [])
            if option
        ]
    if engine.player_draft_started[viewer_id]:
        own["draft_picks"] = [
            {"slot": slot, **_card_def_view(def_id)}
            for slot, def_id in enumerate(engine.draft_picks[viewer_id])
        ]
    if status == "drafting":
        draft_index = len(engine.draft_picks[viewer_id])
        if draft_index < len(engine.draft_type_order):
            own["draft_card_type"] = str(engine.draft_type_order[draft_index])
        own["draft_options"] = [
            {"slot": slot, **_card_view(card)}
            for slot, card in enumerate(engine.draft_options[viewer_id] or [])
            if card is not None
        ]
    if status == "sub_choice":
        own["sub_choice"] = {
            "event_id": str(own_event),
            "selection": env.pregame_selection_view(viewer_id),
            "candidates": [
                {"slot": slot, **_card_def_view(def_id)}
                for slot, def_id in enumerate(env.pregame_candidate_def_ids(viewer_id))
            ],
        }
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "ruleset": "formal_1v1",
        "seat": viewer_id,
        "loadout": {
            "official_mods": list(env.mod_filenames),
            "fingerprint": env.loadout_fingerprint,
            "ruleset_fingerprint": env.ruleset_fingerprint,
        },
        "phase": "pregame",
        "pregame_phase": status,
        "round": 0,
        "current_player": None,
        "first_player": None,
        "decision_player": env.decision_player(),
        "game_over": False,
        "winner": -1,
        "self": own,
        "opponent": opponent,
        "pending": None,
        "public_history": list(env.public_history),
        "temporary_reveals": [],
    }


def _opening_event_view(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(event.get("id") or ""),
        "name": str(event.get("name") or event.get("name_cn") or ""),
        "description": str(event.get("desc") or event.get("description") or ""),
        "position": _as_int(event.get("position"), 0),
    }


def _card_def_view(def_id: Any) -> dict[str, Any]:
    try:
        from cards import CARD_DEFS, normalize_card_flags

        card_def = CARD_DEFS[str(def_id)]
        return {
            "def_id": str(card_def.id),
            "card_type": str(card_def.card_type),
            "cost_e": int(card_def.cost_e),
            "cost_m": int(card_def.cost_m),
            "cost_e_override": None,
            "cost_m_override": None,
            "mimic_discount": 0,
            "bonus_damage": 0,
            "swift": int(card_def.swift_value or 0),
            "magic_swift": int(card_def.magic_swift_value or 0),
            "temporary_swift": 0,
            "temporary_heavy": 0,
            "temporary_magic_heavy": 0,
            "flags": sorted(str(flag) for flag in normalize_card_flags(card_def.flags)),
            "disabled_flags": [],
            "setup_modifiers": [],
            "fission_level": int(card_def.fission_level or 1),
            "fusion_level": int(card_def.fusion_level or 1),
            "power": 0,
            "charge": int(card_def.charge_value or 0),
            "held_turns": 0,
            "return_to_hand_turns": 0,
            "extra_hits": 0,
            "base_damage": int(card_def.damage or 0),
            "base_hits": int(card_def.hits or 1),
            "quality": str(card_def.quality or ""),
        }
    except Exception:
        return {"def_id": str(def_id or "")}


def _player_view(player, *, include_cards: bool) -> dict[str, Any]:
    view: dict[str, Any] = {
        "player_id": int(player.player_id),
        "health": int(player.health),
        "max_health": int(player.max_health),
        "elixir": int(player.elixir),
        "max_elixir": int(player.max_elixir),
        "magic": int(player.magic),
        "max_magic": int(player.max_magic),
        "armor": int(player.armor),
        "hand_limit": int(player.hand_limit()),
        "equipment": [
            _equipment_view(item, slot=slot)
            for slot, item in enumerate(player.equipment)
        ],
        "statuses": {},
    }
    statuses = view["statuses"]
    for field in PLAYER_STATUS_FIELDS:
        value = getattr(player, field, 0)
        if isinstance(value, bool):
            if value:
                statuses[field] = True
        else:
            try:
                number = int(value or 0)
            except (TypeError, ValueError):
                continue
            if number:
                statuses[field] = number
    for status_id, value in sorted((getattr(player, "custom_statuses", {}) or {}).items()):
        if _is_public_scalar(value) and value:
            statuses[str(status_id)] = _safe_scalar(value)
    public_player_vars = getattr(player, "custom_vars", {}) or {}
    crit_multiplier = public_player_vars.get("hel_crit_multiplier")
    if _is_public_scalar(crit_multiplier) and crit_multiplier not in (None, 0, 1.5):
        view["critical_multiplier"] = _safe_scalar(crit_multiplier)
    if not include_cards:
        view.update(
            hand_count=len(player.hand),
            deck_count=len(player.deck),
            discard_count=len(player.discard),
            exile_count=len(player.exile),
        )
    return view


def _card_view(card) -> dict[str, Any]:
    result: dict[str, Any] = {
        "def_id": str(card.def_id),
        "card_type": str(card.card_type),
        "cost_e": int(card.cost_e),
        "cost_m": int(card.cost_m),
        "cost_e_override": _optional_int(getattr(card, "cost_e_override", None)),
        "cost_m_override": _optional_int(getattr(card, "cost_m_override", None)),
        "mimic_discount": int(getattr(card, "mimic_discount", 0) or 0),
        "bonus_damage": int(getattr(card, "bonus_damage", 0) or 0),
        "swift": int(getattr(card, "swift_value", 0) or 0),
        "magic_swift": int(getattr(card, "magic_swift_value", 0) or 0),
        "temporary_swift": int(getattr(card, "temp_swift_value", 0) or 0),
        "temporary_heavy": int(getattr(card, "temp_heavy_value", 0) or 0),
        "temporary_magic_heavy": int(getattr(card, "temp_magic_heavy_value", 0) or 0),
        "flags": sorted(str(flag) for flag in card.flags),
        "disabled_flags": sorted(str(flag) for flag in (getattr(card, "disabled_flags", set()) or set())),
        "setup_modifiers": sorted(str(value) for value in (getattr(card, "setup_modifiers", set()) or set())),
        "fission_level": int(getattr(card, "fission_level", 1) or 1),
        "fusion_level": int(getattr(card, "fusion_level", 1) or 1),
        "power": int(getattr(card, "power_value", 0) or 0),
        "charge": int(getattr(card, "charge_value", 0) or 0),
        "held_turns": int(getattr(card, "held_turns", 0) or 0),
        "return_to_hand_turns": int(getattr(card, "return_to_hand_turns", 0) or 0),
        "extra_hits": int(getattr(card, "extra_hits", 0) or 0),
        "base_damage": int(getattr(card.card_def, "damage", 0) or 0),
        "base_hits": int(getattr(card.card_def, "hits", 1) or 1),
        "quality": str(getattr(card.card_def, "quality", "") or ""),
    }
    visible_custom = _visible_custom_vars(
        getattr(card, "custom_vars", {}) or {},
        allowed=PUBLIC_CARD_CUSTOM_FIELDS,
    )
    if visible_custom:
        result["custom"] = visible_custom
    return result


def _card_dict_view(card: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    try:
        from cards import CARD_DEFS
        card_def = CARD_DEFS.get(str(card.get("def_id") or ""))
    except Exception:
        card_def = None
    result = {
        "def_id": str(card.get("def_id") or ""),
        "card_type": str(getattr(card_def, "card_type", "") or ""),
        "cost_e_override": card.get("cost_e_override"),
        "cost_m_override": card.get("cost_m_override"),
        "mimic_discount": _as_int(card.get("mimic_discount"), 0),
        "bonus_damage": _as_int(card.get("bonus_damage"), 0),
        "swift": _as_int(card.get("swift_value"), 0),
        "magic_swift": _as_int(card.get("magic_swift_value"), 0),
        "temporary_swift": _as_int(card.get("temp_swift_value"), 0),
        "temporary_heavy": _as_int(card.get("temp_heavy_value"), 0),
        "temporary_magic_heavy": _as_int(card.get("temp_magic_heavy_value"), 0),
        "fission_level": _as_int(card.get("fission_level"), 1),
        "fusion_level": _as_int(card.get("fusion_level"), 1),
        "power": _as_int(card.get("power_value"), 0),
        "charge": _as_int(card.get("charge_value"), 0),
        "flags": sorted(str(flag) for flag in (card.get("instance_flags") or [])),
        "disabled_flags": sorted(str(flag) for flag in (card.get("disabled_flags") or [])),
        "setup_modifiers": sorted(str(value) for value in (card.get("setup_modifiers") or [])),
    }
    custom = _visible_custom_vars(
        card.get("custom_vars") or {},
        allowed=PUBLIC_CARD_CUSTOM_FIELDS,
    )
    if custom:
        result["custom"] = custom
    return result


def _equipment_view(equipment, *, slot: int) -> dict[str, Any]:
    result = {
        "slot": int(slot),
        "card": _card_view(equipment.card_instance),
        "owner": int(equipment.owner),
        "effect_target": int(equipment.effect_target),
        "turns_equipped": int(equipment.turns_equipped),
        "uses_this_turn": int(equipment.uses_this_turn),
        "armor": int(equipment.armor),
        "corruption_active": bool(equipment.corruption_active),
    }
    counters = _visible_custom_vars(
        getattr(equipment, "custom_vars", {}) or {},
        allowed=PUBLIC_EQUIPMENT_CUSTOM_FIELDS,
    )
    if counters:
        result["counters"] = counters
    return result


def _zone_multiset(cards: Iterable) -> list[dict[str, Any]]:
    counter = Counter(_card_identity(card) for card in cards)
    return [
        {"card": dict(identity), "count": count}
        for identity, count in sorted(counter.items(), key=lambda item: _identity_sort_key(item[0]))
    ]


def _identity_sort_key(identity: tuple[tuple[str, Any], ...]) -> str:
    return json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _card_identity(card) -> tuple[tuple[str, Any], ...]:
    view = _card_view(card)
    view.pop("custom", None)
    return tuple(sorted((key, _freeze_identity(value)) for key, value in view.items()))


def _freeze_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_identity(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_identity(item) for item in value)
    return value


def _merge_revealed_cards(*collections) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cards in collections:
        if not isinstance(cards, list):
            continue
        for card in cards:
            if not isinstance(card, dict):
                continue
            instance_id = _as_int(card.get("instance_id"), -1)
            view = _card_dict_view(card)
            if instance_id >= 0 and instance_id not in seen:
                seen.add(instance_id)
                merged.append(view)
    return [
        {"reveal_slot": slot, **view}
        for slot, view in enumerate(merged)
    ]


def _pending_view(env, viewer_id: int, public: dict[str, Any]) -> dict[str, Any] | None:
    engine = env.engine
    blind_level = max(0, int(getattr(engine.players[viewer_id], "blind", 0) or 0))
    if engine.pending_response is not None:
        pending = engine.pending_response
        return {
            "kind": "response",
            "actor": 1 - int(pending.get("player_id", 0)),
            "source_player": _as_int(pending.get("player_id"), -1),
            "target_player": _as_int(pending.get("target_player_id"), -1),
            "card": _mask_card_dict(_card_dict_view(pending.get("card") or {}), blind_level),
        }
    if engine.pending_choice is not None:
        pending = engine.pending_choice
        actor = _as_int(pending.get("player_id"), -1)
        result: dict[str, Any] = {
            "kind": "choice",
            "actor": actor,
            "choice_type": str(pending.get("choice_type") or ""),
        }
        if actor == viewer_id:
            result["selection"] = env.choice_state_view()
            candidates = []
            for slot, card in enumerate(env.choice_candidate_cards()):
                if card is not None:
                    candidates.append({
                        "slot": slot,
                        **_visible_choice_card(card, viewer_id, engine, blind_level),
                    })
            if candidates:
                result["candidates"] = candidates
            params = pending.get("choice_params") or {}
            result["constraints"] = {
                key: _safe_scalar(value)
                for key, value in params.items()
                if key in {"min_count", "max_count", "count", "cancellable", "target", "allowed"}
                and _is_public_scalar(value)
            }
        else:
            result["private"] = True
        return result
    if engine.pending_v2_ui is not None:
        private_pending = engine.pending_v2_ui
        actor = _as_int(private_pending.get("player_id"), -1)
        result: dict[str, Any] = {
            "kind": "mod_ui",
            "actor": actor,
        }
        if actor != viewer_id:
            result["private"] = True
            return result
        public_pending = public.get("pending_v2_ui") or {}
        component = public_pending.get("component") if isinstance(public_pending, dict) else None
        if not isinstance(component, dict):
            component = private_pending.get("component")
        if not isinstance(component, dict):
            result["component_type"] = "unknown"
            return result
        result["component_type"] = str(component.get("type") or "modal")
        result["buttons"] = [
            {
                "slot": slot,
                "role": str(button.get("role") or (
                    "cancel" if str(button.get("id")) == "cancel" else "confirm"
                )),
            }
            for slot, button in enumerate(component.get("buttons") or [])
            if isinstance(button, dict)
        ]
        controls = []
        candidates = []
        for control_slot, control in enumerate(component.get("controls") or []):
            if not isinstance(control, dict):
                continue
            ctype = str(control.get("type") or "text")
            summary: dict[str, Any] = {
                "slot": int(control_slot),
                "type": ctype,
            }
            if ctype in {"slider", "number", "number_input"}:
                for key in ("min", "max", "step", "default"):
                    if _is_public_scalar(control.get(key)):
                        summary[key] = _safe_scalar(control.get(key))
            options = [
                option for option in (control.get("options") or [])
                if isinstance(option, dict)
            ]
            if options:
                summary["option_count"] = len(options)
            for key in ("min_select", "max_select"):
                if _is_public_scalar(control.get(key)):
                    summary[key] = _safe_scalar(control.get(key))
            controls.append(summary)
            for option_slot, option in enumerate(options):
                candidate: dict[str, Any] = {
                    "slot": len(candidates),
                    "control_slot": int(control_slot),
                    "option_slot": int(option_slot),
                }
                card = option.get("card")
                if isinstance(card, dict):
                    candidate.update(_mask_card_dict(_card_dict_view(card), blind_level))
                elif ctype in {"player_picker", "target_picker"}:
                    player_id = _as_int(option.get("value"), -1)
                    if player_id in (0, 1):
                        candidate["target_player"] = player_id
                else:
                    semantic = option.get("value")
                    if isinstance(semantic, (str, bool)) and len(str(semantic)) <= 64:
                        candidate["semantic"] = semantic
                candidates.append(candidate)
        result["controls"] = controls
        if candidates:
            result["candidates"] = candidates
        return result
    return None


def _visible_custom_vars(values: dict[str, Any], *, allowed: frozenset[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(values.items()):
        key_text = str(key)
        if key_text not in allowed:
            continue
        if _is_public_scalar(value):
            result[key_text] = _safe_scalar(value)
    return result


def _temporary_reveals(public: dict[str, Any], blind_level: int) -> list[dict[str, Any]]:
    if blind_level >= 3:
        return []
    reveal = public.get("garden_initial_deck_reveal")
    if not isinstance(reveal, dict) or not isinstance(reveal.get("cards"), list):
        return []
    return [{
        "kind": "initial_deck",
        "token": _as_int(reveal.get("token"), 0),
        "target_player": _as_int(reveal.get("target_player_id"), -1),
        "cards": [
            {"slot": slot, **_card_dict_view(card)}
            for slot, card in enumerate(reveal["cards"])
            if isinstance(card, dict)
        ],
    }]


def _visible_own_hand_card(card, global_blind_level: int) -> dict[str, Any]:
    card_blind = max(0, int(getattr(card, "hand_blind_turns", 0) or 0))
    level = max(global_blind_level, 1 if card_blind > 0 else 0)
    if level <= 0:
        return _card_view(card)
    result: dict[str, Any] = {"unknown": True}
    if level < 2:
        result["card_type"] = str(card.card_type)
    return result


def _visible_choice_card(card, viewer_id: int, engine, blind_level: int) -> dict[str, Any]:
    owner_id, zone_name, _ = engine._find_card_location(card)
    if owner_id == viewer_id and zone_name == "hand":
        return _visible_own_hand_card(card, blind_level)
    if blind_level > 0:
        return _mask_card_dict(_card_view(card), blind_level)
    return _card_view(card)


def _mask_card_dict(card: dict[str, Any], blind_level: int) -> dict[str, Any]:
    if blind_level <= 0:
        return card
    result: dict[str, Any] = {"unknown": True}
    if blind_level < 2 and card.get("card_type"):
        result["card_type"] = card["card_type"]
    return result


def _is_public_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _as_int(value, 0)
