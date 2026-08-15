from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable, Iterator


CONTEXTUAL_VALUE_SCHEMA_VERSION = 1

# Text and asset fields describe presentation, not rules. Keeping them out also
# prevents language selection from changing the model input.
_PRESENTATION_KEYS = frozenset({
    "assets",
    "description",
    "description_cn",
    "description_en",
    "description_i18n",
    "effect_text",
    "effect_text_cn",
    "effect_text_en",
    "effect_text_i18n",
    "image",
    "image_url",
    "name",
    "name_cn",
    "name_en",
    "name_i18n",
    "response_content",
    "response_content_i18n",
    "response_title",
    "response_title_i18n",
    "trigger_effect_text",
    "trigger_effect_text_i18n",
    "ui_effect_size",
    "upgraded_image",
    "upgraded_image_url",
})

_RULE_OPERATION_KEYS = frozenset({"op", "operation", "action", "kind"})
_NUMERIC_RULE_KEYS = frozenset({
    "amount",
    "armor",
    "burn",
    "count",
    "damage",
    "dodge",
    "draw",
    "gain_e",
    "gain_m",
    "heal",
    "hits",
    "layers",
    "level",
    "max_count",
    "min_count",
    "poison",
    "power",
    "trigger_cost_e",
    "trigger_cost_m",
    "value",
})


def contextual_value_tokens(observation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Build public-information-only context tokens for dynamic card valuation.

    The returned values intentionally contain no scalar card score. They expose
    resource supply and demand, rule operations, card-instance mutations, and
    deck composition separately so a policy can learn different weights for
    different loadouts and game states.
    """

    own = observation.get("self") or {}
    opponent = observation.get("opponent") or {}
    phase = str(observation.get("phase") or "")
    tokens: list[tuple[str, dict[str, Any]]] = [
        (
            "value_context:resources",
            {
                "schema_version": CONTEXTUAL_VALUE_SCHEMA_VERSION,
                "phase": phase,
                "self": _player_resource_context(own),
                "opponent": _player_resource_context(opponent),
            },
        ),
    ]

    own_hand = list(_iter_zone_cards(own, ("hand",)))
    own_pool = list(_iter_zone_cards(own, ("hand", "deck", "discard", "exile")))
    own_equipment = list(_iter_equipment_cards(own))
    opponent_public = list(_iter_zone_cards(
        opponent,
        ("revealed_hand", "deck_ordered", "discard_ordered"),
    ))
    opponent_equipment = list(_iter_equipment_cards(opponent))

    tokens.extend((
        (
            "value_context:self_hand",
            _summarize_cards(own_hand, include_rule_aggregates=True),
        ),
        (
            "value_context:self_pool",
            _summarize_cards(own_pool, include_rule_aggregates=True),
        ),
        (
            "value_context:equipment",
            {
                "self": _summarize_cards(
                    own_equipment,
                    include_rule_aggregates=True,
                ),
                "opponent": _summarize_cards(
                    opponent_equipment,
                    include_rule_aggregates=True,
                ),
            },
        ),
    ))

    if opponent_public:
        tokens.append((
            "value_context:opponent_public",
            _summarize_cards(opponent_public, include_rule_aggregates=True),
        ))

    if phase == "pregame":
        draft_picks = list(_iter_plain_cards(own.get("draft_picks") or (), "draft_pick"))
        draft_options = list(_iter_plain_cards(own.get("draft_options") or (), "draft_option"))
        sub_choice = own.get("sub_choice") or {}
        candidates = list(_iter_plain_cards(
            sub_choice.get("candidates") or (),
            "pregame_candidate",
        ))
        tokens.append((
            "value_context:draft",
            {
                "picks": _summarize_cards(draft_picks, include_rule_aggregates=True),
                "options": _summarize_cards(draft_options, include_rule_aggregates=True),
                "candidates": _summarize_cards(candidates, include_rule_aggregates=True),
                "target": _safe_int(own.get("draft_target")),
                "picked": _safe_int(own.get("draft_count")),
            },
        ))

    return tokens


def enrich_card_for_context(card: dict[str, Any]) -> dict[str, Any]:
    """Attach rule semantics and current instance deltas to a visible card."""

    value = dict(card or {})
    definition = card_definition_semantics(str(value.get("def_id") or ""))
    if not definition:
        return value
    observed_flags = frozenset(str(flag) for flag in (value.get("flags") or ()))
    base_flags = frozenset(str(flag) for flag in (definition.get("base_flags") or ()))
    disabled_flags = frozenset(str(flag) for flag in (value.get("disabled_flags") or ()))
    effective_flags = (base_flags | observed_flags) - disabled_flags
    effective_cost_e = _effective_cost(value, definition, resource="e")
    effective_cost_m = _effective_cost(value, definition, resource="m")
    value["card_type"] = str(value.get("card_type") or definition.get("card_type") or "")
    value["cost_e"] = effective_cost_e
    value["cost_m"] = effective_cost_m
    value["flags"] = sorted(effective_flags)
    value["definition_semantics"] = definition
    value["instance_semantics"] = {
        "added_flags": sorted(effective_flags - base_flags),
        "removed_flags": sorted((base_flags - effective_flags) | disabled_flags),
        "cost_delta_e": effective_cost_e - _safe_int(definition.get("base_cost_e")),
        "cost_delta_m": effective_cost_m - _safe_int(definition.get("base_cost_m")),
        "fission_delta": _safe_int(value.get("fission_level"), 1) - _safe_int(
            definition.get("base_fission_level"), 1
        ),
        "fusion_delta": _safe_int(value.get("fusion_level"), 1) - _safe_int(
            definition.get("base_fusion_level"), 1
        ),
        "power": _safe_int(value.get("power")),
        "charge": _safe_int(value.get("charge")),
        "bonus_damage": _safe_int(value.get("bonus_damage")),
        "extra_hits": _safe_int(value.get("extra_hits")),
        "setup_modifiers": sorted(str(item) for item in (value.get("setup_modifiers") or ())),
    }
    return value


@lru_cache(maxsize=2048)
def card_definition_semantics(def_id: str) -> dict[str, Any]:
    """Return language-independent rule metadata for an official card."""

    if not def_id:
        return {}
    try:
        from cards import CARD_DEFS, normalize_card_flags

        card_def = CARD_DEFS.get(str(def_id))
    except (ImportError, AttributeError):
        card_def = None
        normalize_card_flags = lambda values: set(values or ())  # type: ignore[assignment]
    if card_def is None:
        return {"def_id": str(def_id)}

    operations: Counter[str] = Counter()
    numeric_rules: Counter[str] = Counter()
    events = getattr(card_def, "v2_events", {}) or {}
    _collect_rule_tree(events, operations, numeric_rules, path="events")
    legacy_effects = getattr(card_def, "effects", ()) or ()
    legacy_scripts = getattr(card_def, "scripts", {}) or {}
    _collect_rule_tree(legacy_effects, operations, numeric_rules, path="effects")
    _collect_rule_tree(legacy_scripts, operations, numeric_rules, path="scripts")

    resource = getattr(card_def, "v2_resource", {}) or {}
    if isinstance(resource, dict):
        for key in sorted(_NUMERIC_RULE_KEYS):
            value = resource.get(key)
            if _is_number(value) and float(value) != 0.0:
                numeric_rules[f"resource.{key}"] += float(value)

    return {
        "def_id": str(card_def.id),
        "mod_id": str(getattr(card_def, "v2_mod_id", "") or ""),
        "card_type": str(getattr(card_def, "card_type", "") or ""),
        "quality": str(getattr(card_def, "quality", "") or ""),
        "base_cost_e": _safe_int(getattr(card_def, "cost_e", 0)),
        "base_cost_m": _safe_int(getattr(card_def, "cost_m", 0)),
        "trigger_cost_e": max(0, _safe_int(getattr(card_def, "trigger_cost_e", 0))),
        "trigger_cost_m": max(0, _safe_int(getattr(card_def, "trigger_cost_m", 0))),
        "base_damage": _safe_int(getattr(card_def, "damage", 0)),
        "base_hits": max(1, _safe_int(getattr(card_def, "hits", 1), 1)),
        "base_fission_level": max(1, _safe_int(getattr(card_def, "fission_level", 1), 1)),
        "base_fusion_level": max(1, _safe_int(getattr(card_def, "fusion_level", 1), 1)),
        "copy_count": max(0, _safe_int(getattr(card_def, "copy_count", 0))),
        "base_swift": max(0, _safe_int(getattr(card_def, "swift_value", 0))),
        "base_magic_swift": max(0, _safe_int(getattr(card_def, "magic_swift_value", 0))),
        "base_flags": sorted(str(flag) for flag in normalize_card_flags(card_def.flags)),
        "response_trigger": str(getattr(card_def, "response_trigger", "") or ""),
        "events": sorted(str(key) for key in events) if isinstance(events, dict) else [],
        "rule_operations": _compact_counter(operations),
        "rule_numbers": _compact_counter(numeric_rules),
    }


def transition_value_components(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Extract learnable outcome components from two public observations.

    This deliberately does not combine the components into a reward. Training
    can learn context-dependent weights, while diagnostics can inspect exactly
    why an action was considered useful or harmful.
    """

    return {
        "self": _player_transition_components(before.get("self") or {}, after.get("self") or {}),
        "opponent": _player_transition_components(
            before.get("opponent") or {},
            after.get("opponent") or {},
        ),
        "round_delta": _delta(before, after, "round"),
        "turn_changed": before.get("current_player") != after.get("current_player"),
        "pending_before": _pending_kind(before),
        "pending_after": _pending_kind(after),
        "terminal": bool(after.get("game_over")),
        "winner": _safe_int(after.get("winner"), -1),
    }


def _player_transition_components(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "health",
        "max_health",
        "elixir",
        "max_elixir",
        "magic",
        "max_magic",
        "armor",
        "hand_count",
        "deck_count",
        "discard_count",
        "exile_count",
    )
    status_before = before.get("statuses") or {}
    status_after = after.get("statuses") or {}
    status_ids = sorted(set(status_before) | set(status_after))
    return {
        "deltas": {
            field: _delta(before, after, field)
            for field in fields
            if before.get(field) is not None and after.get(field) is not None
        },
        "status_deltas": {
            status_id: _safe_number(status_after.get(status_id))
            - _safe_number(status_before.get(status_id))
            for status_id in status_ids
            if _safe_number(status_after.get(status_id))
            != _safe_number(status_before.get(status_id))
        },
        "equipment_delta": len(after.get("equipment") or ()) - len(before.get("equipment") or ()),
    }


def _player_resource_context(player: dict[str, Any]) -> dict[str, Any]:
    hand_count = player.get("hand_count")
    if hand_count is None and isinstance(player.get("hand"), list):
        hand_count = len(player["hand"])
    hand_limit = player.get("hand_limit")
    free_slots = None
    if hand_count is not None and hand_limit is not None:
        free_slots = _safe_int(hand_limit) - _safe_int(hand_count)
    return {
        "health": player.get("health"),
        "max_health": player.get("max_health"),
        "elixir": player.get("elixir"),
        "max_elixir": player.get("max_elixir"),
        "magic": player.get("magic"),
        "max_magic": player.get("max_magic"),
        "armor": player.get("armor"),
        "hand_count": hand_count,
        "hand_limit": hand_limit,
        "hand_free_slots": free_slots,
        "status_count": len(player.get("statuses") or {}),
        "equipment_count": len(player.get("equipment") or ()),
    }


def _summarize_cards(
    cards: Iterable[tuple[dict[str, Any], int, str]],
    *,
    include_rule_aggregates: bool,
) -> dict[str, Any]:
    total = 0
    definitions: set[str] = set()
    variants: set[str] = set()
    types: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    mods: Counter[str] = Counter()
    zones: Counter[str] = Counter()
    costs_e: Counter[str] = Counter()
    costs_m: Counter[str] = Counter()
    rule_operations: Counter[str] = Counter()
    trigger_cost_e = 0
    trigger_cost_m = 0
    trigger_cards_e = 0
    trigger_cards_m = 0
    layers = Counter()

    for card, raw_count, zone in cards:
        count = max(0, _safe_int(raw_count, 1))
        if not card or count <= 0:
            continue
        enriched = enrich_card_for_context(card)
        definition = enriched.get("definition_semantics") or {}
        instance = enriched.get("instance_semantics") or {}
        total += count
        def_id = str(enriched.get("def_id") or "")
        if def_id:
            definitions.add(def_id)
        variants.add(_variant_fingerprint(enriched))
        types[str(enriched.get("card_type") or definition.get("card_type") or "unknown")] += count
        zones[str(zone)] += count
        for flag in enriched.get("flags") or ():
            flags[str(flag)] += count
        mod_id = str(definition.get("mod_id") or "")
        if mod_id:
            mods[mod_id] += count
        costs_e[str(max(0, _safe_int(enriched.get("cost_e"))))] += count
        costs_m[str(max(0, _safe_int(enriched.get("cost_m"))))] += count
        for field in ("fission_level", "fusion_level", "power", "charge", "extra_hits"):
            value = _safe_int(enriched.get(field))
            if value:
                layers[field] += value * count
        for field in ("added_flags", "removed_flags"):
            for flag in instance.get(field) or ():
                flags[f"instance_{field}:{flag}"] += count

        cost_e = max(0, _safe_int(definition.get("trigger_cost_e")))
        cost_m = max(0, _safe_int(definition.get("trigger_cost_m")))
        if cost_e:
            trigger_cost_e += cost_e * count
            trigger_cards_e += count
        if cost_m:
            trigger_cost_m += cost_m * count
            trigger_cards_m += count
        if include_rule_aggregates:
            for operation, operation_count in (definition.get("rule_operations") or {}).items():
                rule_operations[str(operation)] += _safe_number(operation_count) * count

    return {
        "card_count": total,
        "definition_count": len(definitions),
        "instance_variant_count": len(variants),
        "card_types": _compact_counter(types),
        "flags": _compact_counter(flags),
        "mods": _compact_counter(mods),
        "zones": _compact_counter(zones),
        "cost_e_histogram": _compact_counter(costs_e),
        "cost_m_histogram": _compact_counter(costs_m),
        "trigger_demand": {
            "elixir": trigger_cost_e,
            "magic": trigger_cost_m,
            "elixir_cards": trigger_cards_e,
            "magic_cards": trigger_cards_m,
        },
        "layers": _compact_counter(layers),
        "rule_operations": _compact_counter(rule_operations),
    }


def _iter_zone_cards(
    player: dict[str, Any],
    zones: Iterable[str],
) -> Iterator[tuple[dict[str, Any], int, str]]:
    for zone in zones:
        ordered_key = f"{zone}_ordered"
        if isinstance(player.get(ordered_key), list):
            values = player.get(ordered_key) or []
            actual_zone = ordered_key
        else:
            values = player.get(zone) or []
            actual_zone = zone
        yield from _iter_plain_cards(values, actual_zone)


def _iter_plain_cards(
    values: Iterable[Any],
    zone: str,
) -> Iterator[tuple[dict[str, Any], int, str]]:
    for raw in values:
        if not isinstance(raw, dict):
            continue
        card = raw.get("card") if isinstance(raw.get("card"), dict) else raw
        if not card.get("def_id"):
            continue
        yield card, max(1, _safe_int(raw.get("count"), 1)), zone


def _iter_equipment_cards(player: dict[str, Any]) -> Iterator[tuple[dict[str, Any], int, str]]:
    for equipment in player.get("equipment") or ():
        if not isinstance(equipment, dict):
            continue
        card = equipment.get("card")
        if isinstance(card, dict) and card.get("def_id"):
            yield card, 1, "equipment"


def _collect_rule_tree(
    value: Any,
    operations: Counter[str],
    numeric_rules: Counter[str],
    *,
    path: str,
    depth: int = 0,
) -> None:
    if depth > 12:
        return
    if isinstance(value, dict):
        operation = next((
            value.get(key)
            for key in _RULE_OPERATION_KEYS
            if isinstance(value.get(key), str)
        ), None)
        if operation:
            operations[str(operation)] += 1
        for key, item in value.items():
            if str(key) in _PRESENTATION_KEYS:
                continue
            child_path = f"{path}.{key}"
            if _is_number(item):
                numeric_rules[
                    f"{operation or path.rsplit('.', 1)[-1]}.{key}"
                ] += float(item)
            else:
                _collect_rule_tree(
                    item,
                    operations,
                    numeric_rules,
                    path=child_path,
                    depth=depth + 1,
                )
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_rule_tree(
                item,
                operations,
                numeric_rules,
                path=path,
                depth=depth + 1,
            )


def _variant_fingerprint(card: dict[str, Any]) -> str:
    value = {
        key: card.get(key)
        for key in (
            "def_id",
            "cost_e",
            "cost_m",
            "bonus_damage",
            "swift",
            "magic_swift",
            "temporary_swift",
            "temporary_heavy",
            "temporary_magic_heavy",
            "flags",
            "disabled_flags",
            "setup_modifiers",
            "fission_level",
            "fusion_level",
            "power",
            "charge",
            "extra_hits",
        )
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _effective_cost(
    card: dict[str, Any],
    definition: dict[str, Any],
    *,
    resource: str,
) -> int:
    direct = card.get(f"cost_{resource}")
    if direct is not None:
        return max(0, _safe_int(direct))
    override = card.get(f"cost_{resource}_override")
    base = (
        _safe_int(override)
        if override is not None
        else _safe_int(definition.get(f"base_cost_{resource}"))
    )
    if resource == "e":
        observed_swift = max(0, _safe_int(card.get("swift")))
        swift = observed_swift or max(0, _safe_int(definition.get("base_swift")))
        return max(
            0,
            base
            + max(0, _safe_int(card.get("temporary_heavy")))
            - max(0, _safe_int(card.get("mimic_discount")))
            - swift
            - max(0, _safe_int(card.get("temporary_swift"))),
        )
    observed_magic_swift = max(0, _safe_int(card.get("magic_swift")))
    magic_swift = observed_magic_swift or max(
        0, _safe_int(definition.get("base_magic_swift"))
    )
    return max(
        0,
        base
        + max(0, _safe_int(card.get("temporary_magic_heavy")))
        - magic_swift,
    )


def _compact_counter(counter: Counter) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
        number = float(value)
        result[str(key)] = int(number) if number.is_integer() else number
    return result


def _pending_kind(observation: dict[str, Any]) -> str:
    pending = observation.get("pending") or {}
    return str(pending.get("kind") or "")


def _delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int | float:
    return _safe_number(after.get(key)) - _safe_number(before.get(key))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_number(value: Any) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if _is_number(value):
        number = float(value)
        return int(number) if number.is_integer() else number
    return 0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
