from __future__ import annotations

import hashlib
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAME_ROOT = PROJECT_ROOT / "Python联机版"
_OFFICIAL_CATALOG_CACHE = {}
_OFFICIAL_LOADOUT_CACHE = OrderedDict()
_OFFICIAL_RULESET_FINGERPRINT_CACHE = {}
OFFICIAL_LOADOUT_CACHE_LIMIT = 64

# These modules define the parts of the production runtime exercised by formal
# 1v1. A change to any of them invalidates trained policies even when the mod
# archives themselves did not change.
RULESET_SOURCE_FILES = (
    "cards.py",
    "damage_types.py",
    "formal_logic_runtime.py",
    "game_engine.py",
    "mod_loader.py",
    "mod_loadout_v2.py",
    "mod_runtime_v2.py",
    "mod_spec_v2.py",
    "runtime_budget.py",
    "runtime_errors.py",
    "void_dlc_runtime.py",
)


def configure_game_imports(game_root: os.PathLike | str | None = None) -> Path:
    root = Path(game_root or os.environ.get("GTN_GAME_ROOT") or DEFAULT_GAME_ROOT).resolve()
    if not (root / "game_engine.py").is_file():
        raise FileNotFoundError(f"Garden of Thorn game engine was not found at {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def load_official_content(
    game_root: os.PathLike | str | None = None,
    enabled_mods: Iterable[str] | None = None,
):
    """Load only bundled official mods and return a deterministic loadout."""

    root = configure_game_imports(game_root)
    enabled_key = None if enabled_mods is None else tuple(sorted(str(name).casefold() for name in enabled_mods))
    cache_key = (str(root), enabled_key)
    cached = _OFFICIAL_LOADOUT_CACHE.get(cache_key)
    if cached is not None:
        _OFFICIAL_LOADOUT_CACHE.move_to_end(cache_key)
        return cached
    from card_i18n import apply_card_i18n_defaults
    from cards import (
        ALL_MOD_SHARED_CARD_IDS,
        CARD_DEFS,
        CardDef,
        ERROR_CARD_ID,
        normalize_card_flag,
        normalize_card_flags,
    )
    from mod_loader import load_all_mods, mod_category, sort_mods_for_display
    from mod_loadout_v2 import build_v2_loadout

    official_mods = _OFFICIAL_CATALOG_CACHE.get(str(root))
    if official_mods is None:
        official_mods = tuple(
            mod for mod in sort_mods_for_display(load_all_mods(force=True))
            if not mod.errors and mod_category(mod) == "official"
        )
        full_loadout = build_v2_loadout(official_mods)
        if not full_loadout.ok:
            raise RuntimeError("invalid official loadout: " + "; ".join(full_loadout.errors))

        resources = full_loadout.registries.get("cards", {}) or {}
        for card_id, resource in resources.items():
            runtime_id = str(resource.get("legacy_id") or resource.get("runtime_id") or card_id)
            cost = resource.get("cost") if isinstance(resource.get("cost"), dict) else {}
            flags = normalize_card_flags(resource.get("flags", []) or [])
            for tag in resource.get("tags", []) or []:
                tag_text = str(tag)
                if tag_text.startswith("gtn:"):
                    tag_text = tag_text.split(":", 1)[1]
                flags.add(normalize_card_flag(tag_text))

            card_type = {
                "attack": "thorn",
                "skill": "bloom",
                "equipment": "root",
                "counter": "guard",
            }.get(str(resource.get("card_type", resource.get("type", "bloom"))).lower(),
                  str(resource.get("card_type", resource.get("type", "bloom"))).lower())
            name_path = str(card_id).split(":")[-1]
            fallback_name = " ".join(part.capitalize() for part in name_path.replace("/", "_").split("_") if part)
            card_def = CardDef(
                id=runtime_id,
                name_en=str(resource.get("name_en") or resource.get("name") or fallback_name),
                name_cn=str(resource.get("name_cn") or resource.get("name") or fallback_name),
                cost_e=_as_int(resource.get("cost_e", cost.get("e", 0))),
                cost_m=_as_int(resource.get("cost_m", cost.get("m", 0))),
                card_type=card_type if card_type in {"thorn", "bloom", "root", "guard"} else "bloom",
                count=max(0, _as_int(resource.get("count", resource.get("weight", 3)), 3)),
                quality=str(resource.get("quality") or "Common"),
                description=str(resource.get("description") or resource.get("description_cn") or ""),
                effect_text=str(resource.get("effect_text") or resource.get("effect_text_cn") or ""),
                name_i18n=dict(resource.get("name_i18n") or {}),
                description_i18n=dict(resource.get("description_i18n") or {}),
                effect_text_i18n=dict(resource.get("effect_text_i18n") or {}),
                flags=flags,
                trigger_cost_e=_as_int(resource.get("trigger_cost_e", -1), -1),
                trigger_cost_m=_as_int(resource.get("trigger_cost_m", 0)),
                trigger_effect_text=str(resource.get("trigger_effect_text") or ""),
                trigger_effect_text_i18n=dict(resource.get("trigger_effect_text_i18n") or {}),
                response_trigger=str(resource.get("response_trigger") or ""),
                effects=[],
                scripts={},
                response_title=str(resource.get("response_title") or ""),
                response_content=str(resource.get("response_content") or ""),
                response_title_i18n=dict(resource.get("response_title_i18n") or {}),
                response_content_i18n=dict(resource.get("response_content_i18n") or {}),
                v2_events=resource.get("events") if isinstance(resource.get("events"), dict) else {},
                v2_resource=dict(resource),
                v2_mod_id=str(resource.get("_mod_id") or ""),
                image=str(resource.get("image") or ""),
                image_url=str(resource.get("image_url") or resource.get("image") or ""),
                upgraded_image=str(resource.get("upgraded_image") or ""),
                upgraded_image_url=str(resource.get("upgraded_image_url") or resource.get("upgraded_image") or ""),
                copy_count=_as_int(resource.get("copy_count", 0)),
                swift_value=_as_int(resource.get("swift_value", 0)),
                magic_swift_value=_as_int(resource.get("magic_swift_value", 0)),
                fission_level=max(1, _as_int(resource.get("fission_level", _as_int(resource.get("fission_count", 0)) + 1), 1)),
                fusion_level=max(1, _as_int(resource.get("fusion_level", resource.get("fusion_multiplier", 1)), 1)),
                damage=_as_int(resource.get("damage", 0)),
                hits=max(1, _as_int(resource.get("hits", 1), 1)),
                ui_effect_size=str(resource.get("ui_effect_size") or ""),
            )
            CARD_DEFS[runtime_id] = card_def
        apply_card_i18n_defaults(CARD_DEFS)
        _OFFICIAL_CATALOG_CACHE[str(root)] = official_mods

    selected_names = None if enabled_key is None else set(enabled_key)
    selected_mods = [
        mod for mod in official_mods
        if selected_names is None or mod.filename.casefold() in selected_names
    ]
    if not selected_mods:
        selected_mods = [mod for mod in official_mods if mod.filename == "Vanilla Cards.gtnmod"]

    selected_types = {
        str(card.card_type or "")
        for mod in selected_mods
        for card in mod.cards
        if int(getattr(card, "count", 0) or 0) > 0
    }
    if not {"thorn", "bloom", "root", "guard"}.issubset(selected_types):
        vanilla = next((mod for mod in official_mods if mod.filename == "Vanilla Cards.gtnmod"), None)
        if vanilla is not None and vanilla not in selected_mods:
            selected_mods.insert(0, vanilla)

    loadout = build_v2_loadout(selected_mods)
    if not loadout.ok:
        raise RuntimeError("invalid selected official loadout: " + "; ".join(loadout.errors))
    allowed_ids = {ERROR_CARD_ID}
    for card_id, resource in (loadout.registries.get("cards", {}) or {}).items():
        allowed_ids.add(str(resource.get("legacy_id") or resource.get("runtime_id") or card_id))
    if selected_mods:
        allowed_ids.update(card_id for card_id in ALL_MOD_SHARED_CARD_IDS if card_id in CARD_DEFS)
    if any(mod.filename == "Vanilla Cards.gtnmod" for mod in selected_mods):
        from game_engine import GameEngine

        allowed_ids.update(card_id for card_id in GameEngine.BUILTIN_SETUP_CARD_IDS if card_id in CARD_DEFS)
    result = loadout, frozenset(allowed_ids), tuple(mod.filename for mod in selected_mods)
    _OFFICIAL_LOADOUT_CACHE[cache_key] = result
    _OFFICIAL_LOADOUT_CACHE.move_to_end(cache_key)
    while len(_OFFICIAL_LOADOUT_CACHE) > OFFICIAL_LOADOUT_CACHE_LIMIT:
        _OFFICIAL_LOADOUT_CACHE.popitem(last=False)
    return result


def apply_loadout_to_engine(engine, loadout, allowed_ids: Iterable[str]) -> None:
    registries = getattr(loadout, "registries", {}) or {}
    engine.allowed_card_ids = set(allowed_ids)
    engine.v2_loadout = loadout
    engine.v2_ui_components = dict(registries.get("ui_components") or {})
    engine.v2_tag_defs = dict(registries.get("tags") or {})
    engine.v2_status_defs = dict(registries.get("statuses") or {})
    engine.v2_opening_event_defs = dict(registries.get("opening_events") or {})
    engine.v2_event_hooks = list(getattr(loadout, "event_hooks", []) or [])


def official_ruleset_fingerprint(game_root: os.PathLike | str | None = None) -> str:
    """Fingerprint all official card data and the production 1v1 runtime.

    A loadout hash identifies one selected mod combination. This fingerprint is
    deliberately global so one policy can train across all official loadouts,
    while still being rejected after a rules or card-content update.
    """

    root = configure_game_imports(game_root)
    cached = _OFFICIAL_RULESET_FINGERPRINT_CACHE.get(str(root))
    if cached is not None:
        return cached

    full_loadout, _, official_names = load_official_content(root)
    digest = hashlib.blake2b(digest_size=20, person=b"GTN-AI-rules-v1")
    digest.update(str(getattr(full_loadout, "loadout_hash", "") or "").encode("utf-8"))
    official_mods = {
        mod.filename: mod
        for mod in (_OFFICIAL_CATALOG_CACHE.get(str(root)) or ())
    }
    for name in official_names:
        digest.update(b"\0mod\0")
        digest.update(str(name).encode("utf-8"))
        digest.update(
            str(getattr(official_mods.get(name), "validation_hash", "") or "").encode("utf-8")
        )
    for relative_name in RULESET_SOURCE_FILES:
        path = root / relative_name
        digest.update(b"\0source\0")
        digest.update(relative_name.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    value = digest.hexdigest()
    _OFFICIAL_RULESET_FINGERPRINT_CACHE[str(root)] = value
    return value


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
