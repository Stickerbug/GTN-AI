from __future__ import annotations

from gtn_ai.game_imports import OFFICIAL_LOADOUT_CACHE_LIMIT, _OFFICIAL_LOADOUT_CACHE
from gtn_ai.validate_loadouts import validate_official_loadouts


def test_loadout_validator_checks_requested_masks_and_bounds_cache():
    report = validate_official_loadouts(limit=16)
    assert report["checked_requested_combinations"] == 16
    assert not report["failures"]
    assert report["minimum_allowed_cards"] > 0
    assert len(_OFFICIAL_LOADOUT_CACHE) <= OFFICIAL_LOADOUT_CACHE_LIMIT
