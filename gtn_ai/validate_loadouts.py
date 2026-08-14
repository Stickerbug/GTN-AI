from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from typing import Any, Sequence

from .game_imports import load_official_content, official_ruleset_fingerprint


def validate_official_loadouts(*, limit: int | None = None) -> dict[str, Any]:
    catalog = tuple(load_official_content()[2])
    requested_total = 1 << len(catalog)
    check_total = requested_total if limit is None else min(requested_total, max(0, int(limit)))
    effective = Counter()
    failures = []
    min_cards = None
    max_cards = 0
    started = time.perf_counter()
    masks = itertools.islice(range(requested_total), check_total)
    for mask in masks:
        requested = tuple(
            name for index, name in enumerate(catalog)
            if mask & (1 << index)
        )
        try:
            _, allowed_ids, active = load_official_content(enabled_mods=requested)
            effective[active] += 1
            card_count = max(0, len(allowed_ids) - 1)
            min_cards = card_count if min_cards is None else min(min_cards, card_count)
            max_cards = max(max_cards, card_count)
        except Exception as exc:
            failures.append({
                "mask": mask,
                "requested_mods": list(requested),
                "error": f"{type(exc).__name__}: {exc}",
            })
    elapsed = time.perf_counter() - started
    fallback_count = sum(
        count
        for active, count in effective.items()
        if "Vanilla Cards.gtnmod" in active
    )
    return {
        "ruleset_fingerprint": official_ruleset_fingerprint(),
        "official_mods": list(catalog),
        "possible_requested_combinations": requested_total,
        "checked_requested_combinations": check_total,
        "unique_effective_loadouts": len(effective),
        "requested_combinations_with_vanilla_active": fallback_count,
        "minimum_allowed_cards": min_cards or 0,
        "maximum_allowed_cards": max_cards,
        "failures": failures,
        "seconds": round(elapsed, 3),
        "loadouts_per_second": round(check_total / elapsed, 3) if elapsed else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate every requested combination of official GTN mods"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Check only the first N bitmasks for a quick smoke test",
    )
    args = parser.parse_args(argv)
    report = validate_official_loadouts(limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
