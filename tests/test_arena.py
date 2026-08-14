from __future__ import annotations

from gtn_ai.arena import run_arena


def test_arena_pairs_swap_seats_and_report_every_game():
    result = run_arena(
        pairs=1,
        seed=301,
        policy_a="random",
        policy_b="random",
        enabled_mods=["Vanilla Cards.gtnmod"],
        max_steps=2000,
    )
    assert result["pairs"] == 1
    assert result["games"] == 2
    assert sum(result["outcomes"].values()) == 2
