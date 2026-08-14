from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from collections import Counter
from typing import Any, Sequence

from .environment import Garden1v1Env
from .game_imports import load_official_content
from .policies import policy_from_name
from .progress import ProgressReporter
from .self_play import _sample_official_mods, play_episode


def run_arena(
    *,
    pairs: int,
    seed: int,
    policy_a: str,
    policy_b: str,
    enabled_mods: Sequence[str] | None = None,
    sample_mod_combinations: bool = False,
    max_steps: int = 3000,
    workers: int = 1,
    progress_interval: float = 10.0,
    show_progress: bool = True,
) -> dict[str, Any]:
    if enabled_mods is not None and sample_mod_combinations:
        raise ValueError("enabled_mods and sample_mod_combinations are mutually exclusive")
    pair_count = max(0, int(pairs))
    worker_count = max(1, min(int(workers), pair_count or 1))
    catalog = load_official_content()[2] if sample_mod_combinations else ()
    jobs = []
    for index in range(pair_count):
        pair_seed = int(seed) + index
        mods = (
            _sample_official_mods(catalog, pair_seed)
            if sample_mod_combinations
            else (tuple(enabled_mods) if enabled_mods is not None else None)
        )
        jobs.append({
            "seed": pair_seed,
            "policy_a": str(policy_a),
            "policy_b": str(policy_b),
            "enabled_mods": mods,
            "max_steps": int(max_steps),
        })

    started = time.perf_counter()
    if worker_count == 1:
        results = map(_play_pair, jobs)
        pool = None
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count)
        results = pool.map(_play_pair, jobs, chunksize=max(1, pair_count // (worker_count * 4)))

    outcomes: Counter[str] = Counter()
    scores: list[float] = []
    steps = 0
    pregame_steps = 0
    combat_steps = 0
    loop_recoveries = 0
    forced_fallback_actions = 0
    policy_diagnostics: Counter[str] = Counter()
    diagnostic_policies: set[str] = set()
    offline_search_diagnostics: Counter[str] = Counter()
    offline_search_policies: set[str] = set()
    completed_pairs = 0
    progress = ProgressReporter(
        "arena",
        total=pair_count or None,
        interval=progress_interval,
        enabled=show_progress,
    )
    progress.update(0, force=True, games=0)
    try:
        for result in results:
            steps += int(result["steps"])
            pregame_steps += int(result.get("pregame_steps", 0))
            combat_steps += int(result.get("combat_steps", 0))
            loop_recoveries += int(result.get("loop_recoveries", 0))
            forced_fallback_actions += int(result.get("forced_fallback_actions", 0))
            for game in result["games"]:
                outcome = str(game["outcome"])
                outcomes[outcome] += 1
                if outcome == "a_win":
                    scores.append(1.0)
                elif outcome == "b_win":
                    scores.append(0.0)
                elif outcome == "draw":
                    scores.append(0.5)
                for diagnostic in game.get("policy_diagnostics") or []:
                    policy_name = str(diagnostic.get("policy") or "unnamed-policy")
                    diagnostic_policies.add(policy_name)
                    is_offline_search = bool(diagnostic.get("offline_only"))
                    if is_offline_search:
                        offline_search_policies.add(policy_name)
                    for key in (
                        "decisions",
                        "searched_decisions",
                        "bypassed_decisions",
                        "candidate_evaluations",
                        "rollouts",
                        "rollout_steps",
                        "terminal_rollouts",
                        "leaf_evaluations",
                        "action_changes",
                        "belief_samples",
                        "belief_sampled_cards",
                        "belief_history_constraints",
                        "belief_sampling_attempts",
                        "belief_failures",
                        "confidence_bypassed_decisions",
                        "gated_decisions",
                        "prior_adjusted_decisions",
                        "prior_adjusted_actions",
                        "belief_deck_prior_samples",
                        "belief_deck_prior_cards",
                        "belief_exact_prior_samples",
                    ):
                        value = int(diagnostic.get(key, 0) or 0)
                        policy_diagnostics[key] += value
                        if is_offline_search:
                            offline_search_diagnostics[key] += value
                    milliseconds = round(
                        float(diagnostic.get("seconds", 0.0) or 0.0) * 1000
                    )
                    policy_diagnostics["milliseconds"] += milliseconds
                    if is_offline_search:
                        offline_search_diagnostics["milliseconds"] += milliseconds
            completed_pairs += 1
            current_score = statistics.fmean(scores) if scores else 0.0
            progress.update(
                completed_pairs,
                games=completed_pairs * 2,
                score=f"{current_score * 100:.1f}%",
                searches=policy_diagnostics.get("searched_decisions") or None,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)

    current_score = statistics.fmean(scores) if scores else 0.0
    progress.finish(
        completed_pairs,
        games=completed_pairs * 2,
        score=f"{current_score * 100:.1f}%",
        searches=policy_diagnostics.get("searched_decisions") or None,
    )

    elapsed = time.perf_counter() - started
    mean_score = statistics.fmean(scores) if scores else 0.0
    interval = _normal_score_interval(scores)
    return {
        "policy_a": str(policy_a),
        "policy_b": str(policy_b),
        "pairs": pair_count,
        "games": pair_count * 2,
        "rated_games": len(scores),
        "outcomes": dict(sorted(outcomes.items())),
        "policy_a_score": round(mean_score, 6),
        "policy_a_score_95ci": [round(interval[0], 6), round(interval[1], 6)],
        "approximate_elo_a_minus_b": round(_elo_delta(mean_score), 2) if scores else 0.0,
        "steps": steps,
        "pregame_steps": pregame_steps,
        "combat_steps": combat_steps,
        "loop_recoveries": loop_recoveries,
        "forced_fallback_actions": forced_fallback_actions,
        "seconds": round(elapsed, 3),
        "games_per_second": round((pair_count * 2) / elapsed, 3) if elapsed else 0.0,
        "workers": worker_count,
        "sampled_mod_combinations": bool(sample_mod_combinations),
        "policy_diagnostics": (
            {
                "policies": sorted(diagnostic_policies),
                **{
                    key: int(value)
                    for key, value in sorted(policy_diagnostics.items())
                    if key != "milliseconds"
                },
                "seconds": round(policy_diagnostics.get("milliseconds", 0) / 1000, 3),
            }
            if diagnostic_policies
            else None
        ),
        "offline_search": (
            {
                "warning": (
                    "offline diagnostic; full-state mode is private-information unsafe "
                    "and belief mode remains approximate"
                ),
                "policies": sorted(offline_search_policies),
                **{
                    key: int(value)
                    for key, value in sorted(offline_search_diagnostics.items())
                    if key != "milliseconds"
                },
                "seconds": round(
                    offline_search_diagnostics.get("milliseconds", 0) / 1000, 3
                ),
            }
            if offline_search_policies
            else None
        ),
    }


def _play_pair(job: dict[str, Any]) -> dict[str, Any]:
    pair_seed = int(job["seed"])
    policy_names = (str(job["policy_a"]), str(job["policy_b"]))
    games = []
    total_steps = 0
    pregame_steps = 0
    combat_steps = 0
    loop_recoveries = 0
    forced_fallback_actions = 0
    for reverse in (False, True):
        seat_names = policy_names[::-1] if reverse else policy_names
        env = Garden1v1Env(enabled_mods=job.get("enabled_mods"), seed=pair_seed)
        policies = [
            policy_from_name(seat_names[0], seed=pair_seed * 4 + int(reverse) * 2),
            policy_from_name(seat_names[1], seed=pair_seed * 4 + int(reverse) * 2 + 1),
        ]
        episode = play_episode(env, policies, max_steps=int(job["max_steps"]))
        diagnostics = [
            policy.diagnostics()
            for policy in policies
            if callable(getattr(policy, "diagnostics", None))
        ]
        total_steps += episode.steps
        pregame_steps += episode.pregame_steps
        combat_steps += episode.combat_steps
        loop_recoveries += episode.loop_recoveries
        forced_fallback_actions += episode.forced_fallback_actions
        if episode.truncated:
            outcome = "truncated"
        elif episode.winner not in (0, 1):
            outcome = "draw"
        else:
            winner_is_a = (episode.winner == 1) if reverse else (episode.winner == 0)
            outcome = "a_win" if winner_is_a else "b_win"
        games.append({
            "outcome": outcome,
            "steps": episode.steps,
            "rounds": episode.rounds,
            "policy_diagnostics": diagnostics,
        })
    return {
        "games": games,
        "steps": total_steps,
        "pregame_steps": pregame_steps,
        "combat_steps": combat_steps,
        "loop_recoveries": loop_recoveries,
        "forced_fallback_actions": forced_fallback_actions,
    }


def _normal_score_interval(scores: Sequence[float]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    mean = statistics.fmean(scores)
    if len(scores) < 2:
        return max(0.0, mean), min(1.0, mean)
    standard_error = statistics.stdev(scores) / math.sqrt(len(scores))
    return max(0.0, mean - 1.96 * standard_error), min(1.0, mean + 1.96 * standard_error)


def _elo_delta(score: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, float(score)))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate two GTN policies with paired seat swaps")
    parser.add_argument("--pairs", type=int, default=20, help="Each pair plays twice with seats swapped")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-a", default="heuristic")
    parser.add_argument("--policy-b", default="random")
    parser.add_argument("--mod", action="append", dest="mods", help="Official .gtnmod filename; repeat to combine")
    parser.add_argument("--sample-mod-combinations", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summary = run_arena(
        pairs=args.pairs,
        seed=args.seed,
        policy_a=args.policy_a,
        policy_b=args.policy_b,
        enabled_mods=args.mods,
        sample_mod_combinations=args.sample_mod_combinations,
        max_steps=args.max_steps,
        workers=args.workers,
        progress_interval=args.progress_interval,
        show_progress=not args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
