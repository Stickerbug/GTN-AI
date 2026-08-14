from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .environment import Garden1v1Env
from .game_imports import load_official_content
from .policies import Policy, policy_from_name
from .progress import ProgressReporter
from .trajectory import DecisionRecord, Episode, JsonlTrajectoryWriter, action_dicts


def play_episode(
    env: Garden1v1Env,
    policies: Sequence[Policy],
    *,
    max_steps: int = 3000,
    record_decisions: bool = False,
    max_state_repeats: int = 4,
    max_turn_decisions: int = 128,
) -> Episode:
    if len(policies) != 2:
        raise ValueError("self-play requires two policies")
    env.reset()
    decisions: list[DecisionRecord] = []
    truncated = False
    steps = 0
    pregame_steps = 0
    combat_steps = 0
    loop_recoveries = 0
    forced_fallback_actions = 0
    state_visits: Counter[str] = Counter()
    timeout_turn: tuple[Any, ...] | None = None
    active_turn: tuple[Any, ...] | None = None
    turn_decisions = 0
    while not env.engine.game_over:
        if steps >= max_steps:
            truncated = True
            break
        player = env.decision_player()
        if player not in (0, 1):
            raise RuntimeError("engine paused without a decision owner")
        observation = env.observe(player)
        if observation.get("phase") == "pregame":
            pregame_steps += 1
        else:
            combat_steps += 1
        legal = env.legal_actions(player)
        if not legal:
            raise RuntimeError("non-terminal state has no legal action")
        if observation.get("phase") == "pregame":
            own = observation.get("self") or {}
            turn_key = (
                int(player),
                -1,
                str(own.get("status") or ""),
                int(own.get("draft_count") or 0),
            )
        else:
            turn_key = (int(env.engine.current_player), int(env.engine.round_num))
        if turn_key != active_turn:
            active_turn = turn_key
            turn_decisions = 0
            state_visits.clear()
            if timeout_turn != turn_key:
                timeout_turn = None
        turn_decisions += 1
        loop_key = _decision_loop_key(observation)
        state_visits[loop_key] += 1
        repeated = (
            max_state_repeats > 0
            and state_visits[loop_key] > max(1, int(max_state_repeats))
        )
        over_turn_budget = max_turn_decisions > 0 and turn_decisions > int(max_turn_decisions)
        if timeout_turn is None and (repeated or over_turn_budget):
            timeout_turn = turn_key
            loop_recoveries += 1
        forced_fallback = timeout_turn == turn_key
        if forced_fallback:
            action = _timeout_fallback_action(observation, legal)
            forced_fallback_actions += 1
        else:
            action = _select_policy_action(
                policies[player],
                env,
                observation,
                legal,
                player,
            )
        behavior = None if forced_fallback else getattr(
            policies[player], "last_decision_metadata", None
        )
        teacher = None if forced_fallback else getattr(
            policies[player], "last_teacher_metadata", None
        )
        if behavior and behavior.get("action_key") != action.key:
            raise RuntimeError("policy behavior metadata does not match its selected action")
        if teacher:
            teacher_action_key = str(teacher.get("action_key") or "")
            if teacher_action_key not in {legal_action.key for legal_action in legal}:
                raise RuntimeError("teacher metadata action is not legal")
            if len(teacher.get("logits") or ()) != len(legal):
                raise RuntimeError("teacher metadata does not match the legal action count")
        if record_decisions:
            decisions.append(DecisionRecord(
                step=steps,
                player=player,
                observation=observation,
                legal_actions=action_dicts(legal),
                action=action.to_dict(),
                forced_fallback=forced_fallback,
                behavior_log_prob=(float(behavior["log_prob"]) if behavior else None),
                behavior_value=(float(behavior["value"]) if behavior else None),
                behavior_entropy=(float(behavior["entropy"]) if behavior else None),
                behavior_temperature=(float(behavior["temperature"]) if behavior else None),
                teacher=(dict(teacher) if teacher else None),
            ))
        try:
            env.step(action, player)
        except Exception as exc:
            raise RuntimeError(
                f"decision step {steps}, player {player}, round {env.engine.round_num}, "
                f"action {action.key}: {type(exc).__name__}: {exc}"
            ) from exc
        steps += 1
    return Episode(
        seed=env.seed,
        official_mods=list(env.mod_filenames),
        loadout_fingerprint=env.loadout_fingerprint,
        ruleset_fingerprint=env.ruleset_fingerprint,
        policies=[str(policy.name) for policy in policies],
        winner=int(env.engine.winner),
        terminated=bool(env.engine.game_over),
        truncated=truncated,
        steps=steps,
        rounds=int(env.engine.round_num),
        opening_events=list(env.engine.opening_event_picks),
        drafted_decks=[list(picks) for picks in env.engine.draft_picks],
        first_player=int(env.engine.first_player),
        pregame_steps=pregame_steps,
        combat_steps=combat_steps,
        loop_recoveries=loop_recoveries,
        forced_fallback_actions=forced_fallback_actions,
        policy_fingerprints=[
            str(getattr(policy, "model_fingerprint", "") or "") for policy in policies
        ],
        decisions=decisions,
    )


def _select_policy_action(
    policy: Policy,
    env: Garden1v1Env,
    observation: dict[str, Any],
    legal_actions: Sequence[Any],
    player_id: int,
):
    environment_selector = getattr(policy, "select_action_with_env", None)
    if callable(environment_selector):
        return environment_selector(
            env,
            observation,
            legal_actions,
            int(player_id),
        )
    return policy.select_action(observation, legal_actions)


def run_self_play(
    *,
    games: int,
    seed: int,
    policy_names: Sequence[str],
    policy_pool: Sequence[str] | None = None,
    enabled_mods: Sequence[str] | None = None,
    max_steps: int = 3000,
    output: str | Path | None = None,
    record_decisions: bool = False,
    workers: int = 1,
    sample_mod_combinations: bool = False,
    max_state_repeats: int = 4,
    max_turn_decisions: int = 128,
    progress_interval: float = 10.0,
    show_progress: bool = True,
) -> dict[str, Any]:
    if sample_mod_combinations and enabled_mods is not None:
        raise ValueError("enabled_mods and sample_mod_combinations are mutually exclusive")
    league = tuple(str(name) for name in (policy_pool or ()) if str(name).strip())
    if league and len(league) < 2:
        raise ValueError("policy_pool requires at least two policy entries")
    writer = JsonlTrajectoryWriter(output) if output else None
    winners: Counter[int] = Counter()
    outcomes: Counter[str] = Counter()
    total_steps = 0
    total_rounds = 0
    total_pregame_steps = 0
    total_combat_steps = 0
    loop_recoveries = 0
    forced_fallback_actions = 0
    started = time.perf_counter()
    game_count = max(0, int(games))
    worker_count = max(1, min(int(workers), game_count or 1))
    official_catalog = load_official_content()[2] if sample_mod_combinations else ()
    jobs = [
        {
            "seed": int(seed) + index,
            "policy_names": (
                _sample_policy_pair(league, int(seed) + index)
                if league
                else tuple(policy_names)
            ),
            "enabled_mods": (
                _sample_official_mods(official_catalog, int(seed) + index)
                if sample_mod_combinations
                else (tuple(enabled_mods) if enabled_mods is not None else None)
            ),
            "max_steps": int(max_steps),
            "record_decisions": bool(record_decisions),
            "max_state_repeats": max(0, int(max_state_repeats)),
            "max_turn_decisions": max(0, int(max_turn_decisions)),
        }
        for index in range(game_count)
    ]
    matchups = Counter(
        " vs ".join(str(name) for name in job["policy_names"])
        for job in jobs
    )
    completed_games = 0
    progress = ProgressReporter(
        "self-play",
        total=game_count or None,
        interval=progress_interval,
        enabled=show_progress,
    )
    progress.update(0, force=True, steps=0)
    if worker_count == 1:
        episodes = map(_play_episode_job, jobs)
        pool = None
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count)
        futures = [pool.submit(_play_episode_job, job) for job in jobs]
        episodes = (
            future.result() for future in concurrent.futures.as_completed(futures)
        )
    try:
        for episode in episodes:
            completed_games += 1
            winners[episode.winner] += 1
            if episode.truncated:
                outcomes["truncated"] += 1
            elif episode.winner == 0:
                outcomes["player_0_win"] += 1
            elif episode.winner == 1:
                outcomes["player_1_win"] += 1
            else:
                outcomes["draw"] += 1
            total_steps += episode.steps
            total_rounds += episode.rounds
            total_pregame_steps += episode.pregame_steps
            total_combat_steps += episode.combat_steps
            loop_recoveries += episode.loop_recoveries
            forced_fallback_actions += episode.forced_fallback_actions
            if writer:
                writer.write(episode)
            progress.update(
                completed_games,
                steps=total_steps,
                recoveries=loop_recoveries,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
    progress.finish(
        completed_games,
        steps=total_steps,
        recoveries=loop_recoveries,
    )
    elapsed = time.perf_counter() - started
    return {
        "games": game_count,
        "wins": {str(key): value for key, value in sorted(winners.items())},
        "outcomes": dict(sorted(outcomes.items())),
        "steps": total_steps,
        "pregame_steps": total_pregame_steps,
        "combat_steps": total_combat_steps,
        "rounds": total_rounds,
        "seconds": round(elapsed, 3),
        "games_per_second": round(game_count / elapsed, 3) if elapsed else 0.0,
        "steps_per_second": round(total_steps / elapsed, 3) if elapsed else 0.0,
        "workers": worker_count,
        "loop_recoveries": loop_recoveries,
        "forced_fallback_actions": forced_fallback_actions,
        "sampled_mod_combinations": bool(sample_mod_combinations),
        "league_mode": bool(league),
        "policy_matchups": dict(sorted(matchups.items())),
    }


def _sample_official_mods(catalog: Sequence[str], seed: int) -> tuple[str, ...]:
    rng = random.Random(int(seed))
    mask = rng.getrandbits(len(catalog))
    return tuple(name for index, name in enumerate(catalog) if mask & (1 << index))


def _sample_policy_pair(pool: Sequence[str], seed: int) -> tuple[str, str]:
    entries = tuple(str(name) for name in pool)
    if len(entries) < 2:
        raise ValueError("policy pool requires at least two entries")
    rng = random.Random(int(seed) ^ 0x47544E4C)
    return rng.choice(entries), rng.choice(entries)


def _play_episode_job(job: dict[str, Any]) -> Episode:
    episode_seed = int(job["seed"])
    enabled_mods = job.get("enabled_mods")
    try:
        env = Garden1v1Env(enabled_mods=enabled_mods, seed=episode_seed)
        names = job["policy_names"]
        policies = [
            policy_from_name(names[0], seed=episode_seed * 2),
            policy_from_name(names[1], seed=episode_seed * 2 + 1),
        ]
        return play_episode(
            env,
            policies,
            max_steps=int(job["max_steps"]),
            record_decisions=bool(job["record_decisions"]),
            max_state_repeats=int(job.get("max_state_repeats", 4)),
            max_turn_decisions=int(job.get("max_turn_decisions", 128)),
        )
    except Exception as exc:
        mod_names = ", ".join(enabled_mods or ()) or "all official mods"
        raise RuntimeError(
            f"self-play seed {episode_seed} failed with [{mod_names}]: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _decision_loop_key(observation: dict[str, Any]) -> str:
    stable = dict(observation)
    stable.pop("public_history", None)
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()


def _timeout_fallback_action(observation: dict[str, Any], legal_actions: Sequence[Any]):
    actions = list(legal_actions)
    if not actions:
        raise RuntimeError("cannot recover a loop without legal actions")
    if observation.get("phase") == "pregame":
        for kind in (
            "submit_pregame_choice",
            "select_pregame_choice",
            "append_pregame_order",
            "draft_pick",
            "confirm_opening_reveal",
            "select_opening_event",
        ):
            candidates = [action for action in actions if action.kind == kind]
            if candidates:
                return candidates[0]
        toggles = [action for action in actions if action.kind == "toggle_pregame_choice"]
        if toggles:
            return toggles[0]
    pending = observation.get("pending") or {}
    if pending:
        for kind in ("submit_choice", "default_choice", "select_choice", "append_choice_order"):
            candidates = [action for action in actions if action.kind == kind]
            if candidates:
                return candidates[0]
        toggles = [action for action in actions if action.kind == "toggle_choice"]
        if toggles:
            selected = set((pending.get("selection") or {}).get("selected_slots") or [])
            unselected = [
                action for action in toggles
                if action.payload.get("candidate_slot") not in selected
            ]
            if unselected:
                return unselected[0]
        responses = [action for action in actions if action.kind == "respond"]
        if responses:
            return next(
                (action for action in responses if action.payload.get("hand_slot") is None),
                responses[0],
            )
        resolutions = [action for action in actions if action.kind == "resolve_choice"]
        non_cancel_resolution = [
            action for action in resolutions
            if not bool((action.payload.get("choice") or {}).get("cancelled"))
            and not bool((action.payload.get("choice") or {}).get("cancel"))
        ]
        if non_cancel_resolution:
            return non_cancel_resolution[0]
        if resolutions:
            return resolutions[0]
        if toggles:
            return toggles[0]
    end_turn = next((action for action in actions if action.kind == "end_turn"), None)
    if end_turn is not None:
        return end_turn
    non_cancel = [
        action for action in actions
        if not bool((action.payload.get("choice") or {}).get("cancelled"))
        and not bool((action.payload.get("choice") or {}).get("cancel"))
    ]
    return non_cancel[0] if non_cancel else actions[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Garden of Thorn headless 1v1 self-play")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-0", default="heuristic")
    parser.add_argument("--policy-1", default="heuristic")
    parser.add_argument(
        "--league-policy",
        action="append",
        dest="league_policies",
        help="Policy pool entry; repeat entries to weight them and sample both seats per game",
    )
    parser.add_argument("--mod", action="append", dest="mods", help="Official .gtnmod filename; repeat to combine")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output", help="Append episodes to this JSONL file")
    parser.add_argument("--record-decisions", action="store_true", help="Include full information-set decisions")
    parser.add_argument("--workers", type=int, default=1, help="Independent worker processes")
    parser.add_argument(
        "--max-state-repeats",
        type=int,
        default=4,
        help="Simulate the online turn timeout after an information set repeats N times; 0 disables",
    )
    parser.add_argument(
        "--max-turn-decisions",
        type=int,
        default=128,
        help="Simulate the online turn deadline after N decisions in one turn; 0 disables",
    )
    parser.add_argument(
        "--sample-mod-combinations",
        action="store_true",
        help="Uniformly sample a different subset of all official mods for each game",
    )
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summary = run_self_play(
        games=args.games,
        seed=args.seed,
        policy_names=(args.policy_0, args.policy_1),
        policy_pool=args.league_policies,
        enabled_mods=args.mods,
        max_steps=args.max_steps,
        output=args.output,
        record_decisions=args.record_decisions,
        workers=args.workers,
        sample_mod_combinations=args.sample_mod_combinations,
        max_state_repeats=args.max_state_repeats,
        max_turn_decisions=args.max_turn_decisions,
        progress_interval=args.progress_interval,
        show_progress=not args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
