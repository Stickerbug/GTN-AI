from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .player_data import PLAYER_DECISION_DATASET_SCHEMA_VERSION
from .protocol import Action
from .replay_audit import load_downloaded_replay
from .historical_aggregate import (
    HISTORICAL_AGGREGATE_SCHEMA_VERSION,
    contextual_action_keys_from_values,
)


EXPERIENCE_PRIOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperiencePriorConfig:
    min_strict_exposures: int = 50
    min_historical_uses: int = 100
    strict_prior_exposures: float = 100.0
    outcome_prior_uses: float = 160.0
    historical_prior_uses: float = 240.0
    min_context_exposures: int = 80
    context_prior_exposures: float = 120.0
    context_selection_weight: float = 0.10
    selection_weight: float = 0.12
    strict_outcome_weight: float = 0.08
    historical_outcome_weight: float = 0.04
    max_abs_bonus: float = 0.35
    minimum_replay_rounds: int = 4

    def __post_init__(self) -> None:
        for name in ("min_strict_exposures", "min_historical_uses", "min_context_exposures"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "strict_prior_exposures",
            "outcome_prior_uses",
            "historical_prior_uses",
            "context_prior_exposures",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.max_abs_bonus) < 0:
            raise ValueError("max_abs_bonus must be non-negative")


class ExperiencePrior:
    def __init__(
        self,
        entries: dict[str, dict[str, Any]],
        *,
        config: ExperiencePriorConfig,
        ruleset_fingerprint: str = "",
        metadata: dict[str, Any] | None = None,
        fingerprint: str | None = None,
    ):
        self.entries = dict(entries)
        self.config = config
        self.ruleset_fingerprint = str(ruleset_fingerprint or "")
        self.metadata = dict(metadata or {})
        self.fingerprint = str(fingerprint or _prior_fingerprint(self.to_dict()))

    def action_bonus(self, observation: dict[str, Any], action: Action) -> float:
        observed_ruleset = str(
            (observation.get("loadout") or {}).get("ruleset_fingerprint") or ""
        )
        if self.ruleset_fingerprint and observed_ruleset != self.ruleset_fingerprint:
            return 0.0
        for key in action_experience_keys(observation, action):
            entry = self.entries.get(key)
            if entry is not None:
                return float(entry.get("bonus", 0.0) or 0.0)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIENCE_PRIOR_SCHEMA_VERSION,
            "config": asdict(self.config),
            "ruleset_fingerprint": self.ruleset_fingerprint,
            "metadata": self.metadata,
            "entries": self.entries,
        }

    def save(self, path: str | Path) -> None:
        payload = self.to_dict()
        payload["fingerprint"] = _prior_fingerprint(payload)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperiencePrior":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != EXPERIENCE_PRIOR_SCHEMA_VERSION:
            raise ValueError("unsupported experience prior schema")
        expected = str(payload.get("fingerprint") or "")
        unsigned = dict(payload)
        unsigned.pop("fingerprint", None)
        actual = _prior_fingerprint(unsigned)
        if expected and expected != actual:
            raise ValueError("experience prior fingerprint mismatch")
        allowed = set(ExperiencePriorConfig.__dataclass_fields__)
        raw_config = payload.get("config") or {}
        config = ExperiencePriorConfig(**{
            key: raw_config[key] for key in allowed if key in raw_config
        })
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("experience prior entries must be an object")
        return cls(
            entries,
            config=config,
            ruleset_fingerprint=str(payload.get("ruleset_fingerprint") or ""),
            metadata=dict(payload.get("metadata") or {}),
            fingerprint=actual,
        )


class ExperiencePriorPolicy:
    """Small, public-information-only reranker around a learned policy."""

    def __init__(
        self,
        base_policy,
        prior: ExperiencePrior,
        *,
        seed: int = 0,
        name: str | None = None,
    ):
        if not callable(getattr(base_policy, "evaluate_actions", None)):
            raise TypeError("experience prior requires a policy with evaluate_actions")
        self.base_policy = base_policy
        self.prior = prior
        self.name = name or f"{base_policy.name}+experience-prior"
        self.ruleset_fingerprint = str(
            getattr(base_policy, "ruleset_fingerprint", "") or ""
        )
        if (
            prior.ruleset_fingerprint
            and self.ruleset_fingerprint
            and prior.ruleset_fingerprint != self.ruleset_fingerprint
        ):
            raise ValueError("experience prior ruleset does not match the base policy")
        self.model_fingerprint = (
            f"{getattr(base_policy, 'model_fingerprint', 'unknown')}+prior:{prior.fingerprint}"
        )
        self._rng = random.Random(int(seed))
        self._decisions = 0
        self._adjusted_decisions = 0
        self._adjusted_actions = 0
        self._action_changes = 0
        self._absolute_bonus = 0.0
        self.last_decision_metadata = None

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        logits, _ = self.evaluate_actions(observation, actions)
        peak = max(logits)
        best = [
            index for index, value in enumerate(logits)
            if abs(float(value) - peak) <= 1e-8
        ]
        return actions[self._rng.choice(best)]

    def evaluate_actions(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> tuple[list[float], float]:
        actions = list(legal_actions)
        logits, value = self.base_policy.evaluate_actions(observation, actions)
        bonuses = [self.prior.action_bonus(observation, action) for action in actions]
        base_peak = max(range(len(logits)), key=lambda index: float(logits[index]))
        adjusted_logits = [
            float(logit) + float(bonus)
            for logit, bonus in zip(logits, bonuses)
        ]
        adjusted_peak = max(
            range(len(adjusted_logits)), key=lambda index: adjusted_logits[index]
        )
        self._decisions += 1
        adjusted_count = sum(abs(bonus) > 1e-12 for bonus in bonuses)
        if adjusted_count:
            self._adjusted_decisions += 1
            self._adjusted_actions += adjusted_count
            self._absolute_bonus += sum(abs(bonus) for bonus in bonuses)
        if actions[base_peak].key != actions[adjusted_peak].key:
            self._action_changes += 1
        return adjusted_logits, float(value)

    def estimate_value(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> float:
        return float(self.base_policy.estimate_value(observation, legal_actions))

    def estimate_values(self, decisions):
        return self.base_policy.estimate_values(decisions)

    def fork(self, *, seed: int, name: str | None = None, **kwargs):
        base = self.base_policy.fork(seed=seed, name=getattr(self.base_policy, "name", None))
        return ExperiencePriorPolicy(base, self.prior, seed=seed, name=name or self.name)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "offline_only": False,
            "decisions": self._decisions,
            "prior_adjusted_decisions": self._adjusted_decisions,
            "prior_adjusted_actions": self._adjusted_actions,
            "action_changes": self._action_changes,
            "prior_mean_abs_bonus": (
                round(self._absolute_bonus / self._adjusted_actions, 6)
                if self._adjusted_actions
                else 0.0
            ),
        }


def build_experience_prior(
    *,
    strict_datasets: Iterable[str | Path] = (),
    historical_replays: Iterable[str | Path] = (),
    historical_aggregates: Iterable[str | Path] = (),
    config: ExperiencePriorConfig | None = None,
) -> tuple[ExperiencePrior, dict[str, Any]]:
    config = config or ExperiencePriorConfig()
    statistics: dict[str, Counter[str]] = defaultdict(Counter)
    quality: Counter[str] = Counter()
    rulesets: set[str] = set()
    global_stats: Counter[str] = Counter()

    for path in strict_datasets:
        for row in _iter_jsonl(path):
            quality["strict_rows"] += 1
            if int(row.get("schema_version", -1)) != PLAYER_DECISION_DATASET_SCHEMA_VERSION:
                quality["strict_schema_rejected"] += 1
                continue
            observation = row.get("observation")
            legal_raw = row.get("legal_actions")
            selected_raw = row.get("selected_action")
            if not isinstance(observation, dict) or not isinstance(legal_raw, list):
                quality["strict_invalid_rejected"] += 1
                continue
            try:
                legal = [Action.from_dict(value) for value in legal_raw]
                selected = Action.from_dict(selected_raw)
            except (TypeError, ValueError):
                quality["strict_invalid_rejected"] += 1
                continue
            legal_keys = {action.key for action in legal}
            if not legal or selected.key not in legal_keys:
                quality["strict_illegal_rejected"] += 1
                continue
            outcome = max(-1.0, min(1.0, float(row.get("outcome", 0.0) or 0.0)))
            outcome_score = (outcome + 1.0) / 2.0
            for action in legal:
                for key in action_experience_keys(observation, action):
                    statistics[key]["strict_exposures"] += 1
                    global_stats["strict_exposures"] += 1
            for key in action_experience_keys(observation, selected):
                statistics[key]["strict_selections"] += 1
                statistics[key]["strict_outcome_sum"] += outcome_score
                global_stats["strict_selections"] += 1
                global_stats["strict_outcome_sum"] += outcome_score
            ruleset = str(row.get("ruleset_fingerprint") or "")
            if ruleset:
                rulesets.add(ruleset)
            quality["strict_accepted"] += 1

    for path in historical_replays:
        quality["historical_replays"] += 1
        try:
            _, replay = load_downloaded_replay(path)
        except (OSError, ValueError, json.JSONDecodeError):
            quality["historical_invalid_rejected"] += 1
            continue
        reason = _historical_replay_rejection(replay, config=config)
        if reason:
            quality[f"historical_{reason}_rejected"] += 1
            continue
        meta = replay.get("meta") or {}
        winner = _as_int(meta.get("winner_index"), -1)
        for raw_action in replay.get("actions") or []:
            if not isinstance(raw_action, dict):
                continue
            actor = _as_int(raw_action.get("actor"), -1)
            if actor not in (0, 1):
                continue
            keys = historical_action_keys(raw_action, actor=actor)
            if not keys:
                continue
            outcome_score = 0.5 if winner < 0 else (1.0 if winner == actor else 0.0)
            for key in keys:
                statistics[key]["historical_uses"] += 1
                statistics[key]["historical_outcome_sum"] += outcome_score
                global_stats["historical_uses"] += 1
                global_stats["historical_outcome_sum"] += outcome_score
            quality["historical_actions"] += 1
        quality["historical_accepted"] += 1

    for path in historical_aggregates:
        aggregate = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(aggregate.get("schema_version", -1)) != HISTORICAL_AGGREGATE_SCHEMA_VERSION:
            raise ValueError("unsupported historical aggregate schema")
        raw_statistics = aggregate.get("statistics")
        if not isinstance(raw_statistics, dict):
            raise ValueError("historical aggregate statistics must be an object")
        for key, values in raw_statistics.items():
            if not isinstance(values, dict):
                continue
            statistics[str(key)]["historical_uses"] += float(
                values.get("historical_uses", 0.0) or 0.0
            )
            statistics[str(key)]["historical_outcome_sum"] += float(
                values.get("historical_outcome_sum", 0.0) or 0.0
            )
            statistics[str(key)]["historical_raw_uses"] += float(
                values.get("historical_raw_uses", 0.0) or 0.0
            )
            statistics[str(key)]["context_exposures"] += float(
                values.get("context_exposures", 0.0) or 0.0
            )
            statistics[str(key)]["context_selections"] += float(
                values.get("context_selections", 0.0) or 0.0
            )
            statistics[str(key)]["context_raw_exposures"] += float(
                values.get("context_raw_exposures", 0.0) or 0.0
            )
            statistics[str(key)]["context_raw_selections"] += float(
                values.get("context_raw_selections", 0.0) or 0.0
            )
        for key, value in (aggregate.get("global") or {}).items():
            global_stats[str(key)] += float(value or 0.0)
        for key, value in (aggregate.get("quality") or {}).items():
            quality[f"aggregate_{key}"] += float(value or 0.0)
        quality["historical_aggregate_files"] += 1

    if len(rulesets) > 1:
        raise ValueError("strict player datasets contain mixed rulesets")
    entries = _finalize_entries(statistics, global_stats, config=config)
    metadata = {
        "quality": dict(sorted(quality.items())),
        "global": dict(sorted(global_stats.items())),
        "retained_entries": len(entries),
        "design": "hierarchical_shrunk_action_prior",
    }
    prior = ExperiencePrior(
        entries,
        config=config,
        ruleset_fingerprint=next(iter(rulesets), ""),
        metadata=metadata,
    )
    return prior, metadata


def action_experience_keys(
    observation: dict[str, Any], action: Action
) -> tuple[str, ...]:
    phase = "pregame" if observation.get("phase") == "pregame" else "combat"
    card_id = _action_card_id(observation, action)
    target = _action_target_relation(observation, action)
    round_bucket = _round_bucket(observation.get("round"), phase=phase)
    contextual = ()
    if phase == "combat" and action.kind in {"play_card", "end_turn"}:
        own = observation.get("self") or {}
        opponent = observation.get("opponent") or {}
        contextual = contextual_action_keys_from_values(
            kind=action.kind,
            card_id=card_id,
            round_value=observation.get("round"),
            own_health=own.get("health"),
            own_max_health=own.get("max_health"),
            opponent_health=opponent.get("health"),
            opponent_max_health=opponent.get("max_health"),
            elixir=own.get("elixir"),
            magic=own.get("magic"),
            hand_count=len(own.get("hand") or []),
        )
    return contextual + _hierarchical_keys(
        phase, action.kind, card_id, target, round_bucket
    )


def historical_action_keys(
    raw_action: dict[str, Any], *, actor: int
) -> tuple[str, ...]:
    action_type = str(raw_action.get("type") or "")
    kind_map = {
        "draft_pick": "draft_pick",
        "end_turn": "end_turn",
        "play_card": "play_card",
    }
    kind = kind_map.get(action_type)
    if kind is None:
        return ()
    payload = raw_action.get("payload") if isinstance(raw_action.get("payload"), dict) else {}
    card_id = str(payload.get("def_id") or "")
    if kind == "play_card" and not card_id:
        return ()
    target_id = payload.get("target_player_id")
    choice = payload.get("choice") if isinstance(payload.get("choice"), dict) else {}
    if target_id is None:
        target_id = choice.get("target_player_id", choice.get("target_player"))
    target = _target_relation(target_id, seat=actor)
    phase = "pregame" if kind == "draft_pick" else "combat"
    round_bucket = _round_bucket(raw_action.get("round"), phase=phase)
    return _hierarchical_keys(phase, kind, card_id, target, round_bucket)


def _finalize_entries(statistics, global_stats, *, config: ExperiencePriorConfig):
    global_selection = _ratio(
        global_stats["strict_selections"], global_stats["strict_exposures"], 0.1
    )
    global_strict_outcome = _ratio(
        global_stats["strict_outcome_sum"], global_stats["strict_selections"], 0.5
    )
    global_historical_outcome = _ratio(
        global_stats["historical_outcome_sum"], global_stats["historical_uses"], 0.5
    )
    global_context_selection = _ratio(
        global_stats["context_selections"], global_stats["context_exposures"], 0.1
    )
    entries = {}
    for key, raw in statistics.items():
        exposure = int(raw["strict_exposures"])
        historical_uses = float(raw["historical_uses"])
        context_exposures = float(raw["context_exposures"])
        strict_eligible = exposure >= int(config.min_strict_exposures)
        historical_eligible = historical_uses >= int(config.min_historical_uses)
        context_eligible = context_exposures >= int(config.min_context_exposures)
        if not strict_eligible and not historical_eligible and not context_eligible:
            continue
        bonus = 0.0
        if context_eligible:
            context_rate = _posterior_mean(
                raw["context_selections"],
                context_exposures,
                global_context_selection,
                config.context_prior_exposures,
            )
            context_confidence = context_exposures / (
                context_exposures + float(config.context_prior_exposures)
            )
            bonus += (
                config.context_selection_weight
                * context_confidence
                * (_logit(context_rate) - _logit(global_context_selection))
            )
        if strict_eligible:
            selected = int(raw["strict_selections"])
            selection_rate = _posterior_mean(
                selected,
                exposure,
                global_selection,
                config.strict_prior_exposures,
            )
            selection_confidence = exposure / (
                exposure + float(config.strict_prior_exposures)
            )
            bonus += (
                config.selection_weight
                * selection_confidence
                * (_logit(selection_rate) - _logit(global_selection))
            )
            if selected:
                outcome_rate = _posterior_mean(
                    raw["strict_outcome_sum"],
                    selected,
                    global_strict_outcome,
                    config.outcome_prior_uses,
                )
                outcome_confidence = selected / (
                    selected + float(config.outcome_prior_uses)
                )
                bonus += (
                    config.strict_outcome_weight
                    * outcome_confidence
                    * (_logit(outcome_rate) - _logit(global_strict_outcome))
                )
        if historical_eligible:
            historical_rate = _posterior_mean(
                raw["historical_outcome_sum"],
                historical_uses,
                global_historical_outcome,
                config.historical_prior_uses,
            )
            historical_confidence = historical_uses / (
                historical_uses + float(config.historical_prior_uses)
            )
            bonus += (
                config.historical_outcome_weight
                * historical_confidence
                * (_logit(historical_rate) - _logit(global_historical_outcome))
            )
        bonus = max(-config.max_abs_bonus, min(config.max_abs_bonus, bonus))
        entries[key] = {
            "bonus": round(float(bonus), 6),
            "strict_exposures": exposure,
            "strict_selections": int(raw["strict_selections"]),
            "historical_uses": round(historical_uses, 3),
            "historical_raw_uses": int(raw["historical_raw_uses"]),
            "context_exposures": round(context_exposures, 3),
            "context_selections": round(float(raw["context_selections"]), 3),
            "context_raw_exposures": int(raw["context_raw_exposures"]),
        }
    return dict(sorted(entries.items()))


def _historical_replay_rejection(
    replay: dict[str, Any], *, config: ExperiencePriorConfig
) -> str | None:
    meta = replay.get("meta") if isinstance(replay.get("meta"), dict) else {}
    if str(meta.get("mode") or "") != "1v1":
        return "mode"
    if bool(meta.get("truncated") or replay.get("truncated")):
        return "truncated"
    winner = _as_int(meta.get("winner_index"), -1)
    if winner not in (-1, 0, 1):
        return "winner"
    actions = [value for value in replay.get("actions") or [] if isinstance(value, dict)]
    action_types = {str(value.get("type") or "") for value in actions}
    if action_types & {"disconnect_timeout", "both_disconnected_result", "player_exit"}:
        return "disconnect"
    maximum_round = max((_as_int(value.get("round"), 0) for value in actions), default=0)
    if "surrender" in action_types and maximum_round < int(config.minimum_replay_rounds):
        return "early_surrender"
    if maximum_round < int(config.minimum_replay_rounds):
        return "too_short"
    return None


def _action_card_id(observation: dict[str, Any], action: Action) -> str:
    payload = action.payload
    explicit = payload.get("def_id")
    if explicit:
        return str(explicit)
    slot = payload.get("hand_slot")
    if slot is not None:
        card = _card_at_slot((observation.get("self") or {}).get("hand"), slot)
        return str((card or {}).get("def_id") or "")
    candidate_slot = payload.get("candidate_slot")
    if candidate_slot is not None:
        own = observation.get("self") or {}
        for zone in ("draft_options", "opening_event_options"):
            card = _card_at_slot(own.get(zone), candidate_slot)
            if card:
                return str(card.get("def_id") or card.get("id") or "")
        pending = observation.get("pending") or {}
        card = _card_at_slot(pending.get("candidates"), candidate_slot)
        if card:
            return str(card.get("def_id") or card.get("id") or "")
    equipment_slot = payload.get("equipment_slot")
    if equipment_slot is not None:
        equipment = _card_at_slot(
            (observation.get("self") or {}).get("equipment"), equipment_slot
        )
        card = (equipment or {}).get("card") if isinstance(equipment, dict) else None
        return str((card or {}).get("def_id") or "")
    return ""


def _action_target_relation(observation: dict[str, Any], action: Action) -> str:
    payload = action.payload
    target = payload.get("target_player_id")
    choice = payload.get("choice") if isinstance(payload.get("choice"), dict) else {}
    if target is None:
        target = choice.get("target_player_id", choice.get("target_player"))
    return _target_relation(target, seat=_as_int(observation.get("seat"), 0))


def _target_relation(target: Any, *, seat: int) -> str:
    target_id = _as_int(target, -1)
    if target_id < 0:
        return "none"
    return "self" if target_id == seat else "opponent"


def _hierarchical_keys(
    phase: str, kind: str, card_id: str, target: str, round_bucket: str
) -> tuple[str, ...]:
    values = (
        (phase, kind, card_id, target, round_bucket),
        (phase, kind, card_id, target, "*"),
        (phase, kind, card_id, "*", "*"),
    )
    return tuple(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        for value in values
    )


def _round_bucket(value: Any, *, phase: str) -> str:
    if phase == "pregame":
        return "pregame"
    round_number = _as_int(value, 0)
    if round_number <= 2:
        return "1-2"
    if round_number <= 5:
        return "3-5"
    if round_number <= 9:
        return "6-9"
    return "10+"


def _card_at_slot(cards: Any, slot: Any) -> dict[str, Any] | None:
    if not isinstance(cards, list):
        return None
    expected = _as_int(slot, -1)
    for position, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        if _as_int(card.get("slot"), position) == expected:
            return card
    return None


def _iter_jsonl(path: str | Path):
    source = Path(path)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _posterior_mean(successes, trials, center: float, prior: float) -> float:
    return (float(successes) + float(center) * float(prior)) / (
        float(trials) + float(prior)
    )


def _ratio(numerator, denominator, default: float) -> float:
    return float(numerator) / float(denominator) if denominator else float(default)


def _logit(value: float) -> float:
    bounded = max(1e-5, min(1.0 - 1e-5, float(value)))
    return math.log(bounded / (1.0 - bounded))


def _prior_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a conservative action prior from anonymized player data"
    )
    parser.add_argument("--strict", nargs="*", default=[], help="Strict player JSONL data")
    parser.add_argument(
        "--historical", nargs="*", default=[], help="Old .gtnreplay files (statistics only)"
    )
    parser.add_argument(
        "--historical-aggregate",
        nargs="*",
        default=[],
        help="Identity-free aggregate JSON files",
    )
    parser.add_argument("--output", required=True, help="Output prior JSON")
    parser.add_argument("--min-strict", type=int, default=50)
    parser.add_argument("--min-historical", type=int, default=100)
    parser.add_argument("--historical-weight", type=float, default=0.04)
    parser.add_argument("--context-weight", type=float, default=0.10)
    parser.add_argument("--min-context", type=int, default=80)
    parser.add_argument("--max-bonus", type=float, default=0.35)
    args = parser.parse_args(argv)
    config = ExperiencePriorConfig(
        min_strict_exposures=args.min_strict,
        min_historical_uses=args.min_historical,
        historical_outcome_weight=args.historical_weight,
        context_selection_weight=args.context_weight,
        min_context_exposures=args.min_context,
        max_abs_bonus=args.max_bonus,
    )
    prior, report = build_experience_prior(
        strict_datasets=args.strict,
        historical_replays=args.historical,
        historical_aggregates=args.historical_aggregate,
        config=config,
    )
    prior.save(args.output)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "fingerprint": prior.fingerprint,
        **report,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
