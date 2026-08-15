from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .historical_aggregate import HISTORICAL_AGGREGATE_SCHEMA_VERSION


DECK_PRIOR_SCHEMA_VERSION = 1
DECK_BELIEF_FEATURE_SCHEMA_VERSION = 2
GLOBAL_DECK_BUCKET = "*"


@dataclass(frozen=True)
class DeckPriorConfig:
    min_exact_decks: float = 12.0
    min_global_decks: float = 50.0
    exact_prior_decks: float = 24.0
    related_prior_decks: float = 18.0
    global_prior_decks: float = 120.0
    exact_max_mix: float = 0.78
    related_max_mix: float = 0.55
    global_max_mix: float = 0.30
    min_related_decks: float = 5.0
    min_related_similarity: float = 0.20
    max_related_buckets: int = 8

    def __post_init__(self) -> None:
        for name in (
            "min_exact_decks",
            "min_global_decks",
            "exact_prior_decks",
            "related_prior_decks",
            "global_prior_decks",
            "min_related_decks",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "exact_max_mix",
            "related_max_mix",
            "global_max_mix",
            "min_related_similarity",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if int(self.max_related_buckets) < 0:
            raise ValueError("max_related_buckets must be non-negative")


class DeckPrior:
    """Anonymous population draft frequencies for hidden-card sampling."""

    def __init__(
        self,
        buckets: dict[str, dict[str, Any]],
        *,
        config: DeckPriorConfig | None = None,
        metadata: dict[str, Any] | None = None,
        fingerprint: str | None = None,
    ) -> None:
        self.buckets = dict(buckets)
        self.config = config or DeckPriorConfig()
        self.metadata = dict(metadata or {})
        self.fingerprint = str(fingerprint or _fingerprint(self.to_dict()))
        self._mod_sets = _bucket_mod_sets(self.buckets)
        self._weight_cache: dict[tuple[Any, ...], tuple[tuple[float, ...], str]] = {}
        self._belief_feature_cache: dict[tuple[Any, ...], tuple[dict[str, Any], ...]] = {}

    def adjusted_weights(
        self,
        *,
        mod_ids: Iterable[str],
        def_ids: Sequence[str],
        native_weights: Sequence[float],
    ) -> tuple[list[float], str]:
        if len(def_ids) != len(native_weights):
            raise ValueError("deck prior ids and weights must have equal length")
        if not def_ids:
            return [], "native"
        normalized_mods = tuple(sorted({str(value) for value in mod_ids if str(value)}))
        cache_key = (
            normalized_mods,
            tuple(str(value) for value in def_ids),
            tuple(float(value) for value in native_weights),
        )
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return list(cached[0]), cached[1]
        native = _normalized(native_weights)
        probabilities = list(native)
        source = "native"

        global_bucket = self.buckets.get(GLOBAL_DECK_BUCKET)
        if _bucket_support(global_bucket) >= float(self.config.min_global_decks):
            empirical = _bucket_distribution(global_bucket, def_ids)
            if empirical is not None:
                support = _bucket_support(global_bucket)
                mix = float(self.config.global_max_mix) * _saturation(
                    support, float(self.config.global_prior_decks)
                )
                probabilities = _blend(probabilities, empirical, mix)
                source = "global"

        related = self._related_distribution(normalized_mods, def_ids)
        if related is not None:
            empirical, related_strength = related
            mix = float(self.config.related_max_mix) * related_strength
            probabilities = _blend(probabilities, empirical, mix)
            source = "related"

        exact_bucket = self.buckets.get(deck_bucket_key(normalized_mods))
        if _bucket_support(exact_bucket) >= float(self.config.min_exact_decks):
            empirical = _bucket_distribution(exact_bucket, def_ids)
            if empirical is not None:
                support = _bucket_support(exact_bucket)
                mix = float(self.config.exact_max_mix) * _saturation(
                    support, float(self.config.exact_prior_decks)
                )
                probabilities = _blend(probabilities, empirical, mix)
                source = "exact"

        # random.choices accepts arbitrary positive weights. Keeping probabilities
        # normalized makes diagnostics and tests easier to reason about.
        if len(self._weight_cache) >= 2048:
            self._weight_cache.clear()
        self._weight_cache[cache_key] = (tuple(probabilities), source)
        return probabilities, source

    def observation_features(
        self,
        official_mods: Iterable[str],
        *,
        max_cards_per_type: int = 5,
    ) -> list[dict[str, Any]]:
        filenames = tuple(sorted({str(value) for value in official_mods if str(value)}))
        limit = max(1, int(max_cards_per_type))
        cache_key = (filenames, limit)
        cached = self._belief_feature_cache.get(cache_key)
        if cached is not None:
            return [dict(value) for value in cached]

        from .game_imports import load_official_content

        loadout, allowed_ids, _ = load_official_content(enabled_mods=filenames)
        from cards import CARD_DEFS, DRAFT_RATIO, _effective_draft_weights, normalize_card_flags
        effective = _effective_draft_weights(set(allowed_ids))
        by_type: dict[str, tuple[list[str], list[float]]] = {}
        for def_id, weight in effective.items():
            card = CARD_DEFS.get(def_id)
            if card is None or float(weight) <= 0.0:
                continue
            flags = normalize_card_flags(getattr(card, "flags", set()) or set())
            if "team_limited" in flags:
                continue
            ids, weights = by_type.setdefault(str(card.card_type), ([], []))
            ids.append(str(def_id))
            weights.append(float(weight))
        mod_ids = tuple(sorted({str(value) for value in getattr(loadout, "load_order", ())}))
        features: list[dict[str, Any]] = []
        for card_type, quota in DRAFT_RATIO.items():
            ids, native_weights = by_type.get(str(card_type), ([], []))
            if not ids:
                continue
            probabilities, source = self.adjusted_weights(
                mod_ids=mod_ids,
                def_ids=ids,
                native_weights=native_weights,
            )
            ranked = sorted(
                range(len(ids)),
                key=lambda index: (-probabilities[index], ids[index]),
            )[:limit]
            for rank, index in enumerate(ranked):
                features.append({
                    "def_id": ids[index],
                    "card_type": str(card_type),
                    "probability": round(float(probabilities[index]), 7),
                    "expected_count": round(float(probabilities[index]) * int(quota), 7),
                    "rank": rank,
                    "source": source,
                })
        frozen = tuple(dict(value) for value in features)
        if len(self._belief_feature_cache) >= 256:
            self._belief_feature_cache.clear()
        self._belief_feature_cache[cache_key] = frozen
        return [dict(value) for value in frozen]

    def public_observation_features(
        self,
        observation: dict[str, Any],
        *,
        max_cards_per_type: int = 5,
    ) -> list[dict[str, Any]]:
        """Summarize the population prior conditioned only on public evidence."""

        loadout = observation.get("loadout") or {}
        official_mods = loadout.get("official_mods") or ()
        evidence = _public_opponent_card_evidence(observation)
        if not evidence:
            return self.observation_features(
                official_mods,
                max_cards_per_type=max_cards_per_type,
            )

        # The complete distribution is small (currently below a few hundred
        # cards) and cached by observation_features. It lets observed cards
        # survive even when they were outside the static top-k prior.
        population = self.observation_features(
            official_mods,
            max_cards_per_type=1 << 20,
        )
        by_type: dict[str, list[dict[str, Any]]] = {}
        for feature in population:
            by_type.setdefault(str(feature.get("card_type") or ""), []).append(feature)

        limit = max(1, int(max_cards_per_type))
        result: list[dict[str, Any]] = []
        for card_type, features in by_type.items():
            observed = [
                feature for feature in features
                if str(feature.get("def_id") or "") in evidence
            ]
            unobserved = [
                feature for feature in features
                if str(feature.get("def_id") or "") not in evidence
            ]
            selected = observed + unobserved[:max(0, limit - len(observed))]
            for rank, feature in enumerate(selected):
                def_id = str(feature.get("def_id") or "")
                item = dict(feature)
                item["rank"] = rank
                card_evidence = evidence.get(def_id)
                if card_evidence is not None:
                    item.update({
                        "observed": True,
                        "evidence_count": int(card_evidence["count"]),
                        "evidence_sources": sorted(card_evidence["sources"]),
                    })
                result.append(item)
        return result

    def _related_distribution(
        self,
        mod_ids: Sequence[str],
        def_ids: Sequence[str],
    ) -> tuple[list[float], float] | None:
        current = frozenset(mod_ids)
        if not current or int(self.config.max_related_buckets) == 0:
            return None
        exact_key = deck_bucket_key(current)
        candidates = []
        for key, bucket_mods in self._mod_sets.items():
            if key == exact_key or not bucket_mods:
                continue
            support = _bucket_support(self.buckets.get(key))
            if support < float(self.config.min_related_decks):
                continue
            overlap = len(current & bucket_mods)
            union = len(current | bucket_mods)
            similarity = overlap / union if union else 0.0
            if similarity < float(self.config.min_related_similarity):
                continue
            distribution = _bucket_distribution(self.buckets.get(key), def_ids)
            if distribution is None:
                continue
            candidates.append((similarity, support, key, distribution))
        candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
        selected = candidates[: int(self.config.max_related_buckets)]
        if not selected:
            return None
        total_weight = 0.0
        combined = [0.0 for _ in def_ids]
        strength_sum = 0.0
        for similarity, support, _, distribution in selected:
            saturation = _saturation(support, float(self.config.related_prior_decks))
            weight = similarity ** 3 * saturation
            total_weight += weight
            strength_sum += similarity * weight
            for index, probability in enumerate(distribution):
                combined[index] += weight * float(probability)
        if total_weight <= 0.0:
            return None
        return (
            _normalized([value / total_weight for value in combined]),
            max(0.0, min(1.0, strength_sum / total_weight)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DECK_PRIOR_SCHEMA_VERSION,
            "config": asdict(self.config),
            "metadata": self.metadata,
            "buckets": self.buckets,
        }

    def save(self, path: str | Path) -> None:
        payload = self.to_dict()
        payload["fingerprint"] = _fingerprint(payload)
        _write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "DeckPrior":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != DECK_PRIOR_SCHEMA_VERSION:
            raise ValueError("unsupported deck prior schema")
        expected = str(payload.get("fingerprint") or "")
        unsigned = dict(payload)
        unsigned.pop("fingerprint", None)
        actual = _fingerprint(unsigned)
        if expected and expected != actual:
            raise ValueError("deck prior fingerprint mismatch")
        raw_config = payload.get("config") or {}
        allowed = set(DeckPriorConfig.__dataclass_fields__)
        config = DeckPriorConfig(**{
            key: raw_config[key] for key in allowed if key in raw_config
        })
        buckets = payload.get("buckets")
        if not isinstance(buckets, dict):
            raise ValueError("deck prior buckets must be an object")
        return cls(
            buckets,
            config=config,
            metadata=dict(payload.get("metadata") or {}),
            fingerprint=actual,
        )


class DeckBeliefPolicy:
    """Public-information policy wrapper that adds population deck tokens."""

    def __init__(
        self,
        base_policy: Any,
        prior: DeckPrior,
        *,
        name: str | None = None,
        max_cards_per_type: int = 5,
        include_public_evidence: bool = False,
    ) -> None:
        self.base_policy = base_policy
        self.prior = prior
        self.max_cards_per_type = max(1, int(max_cards_per_type))
        self.include_public_evidence = bool(include_public_evidence)
        self.name = name or f"{base_policy.name}+deck-belief"
        self.ruleset_fingerprint = str(
            getattr(base_policy, "ruleset_fingerprint", "") or ""
        )
        self.model_fingerprint = (
            f"{getattr(base_policy, 'model_fingerprint', 'unknown')}"
            f"+deck:{prior.fingerprint}"
        )
        self.offline_only = bool(getattr(base_policy, "offline_only", False))
        self.uses_private_engine_state = bool(
            getattr(base_policy, "uses_private_engine_state", False)
        )
        self.last_decision_metadata = None

    def _augment(self, observation: dict[str, Any]) -> dict[str, Any]:
        return augment_observation_with_deck_prior(
            observation,
            self.prior,
            max_cards_per_type=self.max_cards_per_type,
            include_public_evidence=self.include_public_evidence,
        )

    def select_action(self, observation, legal_actions):
        action = self.base_policy.select_action(self._augment(observation), legal_actions)
        self.last_decision_metadata = getattr(
            self.base_policy, "last_decision_metadata", None
        )
        return action

    def evaluate_actions(self, observation, legal_actions):
        return self.base_policy.evaluate_actions(self._augment(observation), legal_actions)

    def estimate_value(self, observation, legal_actions):
        return self.base_policy.estimate_value(self._augment(observation), legal_actions)

    def estimate_values(self, decisions):
        return self.base_policy.estimate_values([
            (self._augment(observation), legal_actions)
            for observation, legal_actions in decisions
        ])

    def fork(self, *, seed: int, name: str | None = None, **kwargs):
        base = self.base_policy.fork(
            seed=seed,
            name=getattr(self.base_policy, "name", None),
            **kwargs,
        )
        return DeckBeliefPolicy(
            base,
            self.prior,
            name=name or self.name,
            max_cards_per_type=self.max_cards_per_type,
            include_public_evidence=self.include_public_evidence,
        )

    def diagnostics(self) -> dict[str, Any]:
        base_diagnostics = getattr(self.base_policy, "diagnostics", None)
        result = dict(base_diagnostics() if callable(base_diagnostics) else {})
        result.update({
            "policy": self.name,
            "offline_only": self.offline_only,
            "deck_belief_fingerprint": self.prior.fingerprint,
            "deck_belief_schema_version": (
                DECK_BELIEF_FEATURE_SCHEMA_VERSION
                if self.include_public_evidence
                else None
            ),
        })
        return result


def build_deck_prior(
    historical_aggregates: Iterable[str | Path],
    *,
    config: DeckPriorConfig | None = None,
    minimum_raw_decks: int = 5,
) -> tuple[DeckPrior, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    quality: Counter[str] = Counter()
    sources = 0
    for path in historical_aggregates:
        aggregate = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(aggregate.get("schema_version", -1)) != HISTORICAL_AGGREGATE_SCHEMA_VERSION:
            quality["rejected_schema"] += 1
            continue
        statistics = aggregate.get("deck_statistics")
        if not isinstance(statistics, dict):
            quality["rejected_missing_decks"] += 1
            continue
        sources += 1
        for key, raw_bucket in statistics.items():
            if not isinstance(raw_bucket, dict):
                continue
            bucket = merged.setdefault(str(key), {
                "effective_decks": 0.0,
                "raw_decks": 0,
                "card_weights": Counter(),
                "raw_card_counts": Counter(),
            })
            bucket["effective_decks"] += _finite_float(
                raw_bucket.get("effective_decks"), 0.0
            )
            bucket["raw_decks"] += max(0, int(raw_bucket.get("raw_decks", 0) or 0))
            for field in ("card_weights", "raw_card_counts"):
                values = raw_bucket.get(field)
                if not isinstance(values, dict):
                    continue
                for def_id, value in values.items():
                    amount = _finite_float(value, 0.0)
                    if amount > 0.0:
                        bucket[field][str(def_id)] += amount

    retained: dict[str, dict[str, Any]] = {}
    for key, bucket in merged.items():
        raw_decks = int(bucket["raw_decks"])
        if key != GLOBAL_DECK_BUCKET and raw_decks < max(1, int(minimum_raw_decks)):
            quality["small_buckets_removed"] += 1
            continue
        retained[key] = {
            "effective_decks": round(float(bucket["effective_decks"]), 6),
            "raw_decks": raw_decks,
            "card_weights": _rounded_positive_map(bucket["card_weights"]),
            "raw_card_counts": {
                card_id: int(round(value))
                for card_id, value in sorted(bucket["raw_card_counts"].items())
                if value > 0.0
            },
        }
    prior = DeckPrior(
        retained,
        config=config or DeckPriorConfig(),
        metadata={
            "source_aggregates": sources,
            "minimum_raw_decks": max(1, int(minimum_raw_decks)),
            "identity_free": True,
        },
    )
    report = {
        "source_aggregates": sources,
        "retained_buckets": len(retained),
        "retained_cards": len({
            card_id
            for bucket in retained.values()
            for card_id in bucket.get("card_weights", {})
        }),
        "quality": dict(sorted(quality.items())),
    }
    return prior, report


def evaluate_deck_prior(
    prior: DeckPrior,
    historical_aggregate: str | Path,
) -> dict[str, Any]:
    """Compare native and empirical draft likelihood on an aggregate holdout."""

    aggregate = json.loads(Path(historical_aggregate).read_text(encoding="utf-8"))
    statistics = aggregate.get("deck_statistics")
    if not isinstance(statistics, dict):
        raise ValueError("holdout aggregate has no deck statistics")

    from .environment import Garden1v1Env
    from .game_imports import load_official_content

    load_official_content()
    from cards import CARD_DEFS, _effective_draft_weights, normalize_card_flags
    from mod_loader import load_all_mods, mod_category

    official_filenames = {
        str(mod.manifest.id): str(mod.filename)
        for mod in load_all_mods(force=True)
        if not mod.errors and mod_category(mod) == "official"
    }
    native_nll = 0.0
    prior_nll = 0.0
    native_top = 0.0
    prior_top = 0.0
    supported = 0.0
    unsupported = 0.0
    evaluated_buckets = 0
    sources: Counter[str] = Counter()
    for key, bucket in sorted(statistics.items()):
        if key == GLOBAL_DECK_BUCKET or not isinstance(bucket, dict):
            continue
        try:
            mod_ids = json.loads(key)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(mod_ids, list) or any(
            str(mod_id) not in official_filenames for mod_id in mod_ids
        ):
            raw_counts = bucket.get("raw_card_counts") or {}
            unsupported += sum(_finite_float(value, 0.0) for value in raw_counts.values())
            continue
        env = Garden1v1Env(
            enabled_mods=[official_filenames[str(mod_id)] for mod_id in mod_ids],
            seed=0,
            include_pregame=False,
        )
        env.reset()
        effective = _effective_draft_weights(set(env.allowed_card_ids))
        candidates: dict[str, tuple[list[str], list[float]]] = {}
        for def_id, weight in effective.items():
            card = CARD_DEFS.get(def_id)
            if card is None or float(weight) <= 0.0:
                continue
            flags = normalize_card_flags(getattr(card, "flags", set()) or set())
            if "team_limited" in flags:
                continue
            ids, weights = candidates.setdefault(str(card.card_type), ([], []))
            ids.append(str(def_id))
            weights.append(float(weight))
        raw_counts = bucket.get("raw_card_counts") or {}
        if not isinstance(raw_counts, dict):
            continue
        bucket_supported = 0.0
        candidate_ids = {
            def_id for ids, _ in candidates.values() for def_id in ids
        }
        unsupported += sum(
            _finite_float(value, 0.0)
            for def_id, value in raw_counts.items()
            if str(def_id) not in candidate_ids
        )
        for ids, native_weights in candidates.values():
            counts = [
                max(0.0, _finite_float(raw_counts.get(def_id), 0.0))
                for def_id in ids
            ]
            total = sum(counts)
            if total <= 0.0:
                continue
            native_probabilities = _normalized(native_weights)
            prior_probabilities, source = prior.adjusted_weights(
                mod_ids=environment_mod_ids(env),
                def_ids=ids,
                native_weights=native_weights,
            )
            native_best = max(range(len(ids)), key=native_probabilities.__getitem__)
            prior_best = max(range(len(ids)), key=prior_probabilities.__getitem__)
            for index, count in enumerate(counts):
                if count <= 0.0:
                    continue
                native_nll -= count * math.log(max(1e-12, native_probabilities[index]))
                prior_nll -= count * math.log(max(1e-12, prior_probabilities[index]))
            native_top += counts[native_best]
            prior_top += counts[prior_best]
            supported += total
            bucket_supported += total
            sources[source] += total
        if bucket_supported > 0.0:
            evaluated_buckets += 1

    native_cross_entropy = native_nll / supported if supported else 0.0
    prior_cross_entropy = prior_nll / supported if supported else 0.0
    return {
        "evaluated_buckets": evaluated_buckets,
        "supported_card_picks": int(round(supported)),
        "unsupported_card_picks": int(round(unsupported)),
        "native_cross_entropy": round(native_cross_entropy, 7),
        "prior_cross_entropy": round(prior_cross_entropy, 7),
        "cross_entropy_improvement": round(
            native_cross_entropy - prior_cross_entropy, 7
        ),
        "native_perplexity": round(math.exp(min(20.0, native_cross_entropy)), 6),
        "prior_perplexity": round(math.exp(min(20.0, prior_cross_entropy)), 6),
        "native_top1_rate": round(native_top / supported, 6) if supported else 0.0,
        "prior_top1_rate": round(prior_top / supported, 6) if supported else 0.0,
        "prior_sources": {
            key: int(round(value)) for key, value in sorted(sources.items())
        },
    }


def deck_bucket_key(mod_ids: Iterable[str]) -> str:
    normalized = sorted({str(value).strip() for value in mod_ids if str(value).strip()})
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def environment_mod_ids(env: Any) -> tuple[str, ...]:
    load_order = getattr(getattr(env, "loadout", None), "load_order", None)
    if isinstance(load_order, (list, tuple)) and load_order:
        return tuple(sorted({str(value) for value in load_order if str(value)}))
    return tuple(sorted({str(value) for value in getattr(env, "mod_filenames", ()) if str(value)}))


def augment_observation_with_deck_prior(
    observation: dict[str, Any],
    prior: DeckPrior,
    *,
    max_cards_per_type: int = 5,
    include_public_evidence: bool = False,
) -> dict[str, Any]:
    result = dict(observation)
    if include_public_evidence:
        features = prior.public_observation_features(
            observation,
            max_cards_per_type=max_cards_per_type,
        )
    else:
        loadout = observation.get("loadout") or {}
        features = prior.observation_features(
            loadout.get("official_mods") or (),
            max_cards_per_type=max_cards_per_type,
        )
    result["opponent_deck_belief"] = features
    return result


def _public_opponent_card_evidence(
    observation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    opponent = observation.get("opponent") or {}
    opponent_id = _optional_int(opponent.get("player_id"))
    if opponent_id is None:
        seat = _optional_int(observation.get("seat"))
        opponent_id = None if seat is None else 1 - seat
    if opponent_id is None:
        return {}

    evidence: dict[str, dict[str, Any]] = {}

    def record(def_id: Any, source: str) -> None:
        card_id = str(def_id or "")
        if not card_id:
            return
        item = evidence.setdefault(card_id, {"count": 0, "sources": set()})
        item["count"] += 1
        item["sources"].add(source)

    history = observation.get("public_history")
    if isinstance(history, list):
        for event in history:
            if not isinstance(event, dict):
                continue
            if _optional_int(event.get("player")) != opponent_id:
                continue
            if str(event.get("kind") or "") in {"play_card", "respond"}:
                record(event.get("card_def_id"), "history")

    for card in opponent.get("revealed_hand") or ():
        if isinstance(card, dict):
            record(card.get("def_id"), "revealed_hand")
    for zone in ("deck_ordered", "discard_ordered"):
        for card in opponent.get(zone) or ():
            if isinstance(card, dict):
                record(card.get("def_id"), zone)

    for relation in ("self", "opponent"):
        player = observation.get(relation) or {}
        for equipment in player.get("equipment") or ():
            if not isinstance(equipment, dict):
                continue
            if _optional_int(equipment.get("owner")) != opponent_id:
                continue
            card = equipment.get("card") or {}
            if isinstance(card, dict):
                record(card.get("def_id"), "equipment")

    for reveal in observation.get("temporary_reveals") or ():
        if not isinstance(reveal, dict):
            continue
        if _optional_int(reveal.get("target_player")) != opponent_id:
            continue
        for card in reveal.get("cards") or ():
            if isinstance(card, dict):
                record(card.get("def_id"), "initial_deck_reveal")
    return evidence


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bucket_mod_sets(
    buckets: dict[str, dict[str, Any]]
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for key in buckets:
        if key == GLOBAL_DECK_BUCKET:
            continue
        try:
            values = json.loads(key)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(values, list):
            result[key] = frozenset(str(value) for value in values if str(value))
    return result


def _bucket_support(bucket: dict[str, Any] | None) -> float:
    if not isinstance(bucket, dict):
        return 0.0
    return max(0.0, _finite_float(bucket.get("effective_decks"), 0.0))


def _bucket_distribution(
    bucket: dict[str, Any] | None, def_ids: Sequence[str]
) -> list[float] | None:
    values = bucket.get("card_weights") if isinstance(bucket, dict) else None
    if not isinstance(values, dict):
        return None
    weights = [max(0.0, _finite_float(values.get(def_id), 0.0)) for def_id in def_ids]
    if sum(weights) <= 0.0:
        return None
    return _normalized(weights)


def _normalized(values: Sequence[float]) -> list[float]:
    result = [max(0.0, _finite_float(value, 0.0)) for value in values]
    total = sum(result)
    if total <= 0.0:
        return [1.0 / len(result) for _ in result] if result else []
    return [value / total for value in result]


def _blend(left: Sequence[float], right: Sequence[float], amount: float) -> list[float]:
    mix = max(0.0, min(1.0, float(amount)))
    return _normalized([
        (1.0 - mix) * float(base) + mix * float(empirical)
        for base, empirical in zip(left, right)
    ])


def _saturation(support: float, prior: float) -> float:
    if support <= 0.0:
        return 0.0
    return support / (support + max(0.0, prior))


def _finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _rounded_positive_map(values: dict[str, Any]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in sorted(values.items()):
        numeric = _finite_float(value, 0.0)
        if numeric <= 0.0:
            continue
        result[str(key)] = int(numeric) if numeric.is_integer() else round(numeric, 6)
    return result


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16, person=b"GTN-deck-prior").hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".json", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an anonymous hidden-deck prior from replay aggregates"
    )
    parser.add_argument("aggregates", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-raw-decks", type=int, default=5)
    parser.add_argument("--evaluate-aggregate")
    args = parser.parse_args(argv)
    prior, report = build_deck_prior(
        args.aggregates,
        minimum_raw_decks=args.minimum_raw_decks,
    )
    prior.save(args.output)
    evaluation = (
        evaluate_deck_prior(prior, args.evaluate_aggregate)
        if args.evaluate_aggregate
        else None
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        **report,
        "evaluation": evaluation,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
