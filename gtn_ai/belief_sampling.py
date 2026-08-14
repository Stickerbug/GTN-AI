from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

from .deck_prior import DeckPrior, environment_mod_ids
from .environment import Garden1v1Env


class BeliefSamplingError(RuntimeError):
    pass


@dataclass(frozen=True)
class BeliefSampleSummary:
    sampled_cards: int
    preserved_cards: int
    attempts: int
    history_constrained_cards: int = 0
    deck_prior_sampled_cards: int = 0
    deck_prior_source: str = "native"


def determinize_hidden_cards(
    env: Garden1v1Env,
    viewer_id: int,
    *,
    seed: int,
    max_attempts: int = 8,
    deck_prior: DeckPrior | None = None,
) -> BeliefSampleSummary:
    """Replace private opponent hand/deck cards without changing public observation.

    This is deliberately conservative. A sample is accepted only when the complete
    observation visible to ``viewer_id`` remains byte-for-byte equivalent as a Python
    value. Failed attempts restore the untouched hidden state.
    """

    if viewer_id not in (0, 1):
        raise ValueError("viewer_id must be 0 or 1")
    if env.engine is None or env.engine.game_over:
        raise BeliefSamplingError("cannot determinize an inactive environment")
    opponent_id = 1 - int(viewer_id)
    engine = env.engine
    opponent = engine.players[opponent_id]
    expected_observation = copy.deepcopy(env.observe(viewer_id))
    visible_ids = _public_instance_ids(engine.get_public_state(viewer_id))
    zone_names = ("hand", "deck", "discard", "exile")
    original_zones = {
        zone_name: copy.deepcopy(getattr(opponent, zone_name))
        for zone_name in zone_names
    }
    original_draft = copy.deepcopy(engine.draft_picks[opponent_id])
    original_initial_deck = copy.deepcopy(engine._garden_initial_decks[opponent_id])
    publicly_preserved_defs = {
        str(card.def_id)
        for cards in original_zones.values()
        for card in cards
        if int(card.instance_id) in visible_ids
    }
    publicly_preserved_defs.update(
        str(equipment.card_instance.def_id)
        for equipment in opponent.equipment
    )
    history_requirements = _public_history_card_requirements(
        env,
        opponent_id,
        already_public_defs=publicly_preserved_defs,
    )
    hidden_slots = [
        (zone_name, index, int(card.instance_id))
        for zone_name in zone_names
        for index, card in enumerate(original_zones[zone_name])
        if int(card.instance_id) not in visible_ids
    ]
    if not hidden_slots:
        return BeliefSampleSummary(sampled_cards=0, preserved_cards=0, attempts=1)

    reveal = getattr(engine, "_garden_initial_deck_reveal", [None, None])[viewer_id]
    initial_deck_is_public = bool(
        isinstance(reveal, dict)
        and int(reveal.get("target_player_id", -1)) == opponent_id
    )
    for attempt in range(max(1, int(max_attempts))):
        for zone_name, cards in original_zones.items():
            setattr(opponent, zone_name, copy.deepcopy(cards))
        engine.draft_picks[opponent_id] = copy.deepcopy(original_draft)
        engine._garden_initial_decks[opponent_id] = copy.deepcopy(original_initial_deck)
        rng = random.Random(int(seed) + attempt * 104729)
        initial_defs, initial_source = _sample_formal_deck_def_ids(
            env, rng, 15, deck_prior=deck_prior
        )
        sampled_defs, sampled_source = _sample_formal_deck_def_ids(
            env, rng, len(hidden_slots), deck_prior=deck_prior
        )
        initial_defs, _ = _apply_card_requirements(initial_defs, history_requirements, rng)
        sampled_defs, constrained_cards = _apply_card_requirements(
            sampled_defs,
            history_requirements,
            rng,
        )
        for (zone_name, index, instance_id), def_id in zip(hidden_slots, sampled_defs):
            from cards import CardInstance

            zone = getattr(opponent, zone_name)
            zone[index] = CardInstance(def_id=def_id, instance_id=instance_id)
        if not initial_deck_is_public:
            engine.draft_picks[opponent_id] = list(initial_defs)
            from cards import CardInstance

            engine._garden_initial_decks[opponent_id] = [
                CardInstance(def_id=def_id, instance_id=-(index + 1)).to_dict()
                for index, def_id in enumerate(initial_defs)
            ]
        if env.observe(viewer_id) == expected_observation:
            return BeliefSampleSummary(
                sampled_cards=len(hidden_slots),
                preserved_cards=(
                    sum(len(cards) for cards in original_zones.values()) - len(hidden_slots)
                ),
                attempts=attempt + 1,
                history_constrained_cards=constrained_cards,
                deck_prior_sampled_cards=(
                    len(hidden_slots) if sampled_source != "native" else 0
                ),
                deck_prior_source=_stronger_prior_source(
                    initial_source, sampled_source
                ),
            )

    for zone_name, cards in original_zones.items():
        setattr(opponent, zone_name, cards)
    engine.draft_picks[opponent_id] = original_draft
    engine._garden_initial_decks[opponent_id] = original_initial_deck
    raise BeliefSamplingError(
        f"could not sample a public-observation-equivalent hidden state in {max_attempts} attempts"
    )


def _sample_formal_deck_def_ids(
    env: Garden1v1Env,
    rng: random.Random,
    count: int,
    *,
    deck_prior: DeckPrior | None = None,
) -> tuple[list[str], str]:
    from cards import CARD_DEFS, DRAFT_RATIO, _effective_draft_weights, normalize_card_flags

    target = max(0, int(count))
    if target == 0:
        return [], "native"
    effective = _effective_draft_weights(set(env.allowed_card_ids))
    by_type: dict[str, tuple[list[str], list[float]]] = {}
    for def_id, weight in effective.items():
        card_def = CARD_DEFS.get(def_id)
        if card_def is None or float(weight) <= 0:
            continue
        flags = normalize_card_flags(getattr(card_def, "flags", set()) or set())
        if "team_limited" in flags:
            continue
        ids, weights = by_type.setdefault(str(card_def.card_type), ([], []))
        ids.append(str(def_id))
        weights.append(float(weight))
    type_cycle = [
        card_type
        for card_type, quota in DRAFT_RATIO.items()
        for _ in range(max(0, int(quota)))
        if card_type in by_type
    ]
    if not type_cycle:
        raise BeliefSamplingError("current loadout has no weighted draft cards")
    prior_sources: set[str] = set()
    if deck_prior is not None:
        mod_ids = environment_mod_ids(env)
        for card_type, (ids, weights) in tuple(by_type.items()):
            adjusted, source = deck_prior.adjusted_weights(
                mod_ids=mod_ids,
                def_ids=ids,
                native_weights=weights,
            )
            by_type[card_type] = (ids, adjusted)
            prior_sources.add(source)
    types: list[str] = []
    while len(types) < target:
        cycle = list(type_cycle)
        rng.shuffle(cycle)
        types.extend(cycle)
    result = []
    for card_type in types[:target]:
        ids, weights = by_type[card_type]
        result.append(rng.choices(ids, weights=weights, k=1)[0])
    source = (
        "exact" if "exact" in prior_sources
        else "related" if "related" in prior_sources
        else "global" if "global" in prior_sources
        else "native"
    )
    return result, source


def _stronger_prior_source(left: str, right: str) -> str:
    rank = {"native": 0, "global": 1, "related": 2, "exact": 3}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def _public_history_card_requirements(
    env: Garden1v1Env,
    player_id: int,
    *,
    already_public_defs: set[str],
) -> list[str]:
    """Return recent public card identities that a hidden sample should retain.

    A single requirement is kept per definition. Repeated plays may be the same
    physical card, so treating every play as another copy would over-constrain the
    sample. Most-recent-first ordering gives scarce type slots to fresher evidence.
    """

    from cards import CARD_DEFS

    requirements: list[str] = []
    seen = set(already_public_defs)
    for event in reversed(env.public_history):
        if not isinstance(event, dict) or int(event.get("player", -1)) != int(player_id):
            continue
        if str(event.get("kind") or "") not in {"play_card", "respond"}:
            continue
        def_id = str(event.get("card_def_id") or "")
        if not def_id or def_id in seen or def_id not in CARD_DEFS:
            continue
        seen.add(def_id)
        requirements.append(def_id)
    return requirements


def _apply_card_requirements(
    sampled_defs: list[str],
    required_defs: list[str],
    rng: random.Random,
) -> tuple[list[str], int]:
    """Inject public card evidence without changing the sampled type histogram."""

    from cards import CARD_DEFS

    result = list(sampled_defs)
    available_by_type: dict[str, list[int]] = {}
    for index, def_id in enumerate(result):
        card_def = CARD_DEFS.get(def_id)
        if card_def is not None:
            available_by_type.setdefault(str(card_def.card_type), []).append(index)
    constrained = 0
    for def_id in required_defs:
        card_def = CARD_DEFS.get(def_id)
        if card_def is None:
            continue
        candidates = available_by_type.get(str(card_def.card_type)) or []
        if not candidates:
            continue
        slot_offset = rng.randrange(len(candidates))
        slot = candidates.pop(slot_offset)
        result[slot] = def_id
        constrained += 1
    return result, constrained


def _public_instance_ids(value: Any) -> set[int]:
    result: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            instance_id = item.get("instance_id")
            if isinstance(instance_id, int):
                result.add(int(instance_id))
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return result
