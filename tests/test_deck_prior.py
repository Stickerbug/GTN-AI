from __future__ import annotations

import json

from gtn_ai.belief_sampling import determinize_hidden_cards
from gtn_ai.deck_prior import (
    DeckBeliefPolicy,
    DeckPrior,
    DeckPriorConfig,
    build_deck_prior,
    deck_bucket_key,
    environment_mod_ids,
)
from gtn_ai.environment import Garden1v1Env
from gtn_ai.historical_aggregate import HISTORICAL_AGGREGATE_SCHEMA_VERSION


class _RecordingPolicy:
    name = "recording"
    ruleset_fingerprint = "rules"
    model_fingerprint = "model"
    offline_only = False
    uses_private_engine_state = False

    def __init__(self) -> None:
        self.observations = []
        self.last_decision_metadata = None

    def select_action(self, observation, legal_actions):
        self.observations.append(observation)
        return legal_actions[0]

    def evaluate_actions(self, observation, legal_actions):
        self.observations.append(observation)
        return [0.0] * len(legal_actions), 0.0

    def estimate_value(self, observation, legal_actions):
        self.observations.append(observation)
        return 0.0

    def estimate_values(self, decisions):
        self.observations.extend(observation for observation, _ in decisions)
        return [0.0] * len(decisions)

    def fork(self, **_kwargs):
        return _RecordingPolicy()

    def diagnostics(self):
        return {"decisions": len(self.observations)}


def test_deck_prior_builds_an_exact_and_global_backoff(tmp_path) -> None:
    aggregate = {
        "schema_version": HISTORICAL_AGGREGATE_SCHEMA_VERSION,
        "deck_statistics": {
            "*": {
                "effective_decks": 200,
                "raw_decks": 300,
                "card_weights": {"a": 20, "b": 80},
                "raw_card_counts": {"a": 30, "b": 120},
            },
            deck_bucket_key(["vanilla"]): {
                "effective_decks": 100,
                "raw_decks": 150,
                "card_weights": {"a": 95, "b": 5},
                "raw_card_counts": {"a": 140, "b": 10},
            },
        },
    }
    source = tmp_path / "aggregate.json"
    source.write_text(json.dumps(aggregate), encoding="utf-8")

    prior, report = build_deck_prior([source])
    exact, exact_source = prior.adjusted_weights(
        mod_ids=["vanilla"], def_ids=["a", "b"], native_weights=[1, 1]
    )
    fallback, fallback_source = prior.adjusted_weights(
        mod_ids=["unknown"], def_ids=["a", "b"], native_weights=[1, 1]
    )

    assert report["retained_buckets"] == 2
    assert exact_source == "exact"
    assert exact[0] > exact[1]
    assert fallback_source == "global"
    assert fallback[1] > fallback[0]

    related_prior = DeckPrior(
        {
            deck_bucket_key(["garden", "vanilla"]): {
                "effective_decks": 100,
                "raw_decks": 100,
                "card_weights": {"a": 90, "b": 10},
            }
        },
        config=DeckPriorConfig(
            min_global_decks=1000,
            min_exact_decks=1,
            related_max_mix=1,
            min_related_similarity=0.1,
        ),
    )
    related, related_source = related_prior.adjusted_weights(
        mod_ids=["vanilla"], def_ids=["a", "b"], native_weights=[1, 1]
    )
    assert related_source == "related"
    assert related[0] > related[1]

    path = tmp_path / "prior.json"
    prior.save(path)
    loaded = DeckPrior.load(path)
    assert loaded.fingerprint == prior.fingerprint
    assert loaded.adjusted_weights(
        mod_ids=["vanilla"], def_ids=["a", "b"], native_weights=[1, 1]
    ) == (exact, exact_source)


def test_empirical_belief_sample_preserves_public_observation() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=2901,
        include_pregame=False,
    )
    env.reset()
    actor = env.decision_player()
    opponent = 1 - actor
    before = env.observe(actor)
    from cards import CARD_DEFS

    by_type: dict[str, list[str]] = {}
    for def_id in sorted(env.allowed_card_ids):
        card = CARD_DEFS.get(def_id)
        if card is not None and int(getattr(card, "count", 0) or 0) > 0:
            by_type.setdefault(str(card.card_type), []).append(str(def_id))
    preferred = next(values[0] for values in by_type.values() if len(values) >= 2)
    prior = DeckPrior(
        {
            deck_bucket_key(environment_mod_ids(env)): {
                "effective_decks": 100,
                "raw_decks": 100,
                "card_weights": {preferred: 100},
                "raw_card_counts": {preferred: 100},
            }
        },
        config=DeckPriorConfig(
            min_exact_decks=1,
            min_global_decks=1,
            exact_prior_decks=0,
            exact_max_mix=1,
            global_max_mix=0,
        ),
    )

    summary = determinize_hidden_cards(
        env, actor, seed=29011, deck_prior=prior
    )
    hidden_defs = {
        str(card.def_id)
        for zone_name in ("hand", "deck", "discard", "exile")
        for card in getattr(env.engine.players[opponent], zone_name)
    }

    assert env.observe(actor) == before
    assert summary.deck_prior_source == "exact"
    assert summary.deck_prior_sampled_cards == summary.sampled_cards
    assert preferred in hidden_defs


def test_deck_belief_policy_only_augments_public_observation() -> None:
    env = Garden1v1Env(
        enabled_mods=["Vanilla Cards.gtnmod"],
        seed=2902,
        include_pregame=False,
    )
    observation = env.reset()
    legal = env.legal_actions(env.decision_player())
    base = _RecordingPolicy()
    policy = DeckBeliefPolicy(base, DeckPrior({}))

    selected = policy.select_action(observation, legal)
    augmented = base.observations[-1]

    assert selected in legal
    assert "opponent_deck_belief" not in observation
    assert augmented["opponent_deck_belief"]
    assert all(
        set(card) == {
            "def_id", "card_type", "probability", "expected_count", "rank", "source"
        }
        for card in augmented["opponent_deck_belief"]
    )
    assert policy.diagnostics()["deck_belief_fingerprint"] == policy.prior.fingerprint
