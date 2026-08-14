from __future__ import annotations

import pytest

from gtn_ai import Action, Garden1v1Env
from gtn_ai.policies import HeuristicPolicy


VANILLA = ["Vanilla Cards.gtnmod"]


def _force_event_options(env: Garden1v1Env, left: int, right: int = 1) -> None:
    event_defs = env.engine.OPENING_EVENTS
    env.engine.opening_event_options = [
        [dict(event_defs[left])],
        [dict(event_defs[right])],
    ]


def _finish_pregame(env: Garden1v1Env, *, limit: int = 200) -> list[str]:
    policies = [HeuristicPolicy(11, exploration=0), HeuristicPolicy(12, exploration=0)]
    action_kinds = []
    for _ in range(limit):
        observation = env.observe(env.decision_player(default=0))
        if observation["phase"] != "pregame":
            return action_kinds
        actor = env.decision_player()
        legal = env.legal_actions(actor)
        assert legal
        action = policies[actor].select_action(observation, legal)
        action_kinds.append(action.kind)
        env.step(action, actor)
    raise AssertionError("pregame did not finish within the decision limit")


def test_default_reset_starts_at_learnable_pregame_without_leaking_event_pick():
    env = Garden1v1Env(enabled_mods=VANILLA, seed=201)
    observation = env.reset()
    actor = env.decision_player()
    assert observation["phase"] == "pregame"
    assert observation["pregame_phase"] == "event_select"
    assert len(observation["self"]["opening_event_options"]) == 3

    env.step(Action("select_opening_event", {"option_slot": 0}), actor)
    other = 1 - actor
    hidden_view = env.observe(other)
    assert hidden_view["opponent"]["event_selected"] is True
    assert "opening_event" not in hidden_view["opponent"]
    assert "draft_target" not in hidden_view["opponent"]

    env.step(Action("select_opening_event", {"option_slot": 0}), other)
    revealed_view = env.observe(actor)
    assert "opening_event" in revealed_view["opponent"]
    assert "draft_target" in revealed_view["opponent"]


def test_fated_draw_target_count_cannot_reveal_a_hidden_event():
    env = Garden1v1Env(enabled_mods=VANILLA, seed=2011)
    env.reset()
    actor = env.decision_player()
    other = 1 - actor
    _force_event_options(env, 5 if actor == 0 else 1, 5 if actor == 1 else 1)
    env.step(Action("select_opening_event", {"option_slot": 0}), actor)

    hidden_view = env.observe(other)
    assert hidden_view["opponent"]["event_selected"] is True
    assert "opening_event" not in hidden_view["opponent"]
    assert "draft_target" not in hidden_view["opponent"]


def test_opponent_draft_cards_remain_private_while_progress_is_public():
    env = Garden1v1Env(enabled_mods=VANILLA, seed=202)
    env.reset()
    _force_event_options(env, 1, 1)
    for _ in range(4):
        actor = env.decision_player()
        env.step(env.legal_actions(actor)[0], actor)

    actor = env.decision_player()
    assert env.pregame_status(actor) == "drafting"
    env.step(next(action for action in env.legal_actions(actor) if action.kind == "draft_pick"), actor)
    opponent_view = env.observe(1 - actor)
    assert opponent_view["opponent"]["draft_count"] == 1
    assert "draft_picks" not in opponent_view["opponent"]


@pytest.mark.parametrize("event_id", [2, 3, 5, 8, 11])
def test_every_builtin_post_draft_choice_completes_through_atomic_actions(event_id: int):
    env = Garden1v1Env(enabled_mods=VANILLA, seed=210 + event_id)
    env.reset()
    _force_event_options(env, event_id, 1)
    action_kinds = _finish_pregame(env)

    assert env.engine.phase == "action"
    assert env.engine.player_ready == [True, True]
    assert len(env.engine.draft_picks[0]) == env.engine.draft_target_count(0)
    assert "submit_pregame_choice" in action_kinds or "select_pregame_choice" in action_kinds
    sub_choice = env.engine.opening_event_sub_choices[0]
    assert isinstance(sub_choice, dict)
    if event_id == 2:
        assert len(sub_choice["conversions"]) <= 3
    elif event_id == 3:
        assert len(sub_choice["convert_def_ids"]) <= 5
    elif event_id == 5:
        assert len(sub_choice["add_def_ids"]) == 1
        assert len(env.engine.draft_picks[0]) == 14
    elif event_id == 8:
        assert sub_choice["yggdrasil_convert_def_id"]
    elif event_id == 11:
        assert len(sub_choice["deck_order_def_ids"]) == 15


def test_pregame_observation_is_read_only_for_card_instance_sequence():
    env = Garden1v1Env(enabled_mods=VANILLA, seed=221)
    env.reset()
    _force_event_options(env, 11, 1)
    policies = [HeuristicPolicy(21, exploration=0), HeuristicPolicy(22, exploration=0)]
    while env.pregame_status(0) != "sub_choice":
        actor = env.decision_player()
        observation = env.observe(actor)
        env.step(policies[actor].select_action(observation, env.legal_actions(actor)), actor)
    import cards

    before = cards._next_instance_id
    for viewer in (0, 1, 0, 1):
        env.observe(viewer)
    assert cards._next_instance_id == before


def test_pregame_clone_replays_random_draft_generation_identically():
    env = Garden1v1Env(enabled_mods=VANILLA, seed=222)
    env.reset()
    _force_event_options(env, 1, 1)
    for _ in range(2):
        actor = env.decision_player()
        env.step(env.legal_actions(actor)[0], actor)

    clone = env.clone()
    actor = env.decision_player()
    action = Action("confirm_opening_reveal")
    left = env.step(action, actor).observation
    right = clone.step(action, actor).observation
    assert left == right


def test_fast_combat_only_reset_remains_available_for_engine_regressions():
    env = Garden1v1Env(enabled_mods=VANILLA, seed=223, include_pregame=False)
    observation = env.reset()
    assert observation["phase"] == "action"
    assert all(len(picks) == 15 for picks in env.engine.draft_picks)


def test_episode_records_pregame_metadata_and_split_step_counts():
    from gtn_ai.self_play import play_episode

    env = Garden1v1Env(enabled_mods=VANILLA, seed=224)
    episode = play_episode(
        env,
        [HeuristicPolicy(31, exploration=0), HeuristicPolicy(32, exploration=0)],
        max_steps=1500,
    )
    assert episode.terminated
    assert episode.pregame_steps > 0
    assert episode.combat_steps > 0
    assert episode.steps == episode.pregame_steps + episode.combat_steps
    assert len(episode.opening_events) == 2
    assert [len(deck) for deck in episode.drafted_decks] in ([15, 15], [14, 15], [15, 14], [14, 14])
    assert episode.first_player in (0, 1)
