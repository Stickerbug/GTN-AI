from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .game_imports import (
    apply_loadout_to_engine,
    configure_game_imports,
    load_official_content,
    official_ruleset_fingerprint,
)
from .observation import build_observation
from .protocol import Action
from .random_scope import EngineRuntimeScope


class IllegalActionError(ValueError):
    pass


class UnsupportedDecisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    info: dict[str, Any]


class Garden1v1Env:
    """Version-locked headless adapter around the production formal 1v1 engine."""

    def __init__(
        self,
        *,
        game_root=None,
        enabled_mods: Iterable[str] | None = None,
        seed: int = 0,
        deck_size: int = 15,
        opening_events: Sequence[int] = (1, 1),
        history_limit: int = 128,
        include_pregame: bool = True,
    ):
        self.game_root = configure_game_imports(game_root)
        self.enabled_mods = None if enabled_mods is None else tuple(enabled_mods)
        self.seed = int(seed)
        self.deck_size = max(0, int(deck_size))
        if len(opening_events) != 2:
            raise ValueError("opening_events must contain one event per player")
        self.opening_events = (int(opening_events[0]), int(opening_events[1]))
        self.history_limit = max(1, int(history_limit))
        self.include_pregame = bool(include_pregame)
        self.runtime = EngineRuntimeScope(self.seed)
        self.engine = None
        self.loadout = None
        self.allowed_card_ids: frozenset[str] = frozenset()
        self.mod_filenames: tuple[str, ...] = ()
        self.loadout_fingerprint = ""
        self.ruleset_fingerprint = ""
        self.public_history: list[dict[str, Any]] = []
        self._choice_selection: list[int] = []
        self._choice_candidate_ids: list[int] = []
        self._choice_signature: tuple[Any, ...] | None = None
        self._pregame_cursor = self.seed & 1
        self._pregame_selections: list[list[int]] = [[], []]

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        self.runtime = EngineRuntimeScope(self.seed)
        with self.runtime.activate():
            self.loadout, self.allowed_card_ids, self.mod_filenames = load_official_content(
                self.game_root,
                self.enabled_mods,
            )
            self.loadout_fingerprint = str(getattr(self.loadout, "loadout_hash", "") or "")[:16]
            self.ruleset_fingerprint = official_ruleset_fingerprint(self.game_root)
            from cards import DECK_SIZE
            from game_engine import GameEngine

            if self.deck_size != DECK_SIZE:
                raise ValueError(
                    f"formal 1v1 uses exactly {DECK_SIZE} drafted cards; got {self.deck_size}"
                )
            engine = GameEngine()
            apply_loadout_to_engine(engine, self.loadout, self.allowed_card_ids)
            engine.player_names = ["AI-0", "AI-1"]
            if self.include_pregame:
                engine.start_event_select_first()
            else:
                self._populate_random_formal_drafts(engine)
                if not engine.start_game(skip_pregame_validation=True):
                    raise RuntimeError("game engine refused deterministic reset")
            self.engine = engine
        self.public_history = []
        self._choice_selection = []
        self._choice_candidate_ids = []
        self._choice_signature = None
        self._pregame_cursor = self.seed & 1
        self._pregame_selections = [[], []]
        return self.observe(self.decision_player(default=0))

    def clone(self) -> "Garden1v1Env":
        clone = self.__class__.__new__(self.__class__)
        clone.game_root = self.game_root
        clone.enabled_mods = self.enabled_mods
        clone.seed = self.seed
        clone.deck_size = self.deck_size
        clone.opening_events = self.opening_events
        clone.history_limit = self.history_limit
        clone.include_pregame = self.include_pregame
        clone.runtime = copy.deepcopy(self.runtime)
        clone.engine = copy.deepcopy(self.engine)
        clone.loadout = self.loadout
        clone.allowed_card_ids = self.allowed_card_ids
        clone.mod_filenames = self.mod_filenames
        clone.loadout_fingerprint = self.loadout_fingerprint
        clone.ruleset_fingerprint = self.ruleset_fingerprint
        clone.public_history = copy.deepcopy(self.public_history)
        clone._choice_selection = list(self._choice_selection)
        clone._choice_candidate_ids = list(self._choice_candidate_ids)
        # pending_choice itself is deep-copied, so its identity-based signature
        # must be rebuilt without treating the existing selection as a new dialog.
        clone._choice_signature = clone._pending_choice_signature()
        clone._pregame_cursor = self._pregame_cursor
        clone._pregame_selections = copy.deepcopy(self._pregame_selections)
        return clone

    def decision_player(self, *, default: int | None = None) -> int | None:
        if self.engine is None:
            return default
        engine = self.engine
        if engine.game_over:
            return None
        if self._in_pregame():
            return self._pregame_decision_player()
        pending = engine.pending_response
        if isinstance(pending, dict):
            source = _as_int(pending.get("player_id"), -1)
            target = _as_int(pending.get("target_player_id"), -1)
            if target in (0, 1) and target != source:
                return target
            return 1 - source if source in (0, 1) else default
        pending = engine.pending_choice
        if isinstance(pending, dict):
            actor = _as_int(pending.get("player_id"), -1)
            return actor if actor in (0, 1) else default
        pending = engine.pending_v2_ui
        if isinstance(pending, dict):
            actor = _as_int(pending.get("player_id"), -1)
            return actor if actor in (0, 1) else default
        if engine.phase in {"action", "draw", "playing"} and engine.current_player in (0, 1):
            return int(engine.current_player)
        return default

    def observe(self, player_id: int | None = None) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("reset() must be called first")
        if player_id is None:
            player_id = self.decision_player(default=0)
        if player_id is None:
            player_id = 0
        return build_observation(self, int(player_id))

    def _populate_random_formal_drafts(self, engine) -> None:
        """Simulate uniform player picks from the production three-card draft UI."""

        engine.phase = "event_select"
        engine.opening_event_picks = list(self.opening_events)
        engine.opening_event_sub_choices = [None, None]
        for player_id in (0, 1):
            if not engine.start_draft_for_player(player_id):
                raise RuntimeError(f"could not start formal draft for player {player_id}")

        while True:
            unfinished = [
                player_id
                for player_id in (0, 1)
                if len(engine.draft_picks[player_id]) < engine.draft_target_count(player_id)
            ]
            if not unfinished:
                break
            for player_id in unfinished:
                options = list(engine.draft_options[player_id] or [])
                if not options:
                    engine._generate_draft_options_for_player(player_id)
                    options = list(engine.draft_options[player_id] or [])
                if not options:
                    raise RuntimeError(
                        f"official loadout has no draft option for player {player_id}"
                    )
                selected = random.choice(options)
                if not engine.draft_pick(player_id, selected.def_id):
                    raise RuntimeError(
                        f"production engine rejected offered draft card {selected.def_id}"
                    )
        for player_id in (0, 1):
            if engine.needs_sub_choice(player_id):
                engine.opening_event_sub_choices[player_id] = self._default_pregame_sub_choice(
                    engine,
                    player_id,
                )
            engine.player_ready[player_id] = True

    def _in_pregame(self) -> bool:
        return bool(
            self.include_pregame
            and self.engine is not None
            and self.engine.phase in {"event_select", "event_reveal", "draft"}
            and not self.engine.game_over
        )

    def _pregame_decision_player(self) -> int | None:
        if not self._in_pregame():
            return None
        all_events_selected = all(pick is not None for pick in self.engine.opening_event_picks)
        for offset in range(2):
            player_id = (self._pregame_cursor + offset) % 2
            status = self.pregame_status(player_id)
            if status == "event_reveal" and not all_events_selected:
                continue
            if status in {"event_select", "event_reveal", "drafting", "sub_choice"}:
                return player_id
        return None

    def pregame_status(self, player_id: int) -> str:
        engine = self.engine
        if engine.player_ready[player_id]:
            return "ready"
        if engine.opening_event_picks[player_id] is None:
            return "event_select"
        if not engine.player_draft_started[player_id]:
            return "event_reveal"
        if len(engine.draft_picks[player_id]) < engine.draft_target_count(player_id):
            return "drafting"
        if engine.needs_sub_choice(player_id):
            return "sub_choice"
        return "ready"

    def pregame_selection_view(self, player_id: int) -> dict[str, Any]:
        selected = list(self._pregame_selections[player_id])
        return {"selected_slots": selected, "ordered_slots": selected}

    def pregame_candidate_def_ids(self, player_id: int) -> list[str]:
        if not self._in_pregame() or self.pregame_status(player_id) != "sub_choice":
            return []
        engine = self.engine
        event_id = str(engine.opening_event_picks[player_id])
        picks = [str(def_id) for def_id in engine.draft_picks[player_id]]
        if event_id == "2":
            from cards import CARD_DEFS, normalize_card_flags

            return [
                def_id for def_id in picks
                if "sublime" not in normalize_card_flags(CARD_DEFS[def_id].flags)
            ]
        if event_id == "3":
            return [def_id for def_id in picks if engine._opening_light_source_allowed(def_id)]
        if event_id == "5":
            return [str(def_id) for def_id in engine.fated_draw_pool_defs()]
        if event_id == "8":
            from cards import CARD_DEFS, normalize_card_flags

            return [
                def_id for def_id in picks
                if "sublime" not in normalize_card_flags(CARD_DEFS[def_id].flags)
            ]
        if event_id == "11":
            from cards import CARD_DEFS, normalize_card_flags

            return [
                def_id for def_id in picks
                if def_id in CARD_DEFS
                and "sublime" not in normalize_card_flags(CARD_DEFS[def_id].flags)
            ]
        return []

    @staticmethod
    def _default_pregame_sub_choice(engine, player_id: int) -> dict[str, Any]:
        event_id = str(engine.opening_event_picks[player_id])
        picks = [str(def_id) for def_id in engine.draft_picks[player_id]]
        if event_id == "3":
            return {
                "convert_def_ids": [
                    def_id for def_id in picks
                    if engine._opening_light_source_allowed(def_id)
                ][:5]
            }
        if event_id == "5":
            pool = list(engine.fated_draw_pool_defs())
            return {"add_def_ids": [str(pool[0])]} if pool else {}
        if event_id == "8":
            from cards import CARD_DEFS, normalize_card_flags

            candidate = next((
                def_id for def_id in picks
                if "sublime" not in normalize_card_flags(CARD_DEFS[def_id].flags)
            ), None)
            return {"yggdrasil_convert_def_id": candidate} if candidate else {}
        if event_id == "11":
            from cards import CARD_DEFS, normalize_card_flags

            return {"deck_order_def_ids": [
                def_id for def_id in picks
                if def_id in CARD_DEFS
                and "sublime" not in normalize_card_flags(CARD_DEFS[def_id].flags)
            ]}
        return {}

    def legal_actions(self, player_id: int | None = None) -> list[Action]:
        if self.engine is None:
            raise RuntimeError("reset() must be called first")
        if self.engine.game_over:
            return []
        actor = self.decision_player()
        if player_id is None:
            player_id = actor
        if player_id not in (0, 1) or actor != player_id:
            return []
        if self._in_pregame():
            return self._pregame_actions(player_id)
        if self.engine.pending_response is not None:
            return self._response_actions(player_id)
        if self.engine.pending_choice is not None:
            return self._choice_actions(player_id)
        if self.engine.pending_v2_ui is not None:
            raise UnsupportedDecisionError("an official mod opened an unsupported v2 UI decision")
        return self._turn_actions(player_id)

    def _pregame_actions(self, player_id: int) -> list[Action]:
        engine = self.engine
        status = self.pregame_status(player_id)
        if status == "event_select":
            actions = [
                Action("select_opening_event", {"option_slot": slot})
                for slot, option in enumerate(engine.opening_event_options[player_id] or [])
                if option
            ]
            if engine.draft_rerolls[player_id] > 0:
                actions.append(Action("reroll_opening_event"))
            return actions
        if status == "event_reveal":
            if not all(pick is not None for pick in engine.opening_event_picks):
                return []
            return [Action("confirm_opening_reveal")]
        if status == "drafting":
            actions = [
                Action("draft_pick", {"candidate_slot": slot})
                for slot, card in enumerate(engine.draft_options[player_id] or [])
                if card is not None
            ]
            if engine.draft_rerolls[player_id] > 0:
                actions.append(Action("draft_reroll"))
            return actions
        if status != "sub_choice":
            return []

        event_id = str(engine.opening_event_picks[player_id])
        candidates = self.pregame_candidate_def_ids(player_id)
        selected = self._pregame_selections[player_id]
        if event_id in {"2", "3"}:
            maximum = 3 if event_id == "2" else 5
            actions = []
            for slot in range(len(candidates)):
                if slot in selected or len(selected) < maximum:
                    actions.append(Action("toggle_pregame_choice", {"candidate_slot": slot}))
            actions.append(Action("submit_pregame_choice"))
            return actions
        if event_id in {"5", "8"}:
            return [
                Action("select_pregame_choice", {"candidate_slot": slot})
                for slot in range(len(candidates))
            ]
        if event_id == "11":
            actions = [
                Action("append_pregame_order", {"candidate_slot": slot})
                for slot in range(len(candidates))
                if slot not in selected
            ]
            if selected:
                actions.append(Action("reset_pregame_order"))
            if len(selected) == len(candidates):
                actions.append(Action("submit_pregame_choice"))
            return actions
        raise UnsupportedDecisionError(
            f"opening event {event_id!r} requires an unsupported pregame choice"
        )

    def step(self, action: Action | dict[str, Any], player_id: int | None = None) -> StepResult:
        if not isinstance(action, Action):
            action = Action.from_dict(action)
        actor = self.decision_player()
        if player_id is None:
            player_id = actor
        if actor is None or player_id != actor:
            raise IllegalActionError("this player does not currently own a decision")
        legal = {candidate.key: candidate for candidate in self.legal_actions(player_id)}
        if action.key not in legal:
            raise IllegalActionError(f"action is not legal now: {action.key}")

        before_choice = self._pending_choice_signature()
        before_log_count = len(self.engine.log)
        public_context = self._public_action_context(action, int(player_id))
        with self.runtime.activate():
            result = self._execute(action, int(player_id))
        from runtime_errors import MOD_RUNTIME_ERROR_MESSAGE
        if MOD_RUNTIME_ERROR_MESSAGE in self.engine.log[before_log_count:]:
            raise UnsupportedDecisionError(
                "production engine reported a mod runtime error while executing a legal action"
            )
        accepted = isinstance(result, dict) and (result.get("success") or result.get("cancelled"))
        if not accepted:
            error = result.get("error") if isinstance(result, dict) else result
            raise IllegalActionError(f"engine rejected a legal action: {error}")
        self._record_public_action(action, int(player_id), result, public_context)
        if before_choice != self._pending_choice_signature():
            self._reset_choice_builder()

        rewards = self._terminal_rewards()
        next_actor = self.decision_player(default=int(player_id))
        if next_actor is None:
            next_actor = int(player_id)
        observation = self.observe(next_actor)
        return StepResult(
            observation=observation,
            reward=float(rewards[int(player_id)]),
            terminated=bool(self.engine.game_over),
            info={
                "acting_player": int(player_id),
                "next_player": self.decision_player(),
                "rewards": rewards,
                "engine_result": _public_engine_result(result),
            },
        )

    def choice_state_view(self) -> dict[str, Any]:
        self._sync_choice_builder()
        return {
            "selected_slots": [
                self._choice_candidate_ids.index(instance_id)
                for instance_id in self._choice_selection
                if instance_id in self._choice_candidate_ids
            ]
        }

    def choice_candidate_cards(self) -> list[Any]:
        pending = getattr(self.engine, "pending_choice", None)
        actor = self.decision_player()
        if not isinstance(pending, dict) or actor not in (0, 1):
            return []
        cards = []
        self._sync_choice_builder()
        for instance_id in self._choice_candidate_ids:
            card = self._choice_candidate_card(instance_id, pending)
            if card is not None:
                cards.append(card)
            else:
                cards.append(None)
        return cards

    def _turn_actions(self, player_id: int) -> list[Action]:
        engine = self.engine
        player = engine.players[player_id]
        actions: list[Action] = []
        for hand_slot, card in enumerate(list(player.hand)):
            playable, _ = engine.can_play_card(player_id, card)
            if not playable:
                continue
            for choice in self._play_target_choices(player_id, card):
                payload = {"hand_slot": int(hand_slot)}
                if choice:
                    payload["choice"] = choice
                actions.append(Action("play_card", payload))
        for equipment_slot, equipment in enumerate(list(player.equipment)):
            actions.extend(self._trigger_actions(player_id, equipment, equipment_slot))
        actions.append(Action("end_turn"))
        return _unique_actions(actions)

    def _play_target_choices(self, player_id: int, card) -> list[dict[str, int] | None]:
        engine = self.engine
        flags = engine._effective_card_flags(card)
        if "wide_strike" in flags or engine._card_is_self_only(card):
            return [None] if self._play_choice_can_progress(player_id, card, None) else []
        requires_target = (
            card.card_type == "thorn"
            or engine._v2_play_requires_choice_target(card)
            or engine._root_play_requires_owner_target(card)
        )
        if not requires_target:
            return [None] if self._play_choice_can_progress(player_id, card, None) else []
        allow_self = card.card_type != "thorn" or "self_target" in flags
        result = []
        for target_id in (0, 1):
            if not engine._target_can_be_selected(player_id, target_id, allow_self=allow_self):
                continue
            choice = _target_choice(target_id)
            if self._play_choice_can_progress(player_id, card, choice):
                result.append(choice)
        return result

    def _play_choice_can_progress(self, player_id: int, card, choice: dict[str, Any] | None) -> bool:
        """Reject a play/target that must open an impossible mandatory picker."""
        engine = self.engine
        request = engine._get_choice_request(card, choice)
        if not isinstance(request, dict):
            return True
        params = engine._effect_params(request)
        if params.get("continue_on_cancel"):
            return True
        choice_type = str(engine._choice_type_for_effect(request, card) or "")
        picker_types = {
            "choose_attack_from_hand",
            "choose_card_from_hand",
            "choose_card_to_discard",
            "choose_cards_from_hand",
            "choose_same_attacks_from_hand",
            "choose_cards_from_discard",
            "choose_ocean_sapphire",
            "choose_arctic_ruby",
            "choose_from_enemy_hand",
            "choose_from_deck",
            "choose_from_discard",
            "choose_card_from_discard",
            "choose_from_exile",
            "choose_equipment",
            "choose_enemy_equipment",
        }
        if choice_type not in picker_types:
            return True

        previous_choice = getattr(engine, "_active_choice", None)
        if isinstance(choice, dict):
            engine._active_choice = choice
        try:
            target_id = engine._choice_target_id_for_request(player_id, request)
        finally:
            engine._active_choice = previous_choice
        pending = {
            "player_id": int(player_id),
            "choice_type": choice_type,
            "choice_params": params,
            "card": card.to_dict(),
        }
        if target_id is not None:
            pending["target_player_id"] = int(target_id)
        candidates = self._choice_candidates(player_id, pending)

        if choice_type == "choose_cards_from_discard":
            minimum = max(0, engine._eval_int(player_id, params.get("min_count", 0), card, 0))
        elif choice_type in {"choose_cards_from_hand", "choose_same_attacks_from_hand"}:
            minimum = max(0, engine._eval_int(player_id, params.get("min_count", 1), card, 1))
        else:
            minimum = 1
        if minimum == 0:
            return True
        if choice_type == "choose_same_attacks_from_hand" or engine._card_is(card, "Fusion", "vanilla:fusion"):
            counts: dict[str, int] = {}
            for instance_id in candidates:
                candidate = engine._find_card_by_instance_id(instance_id)
                if candidate is not None and candidate.card_type == "thorn":
                    counts[candidate.def_id] = counts.get(candidate.def_id, 0) + 1
            return any(count >= minimum for count in counts.values())
        return len(candidates) >= minimum

    def _trigger_actions(self, player_id: int, equipment, equipment_slot: int) -> list[Action]:
        engine = self.engine
        player = engine.players[player_id]
        card_def = equipment.card_def
        if not engine._equipment_runtime_active(equipment):
            return []
        has_mod_trigger = engine._has_card_event(card_def, "equipment_trigger")
        if card_def.trigger_cost_e < 0 and not has_mod_trigger:
            return []
        if equipment.turns_equipped < 1:
            return []
        cost_e = max(0, int(card_def.trigger_cost_e or 0))
        cost_m = max(0, int(getattr(card_def, "trigger_cost_m", 0) or card_def.v2_resource.get("trigger_cost_m", 0) or 0))
        if player.elixir < cost_e or player.magic < cost_m:
            return []
        max_uses = engine._equipment_trigger_max_uses(equipment)
        if max_uses > 0 and int(equipment.uses_this_turn) >= max_uses:
            return []
        if engine._equipment_trigger_uses_effect_target(card_def):
            targets = [engine._equipment_effect_target_id(equipment, player_id)]
        else:
            targets = [
                target_id for target_id in (0, 1)
                if engine._target_can_be_selected(player_id, target_id, allow_self=True)
            ]
        actions = []
        for target_id in targets:
            if target_id not in (0, 1) or engine.players[target_id].health <= 0:
                continue
            if engine._equipment_trigger_forbids_self_target(card_def) and target_id == player_id:
                continue
            actions.append(Action("use_trigger", {
                "equipment_slot": int(equipment_slot),
                "target_player_id": int(target_id),
            }))
        return actions

    def _response_actions(self, player_id: int) -> list[Action]:
        pending = self.engine.pending_response or {}
        played_card = pending.get("card") or {}
        actions = [Action("respond", {"hand_slot": None})]
        seen: set[int] = set()
        valid_ids: set[int] = set()
        for trigger_type in _response_trigger_types(self.engine, played_card):
            for card in self.engine.get_counter_cards(player_id, trigger_type):
                if card.instance_id in seen or not self.engine._can_pay_counter_card(player_id, card):
                    continue
                seen.add(card.instance_id)
                valid_ids.add(int(card.instance_id))
        for hand_slot, card in enumerate(self.engine.players[player_id].hand):
            if int(card.instance_id) in valid_ids:
                actions.append(Action("respond", {"hand_slot": int(hand_slot)}))
        return actions

    def _choice_actions(self, player_id: int) -> list[Action]:
        pending = self.engine.pending_choice or {}
        choice_type = str(pending.get("choice_type") or "")
        params = pending.get("choice_params") if isinstance(pending.get("choice_params"), dict) else {}
        self._sync_choice_builder()

        if choice_type == "confirm":
            return [
                Action("resolve_choice", {"choice": {"confirmed": True, "accepted": True}}),
                Action("resolve_choice", {"choice": {"cancelled": True}}),
            ]
        if choice_type == "magic_salt_reflect":
            return [
                Action("resolve_choice", {"choice": {"confirmed": True, "accepted": True}}),
                Action("resolve_choice", {"choice": {"cancelled": True}}),
            ]
        if choice_type == "hel_card_suit":
            return [Action("resolve_choice", {"choice": {"hel_suit": suit}}) for suit in ("heart", "diamond", "spade", "club")]
        if choice_type == "bio_blood_sugar_mode":
            return [Action("resolve_choice", {"choice": {"bio_blood_sugar_mode": mode}}) for mode in ("electric_target", "physical_target")]
        if choice_type == "choose_target":
            return [
                Action("resolve_choice", {"choice": _target_choice(target_id)})
                for target_id in self._choice_targets(player_id, params)
            ]
        if choice_type == "reorder_deck":
            return self._reorder_actions(pending)
        if choice_type in {"choose_cards_from_hand", "choose_same_attacks_from_hand", "choose_cards_from_discard", "foresight_replace"}:
            return self._multi_choice_actions(player_id, pending)

        candidates = self._choice_candidates(player_id, pending)
        actions = [
            Action("select_choice", {"candidate_slot": self._choice_slot(instance_id)})
            for instance_id in candidates
            if self._choice_slot(instance_id) >= 0
        ]
        if choice_type == "choose_ocean_sapphire":
            target_ids = self._choice_targets(
                player_id,
                {"target": "all", "include_self": True},
            )
            actions = [
                Action("select_choice", {
                    "candidate_slot": self._choice_slot(instance_id),
                    "target_player_id": int(target),
                })
                for target in target_ids for instance_id in candidates
                if self._choice_slot(instance_id) >= 0
            ]
        if params.get("cancellable") is not False:
            actions.append(Action("resolve_choice", {"choice": {"cancelled": True}}))
        if not actions:
            default = self.engine._default_choice_for_pending(pending)
            if isinstance(default, dict):
                actions.append(Action("default_choice"))
        if not actions:
            raise UnsupportedDecisionError(f"no legal encoding for choice type {choice_type!r}")
        return _unique_actions(actions)

    def _choice_targets(self, player_id: int, params: dict[str, Any]) -> list[int]:
        selector = str(params.get("target") or params.get("allowed") or "all").lower()
        include_self = bool(params.get("include_self", selector in {"all", "any", "self", "friendly"}))
        result = []
        for target_id in (0, 1):
            if selector in {"self", "owner"} and target_id != player_id:
                continue
            if selector in {"enemy", "enemies", "opponent", "opponents"} and target_id == player_id:
                continue
            if not self.engine._target_can_be_selected(player_id, target_id, allow_self=include_self):
                continue
            result.append(target_id)
        return result

    def _choice_candidates(self, player_id: int, pending: dict[str, Any]) -> list[int]:
        choice_type = str(pending.get("choice_type") or "")
        card = self._pending_card(pending)
        target_id = _as_int(pending.get("target_player_id"), player_id)
        serialized_hand = pending.get("hand_cards")
        if isinstance(serialized_hand, list):
            # The production engine snapshots constrained candidates for DNA,
            # Reconstructor, Sapphire/Ruby and enemy-hand choices. That snapshot
            # is authoritative; rebuilding it from the live zone loses card-
            # specific predicates and can incorrectly leave no legal action.
            current_iid = _as_int((pending.get("card") or {}).get("instance_id"), -1)
            candidate_ids = [
                _as_int(item.get("instance_id"), -1)
                for item in serialized_hand
                if isinstance(item, dict)
            ]
            candidate_ids = [
                instance_id
                for instance_id in candidate_ids
                if instance_id >= 0 and instance_id != current_iid
            ]
            return list(dict.fromkeys(
                instance_id
                for instance_id in candidate_ids
                if self._choice_candidate_is_currently_legal(
                    player_id,
                    choice_type,
                    card,
                    self.engine._find_card_by_instance_id(instance_id),
                    pending,
                )
            ))
        if choice_type in {"choose_cards_from_hand", "choose_same_attacks_from_hand", "foresight_replace"}:
            current_iid = _as_int((pending.get("card") or {}).get("instance_id"), -1)
            candidates = [
                candidate for candidate in self.engine.players[player_id].hand
                if int(candidate.instance_id) != current_iid and self.engine._card_selectable_by_action(candidate)
            ]
            source_def_id = str((pending.get("card") or {}).get("def_id") or getattr(card, "def_id", ""))
            fusion_picker = (
                choice_type == "choose_same_attacks_from_hand"
                or source_def_id in {"Fusion", "vanilla:fusion"}
            )
            if fusion_picker:
                candidates = [candidate for candidate in candidates if candidate.card_type == "thorn"]
                if self._choice_selection:
                    selected = self.engine._find_card_by_instance_id(self._choice_selection[0])
                    if selected is not None:
                        candidates = [candidate for candidate in candidates if candidate.def_id == selected.def_id]
            return [int(candidate.instance_id) for candidate in candidates]
        if choice_type == "choose_cards_from_discard":
            return [
                int(candidate.instance_id)
                for candidate in self.engine.players[player_id].discard
                if self.engine._card_selectable_by_action(candidate)
            ]
        if choice_type == "choose_ocean_sapphire":
            return [int(item.instance_id) for item in self.engine._ocean_sapphire_selectable_attacks(player_id, card)]
        if choice_type == "choose_arctic_ruby":
            return [int(item.instance_id) for item in self.engine._arctic_ruby_selectable_attacks(player_id, card)]
        if choice_type in {"choose_equipment", "choose_enemy_equipment"}:
            owner_id = target_id if choice_type == "choose_enemy_equipment" else player_id
            if owner_id not in (0, 1):
                return []
            return [int(item.card_instance.instance_id) for item in self.engine.players[owner_id].equipment]
        zone_name = {
            "choose_attack_from_hand": "hand",
            "choose_card_from_hand": "hand",
            "choose_card_to_discard": "hand",
            "choose_from_enemy_hand": "hand",
            "choose_from_deck": "deck",
            "choose_from_discard": "discard",
            "choose_card_from_discard": "discard",
            "choose_from_exile": "exile",
        }.get(choice_type)
        if zone_name is None:
            return []
        owner_id = target_id if choice_type in {"choose_from_enemy_hand"} else player_id
        if choice_type in {"choose_card_from_hand", "choose_from_deck", "choose_from_discard", "choose_from_exile"} and target_id in (0, 1):
            owner_id = target_id
        zone = getattr(self.engine.players[owner_id], zone_name)
        result = []
        current_iid = int(card.instance_id) if card is not None else -1
        for candidate in zone:
            if candidate.instance_id == current_iid or not self.engine._card_selectable_by_action(candidate):
                continue
            if not self._choice_candidate_is_currently_legal(
                player_id,
                choice_type,
                card,
                candidate,
                pending,
            ):
                continue
            result.append(int(candidate.instance_id))
        return result

    def _choice_candidate_is_currently_legal(
        self,
        player_id: int,
        choice_type: str,
        source_card,
        candidate,
        pending: dict[str, Any] | None = None,
    ) -> bool:
        if candidate is None or not self.engine._card_selectable_by_action(candidate):
            return False
        if choice_type == "choose_attack_from_hand" and candidate.card_type != "thorn":
            return False
        if choice_type == "choose_card_from_hand" and source_card is not None and self.engine._card_is(source_card, "Mimic"):
            special_cost = max(0, self.engine._mimic_special_cost_for_card(candidate))
            source_cost = 0
            if not bool((pending or {}).get("already_paid")):
                live_source = self.engine.players[player_id].find_hand_card(source_card.instance_id)
                source = live_source or source_card
                source_cost = max(
                    0,
                    int(source.cost_e) + self.engine._get_extra_e_for_card(player_id, source),
                )
            if int(self.engine.players[player_id].elixir) < source_cost + special_cost:
                return False
        return True

    def _multi_choice_actions(self, player_id: int, pending: dict[str, Any]) -> list[Action]:
        choice_type = str(pending.get("choice_type") or "")
        params = pending.get("choice_params") if isinstance(pending.get("choice_params"), dict) else {}
        candidates = self._choice_candidates(player_id, pending)
        if choice_type == "foresight_replace":
            candidates = [int(card.instance_id) for card in self.engine.players[player_id].hand]
            minimum = 0
            maximum = max(0, _as_int(params.get("max_count"), len(candidates)))
        else:
            minimum = max(0, self.engine._eval_int(player_id, params.get("min_count", 1), self._pending_card(pending), 1))
            maximum = max(minimum, self.engine._eval_int(player_id, params.get("max_count", params.get("count", len(candidates))), self._pending_card(pending), len(candidates)))
        actions = []
        # Already-selected ids stay removable even after a constrained picker
        # (notably Fusion) narrows the remaining candidate set.
        candidates = list(dict.fromkeys([*self._choice_selection, *candidates]))
        for instance_id in candidates:
            selected = instance_id in self._choice_selection
            slot = self._choice_slot(instance_id)
            if slot >= 0 and (selected or len(self._choice_selection) < maximum):
                actions.append(Action("toggle_choice", {"candidate_slot": slot}))
        if minimum <= len(self._choice_selection) <= maximum:
            actions.append(Action("submit_choice"))
        if params.get("cancellable") is not False:
            actions.append(Action("resolve_choice", {"choice": {"cancelled": True}}))
        return _unique_actions(actions)

    def _reorder_actions(self, pending: dict[str, Any]) -> list[Action]:
        target_id = _as_int(pending.get("target_player_id"), 1 - _as_int(pending.get("player_id"), 0))
        candidates = [int(card.instance_id) for card in self.engine.players[target_id].deck] if target_id in (0, 1) else []
        remaining = [item for item in candidates if item not in self._choice_selection]
        actions = [
            Action("append_choice_order", {"candidate_slot": self._choice_slot(item)})
            for item in remaining
            if self._choice_slot(item) >= 0
        ]
        if self._choice_selection:
            actions.append(Action("reset_choice_order"))
        if len(self._choice_selection) == len(candidates):
            actions.append(Action("submit_choice"))
        params = pending.get("choice_params") or {}
        if params.get("cancellable") is not False:
            actions.append(Action("resolve_choice", {"choice": {"cancelled": True}}))
        return _unique_actions(actions)

    def _execute(self, action: Action, player_id: int) -> dict[str, Any]:
        payload = action.payload
        if self._in_pregame():
            return self._execute_pregame(action, player_id)
        if action.kind == "play_card":
            card = self._hand_card(player_id, payload.get("hand_slot"))
            choice = payload.get("choice")
            if not isinstance(choice, dict):
                request = self.engine._get_choice_request(card)
                request_type = self.engine._choice_type_for_effect(request, card)
                flags = self.engine._effective_card_flags(card)
                if request_type == "choose_target" and "wide_strike" in flags:
                    targets = self.engine._wide_strike_target_ids(player_id, card)
                    if targets:
                        choice = _target_choice(targets[0])
                elif request_type == "choose_target" and self.engine._card_is_self_only(card):
                    choice = _target_choice(player_id)
            return self.engine.play_card(player_id, int(card.instance_id), choice)
        if action.kind == "respond":
            hand_slot = payload.get("hand_slot")
            card = None if hand_slot is None else self._hand_card(player_id, hand_slot)
            return self.engine.handle_response(player_id, None if card is None else int(card.instance_id))
        if action.kind == "resolve_choice":
            return self.engine.resolve_choice(player_id, dict(payload.get("choice") or {}))
        if action.kind == "select_choice":
            instance_id = self._choice_id(payload.get("candidate_slot"))
            choice = {"target_instance_id": instance_id}
            target_id = _as_int(payload.get("target_player_id"), -1)
            if target_id in (0, 1):
                choice.update(_target_choice(target_id))
            return self.engine.resolve_choice(player_id, choice)
        if action.kind == "default_choice":
            choice = self.engine._default_choice_for_pending(self.engine.pending_choice or {})
            if not isinstance(choice, dict):
                raise IllegalActionError("engine has no default choice")
            return self.engine.resolve_choice(player_id, choice)
        if action.kind == "toggle_choice":
            instance_id = self._choice_id(payload.get("candidate_slot"))
            if instance_id in self._choice_selection:
                self._choice_selection.remove(instance_id)
            else:
                self._choice_selection.append(instance_id)
            return {"success": True, "selection_only": True}
        if action.kind == "append_choice_order":
            instance_id = self._choice_id(payload.get("candidate_slot"))
            if instance_id not in self._choice_selection:
                self._choice_selection.append(instance_id)
            return {"success": True, "selection_only": True}
        if action.kind == "reset_choice_order":
            self._choice_selection.clear()
            return {"success": True, "selection_only": True}
        if action.kind == "submit_choice":
            return self.engine.resolve_choice(player_id, self._built_choice_payload())
        if action.kind == "use_trigger":
            equipment = self._equipment(player_id, payload.get("equipment_slot"))
            return self.engine.use_trigger(
                player_id,
                int(equipment.card_instance.instance_id),
                int(payload["target_player_id"]),
            )
        if action.kind == "end_turn":
            return self.engine.end_turn(player_id)
        raise IllegalActionError(f"unknown action kind: {action.kind}")

    def _execute_pregame(self, action: Action, player_id: int) -> dict[str, Any]:
        engine = self.engine
        payload = action.payload
        if action.kind == "select_opening_event":
            slot = self._pregame_slot(payload.get("option_slot"), engine.opening_event_options[player_id])
            event = engine.opening_event_options[player_id][slot]
            if not engine.select_opening_event(player_id, event.get("id")):
                return {"success": False, "error": "opening event was rejected"}
            self._advance_pregame(player_id)
            return {"success": True}
        if action.kind == "reroll_opening_event":
            return {
                "success": bool(engine.reroll_opening_event(player_id)),
                "rerolls_left": int(engine.draft_rerolls[player_id]),
            }
        if action.kind == "confirm_opening_reveal":
            success = bool(engine.start_draft_for_player(player_id))
            if success:
                self._advance_pregame(player_id)
            return {"success": success}
        if action.kind == "draft_pick":
            options = list(engine.draft_options[player_id] or [])
            slot = self._pregame_slot(payload.get("candidate_slot"), options)
            success = bool(engine.draft_pick(player_id, options[slot].def_id))
            if success:
                engine.get_player_status(player_id)
                self._advance_pregame(player_id)
            return {"success": success}
        if action.kind == "draft_reroll":
            return {
                "success": bool(engine.draft_reroll(player_id)),
                "rerolls_left": int(engine.draft_rerolls[player_id]),
            }
        if action.kind == "toggle_pregame_choice":
            candidates = self.pregame_candidate_def_ids(player_id)
            slot = self._pregame_slot(payload.get("candidate_slot"), candidates)
            selected = self._pregame_selections[player_id]
            if slot in selected:
                selected.remove(slot)
            else:
                selected.append(slot)
            return {"success": True, "selection_only": True}
        if action.kind == "append_pregame_order":
            candidates = self.pregame_candidate_def_ids(player_id)
            slot = self._pregame_slot(payload.get("candidate_slot"), candidates)
            selected = self._pregame_selections[player_id]
            if slot not in selected:
                selected.append(slot)
            return {"success": True, "selection_only": True}
        if action.kind == "reset_pregame_order":
            self._pregame_selections[player_id].clear()
            return {"success": True, "selection_only": True}
        if action.kind == "select_pregame_choice":
            candidates = self.pregame_candidate_def_ids(player_id)
            slot = self._pregame_slot(payload.get("candidate_slot"), candidates)
            self._pregame_selections[player_id] = [slot]
            return self._submit_pregame_choice(player_id)
        if action.kind == "submit_pregame_choice":
            return self._submit_pregame_choice(player_id)
        raise IllegalActionError(f"unknown pregame action kind: {action.kind}")

    def _submit_pregame_choice(self, player_id: int) -> dict[str, Any]:
        engine = self.engine
        event_id = str(engine.opening_event_picks[player_id])
        candidates = self.pregame_candidate_def_ids(player_id)
        selected_slots = list(self._pregame_selections[player_id])
        selected_defs = [candidates[slot] for slot in selected_slots]
        if event_id == "2":
            sub_choice = {"conversions": [
                {"source_def_id": def_id} for def_id in selected_defs[:3]
            ]}
        elif event_id == "3":
            sub_choice = {"convert_def_ids": selected_defs[:5]}
        elif event_id == "5":
            if len(selected_defs) != 1:
                return {"success": False, "error": "fated draw requires one card"}
            sub_choice = {"add_def_ids": selected_defs}
        elif event_id == "8":
            if len(selected_defs) != 1:
                return {"success": False, "error": "survival setup requires one card"}
            sub_choice = {"yggdrasil_convert_def_id": selected_defs[0]}
        elif event_id == "11":
            if len(selected_defs) != len(candidates):
                return {"success": False, "error": "deck order is incomplete"}
            sub_choice = {"deck_order_def_ids": selected_defs}
        else:
            return {"success": False, "error": "unsupported opening event sub-choice"}
        engine.opening_event_sub_choices[player_id] = sub_choice
        engine.player_ready[player_id] = True
        self._pregame_selections[player_id].clear()
        self._advance_pregame(player_id)
        return {"success": True}

    def _advance_pregame(self, player_id: int) -> None:
        self._pregame_cursor = 1 - int(player_id)
        if not all(self.engine.player_ready):
            return
        valid, reason, details = self.engine.validate_pregame_ready()
        if not valid:
            raise RuntimeError(f"production pregame validation failed: {reason}: {details}")
        if not self.engine.start_game():
            raise RuntimeError("production engine refused a validated pregame state")

    @staticmethod
    def _pregame_slot(value: Any, candidates: Sequence[Any]) -> int:
        slot = _as_int(value, -1)
        if slot < 0 or slot >= len(candidates):
            raise IllegalActionError("pregame candidate slot is out of range")
        return slot

    def _pending_card(self, pending: dict[str, Any]):
        data = pending.get("card")
        if not isinstance(data, dict):
            return None
        instance_id = _as_int(data.get("instance_id"), -1)
        card = self.engine._find_card_by_instance_id(instance_id)
        if card is not None and str(getattr(card, "def_id", "")) == str(data.get("def_id") or ""):
            return card
        from cards import CardInstance
        return CardInstance.from_dict(data)

    def _pending_choice_signature(self) -> tuple[Any, ...] | None:
        pending = getattr(self.engine, "pending_choice", None)
        if not isinstance(pending, dict):
            return None
        card = pending.get("card") or {}
        return (
            id(pending),
            _as_int(pending.get("player_id"), -1),
            str(pending.get("choice_type") or ""),
            _as_int(card.get("instance_id"), -1) if isinstance(card, dict) else -1,
        )

    def _sync_choice_builder(self) -> None:
        signature = self._pending_choice_signature()
        if signature != self._choice_signature:
            self._choice_signature = signature
            self._choice_selection = []
            self._choice_candidate_ids = self._base_choice_candidate_ids()

    def _reset_choice_builder(self) -> None:
        self._choice_signature = self._pending_choice_signature()
        self._choice_selection = []
        self._choice_candidate_ids = self._base_choice_candidate_ids()

    def _base_choice_candidate_ids(self) -> list[int]:
        pending = getattr(self.engine, "pending_choice", None)
        actor = self.decision_player()
        if not isinstance(pending, dict) or actor not in (0, 1):
            return []
        choice_type = str(pending.get("choice_type") or "")
        if choice_type == "reorder_deck":
            target_id = _as_int(pending.get("target_player_id"), 1 - actor)
            if target_id in (0, 1):
                return [int(card.instance_id) for card in self.engine.players[target_id].deck]
            return []
        if choice_type == "foresight_replace":
            return [int(card.instance_id) for card in self.engine.players[actor].hand]
        return list(dict.fromkeys(self._choice_candidates(actor, pending)))

    def _choice_slot(self, instance_id: int) -> int:
        try:
            return self._choice_candidate_ids.index(int(instance_id))
        except (TypeError, ValueError):
            return -1

    def _choice_id(self, slot: Any) -> int:
        index = _as_int(slot, -1)
        if index < 0 or index >= len(self._choice_candidate_ids):
            raise IllegalActionError("choice candidate slot is out of range")
        return int(self._choice_candidate_ids[index])

    def _choice_candidate_card(self, instance_id: int, pending: dict[str, Any]):
        card = self.engine._find_card_by_instance_id(int(instance_id))
        if card is not None:
            return card
        for item in pending.get("hand_cards") or []:
            if isinstance(item, dict) and _as_int(item.get("instance_id"), -1) == int(instance_id):
                from cards import CardInstance
                return CardInstance.from_dict(item)
        return None

    def _built_choice_payload(self) -> dict[str, Any]:
        pending = self.engine.pending_choice or {}
        choice_type = str(pending.get("choice_type") or "")
        selected = list(self._choice_selection)
        if choice_type == "reorder_deck":
            return {"new_order": selected}
        if choice_type == "foresight_replace":
            return {"selected_instance_ids": selected}
        return {"target_instance_ids": selected}

    def _hand_card(self, player_id: int, slot: Any):
        index = _as_int(slot, -1)
        hand = self.engine.players[player_id].hand
        if index < 0 or index >= len(hand):
            raise IllegalActionError("hand slot is out of range")
        return hand[index]

    def _equipment(self, player_id: int, slot: Any):
        index = _as_int(slot, -1)
        equipment = self.engine.players[player_id].equipment
        if index < 0 or index >= len(equipment):
            raise IllegalActionError("equipment slot is out of range")
        return equipment[index]

    def _terminal_rewards(self) -> dict[int, float]:
        if not self.engine.game_over or self.engine.winner not in (0, 1):
            return {0: 0.0, 1: 0.0}
        return {0: 1.0 if self.engine.winner == 0 else -1.0, 1: 1.0 if self.engine.winner == 1 else -1.0}

    def _public_action_context(self, action: Action, player_id: int) -> dict[str, Any]:
        payload = action.payload
        context: dict[str, Any] = {"pregame": self._in_pregame()}
        try:
            if action.kind == "play_card":
                context["card_def_id"] = str(self._hand_card(player_id, payload.get("hand_slot")).def_id)
            elif action.kind == "respond":
                hand_slot = payload.get("hand_slot")
                if hand_slot is not None:
                    context["card_def_id"] = str(self._hand_card(player_id, hand_slot).def_id)
            elif action.kind == "use_trigger":
                context["equipment_def_id"] = str(
                    self._equipment(player_id, payload.get("equipment_slot")).card_instance.def_id
                )
        except (IllegalActionError, AttributeError):
            pass
        pending = self.engine.pending_choice
        if isinstance(pending, dict):
            context["choice_type"] = str(pending.get("choice_type") or "")
        return context

    def _record_public_action(
        self,
        action: Action,
        player_id: int,
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        if result.get("selection_only") or context.get("pregame"):
            return
        event: dict[str, Any] = {
            "round": int(self.engine.round_num),
            "player": int(player_id),
            "kind": action.kind,
        }
        payload = action.payload
        if action.kind == "play_card":
            card_data = result.get("card") if isinstance(result.get("card"), dict) else {}
            event["card_def_id"] = str(card_data.get("def_id") or context.get("card_def_id") or "")
            choice = payload.get("choice") or {}
            if isinstance(choice, dict) and _as_int(choice.get("target_player_id"), -1) >= 0:
                event["target_player"] = _as_int(choice.get("target_player_id"), -1)
        elif action.kind == "respond":
            event["passed"] = payload.get("hand_slot") is None
            if context.get("card_def_id"):
                event["card_def_id"] = str(context["card_def_id"])
        elif action.kind == "use_trigger":
            event["equipment_def_id"] = str(context.get("equipment_def_id") or "")
            event["target_player"] = _as_int(payload.get("target_player_id"), -1)
        elif action.kind in {"resolve_choice", "select_choice", "default_choice", "submit_choice"}:
            event["choice_type"] = str(context.get("choice_type") or "resolved")
        self.public_history.append(event)
        if len(self.public_history) > self.history_limit:
            del self.public_history[:-self.history_limit]


def _response_trigger_types(engine, played_card: dict[str, Any]) -> list[str]:
    from cards import CARD_DEFS, CardInstance

    card_def = CARD_DEFS.get(str(played_card.get("def_id") or ""))
    if card_def is None:
        return []
    result: list[str] = []

    def add(value: str) -> None:
        if value and value not in result:
            result.append(value)

    if card_def.card_type in {"thorn", "bloom", "root"}:
        add(card_def.card_type)
    try:
        instance = CardInstance.from_dict(played_card)
    except Exception:
        instance = None
    if instance is not None:
        try:
            if engine._would_destroy_equipment(instance):
                add("equipment_destroy")
        except Exception:
            pass
        try:
            from void_dlc_runtime import card_applies_hand_charge
            if card_applies_hand_charge(instance):
                add("hand_charge")
        except Exception:
            pass
        try:
            if engine._would_heal(instance):
                add("heal")
        except Exception:
            pass
    if card_def.id in {"Sewage", "MagicSewage"}:
        add("equipment_destroy")
    add("targeted")
    add("any")
    return result


def _target_choice(target_id: int) -> dict[str, int]:
    return {"target_player": int(target_id), "target_player_id": int(target_id), "target_id": int(target_id)}


def _unique_actions(actions: list[Action]) -> list[Action]:
    result = []
    seen = set()
    for action in actions:
        if action.key in seen:
            continue
        seen.add(action.key)
        result.append(action)
    return result


def _public_engine_result(result: dict[str, Any]) -> dict[str, Any]:
    allowed = {"success", "needs_choice", "needs_response", "cancelled", "target_player_id", "ignored", "reordered"}
    return {key: value for key, value in result.items() if key in allowed and isinstance(value, (bool, int, float, str, type(None)))}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
