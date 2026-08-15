from __future__ import annotations

import argparse
import base64
import copy
import gzip
import json
import os
import pickle
import secrets
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .diagnostics import DiagnosticSessionRecorder, cleanup_diagnostic_storage
from .environment import Garden1v1Env
from .game_imports import configure_game_imports
from .policies import policy_from_name
from .protocol import Action


MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_POLICY = (
    "unsafe-rollout-cpu:{checkpoint};candidates=3;rollouts=2;"
    "horizon=2;belief=true;exploration=0"
)


def default_policy_name() -> str:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "structured-v2-search-dagger-v2.epoch-06.pt"
    )
    return DEFAULT_POLICY.format(checkpoint=checkpoint)


def encode_engine(engine) -> str:
    payload = pickle.dumps(engine, protocol=5)
    return base64.b64encode(gzip.compress(payload, compresslevel=3)).decode("ascii")


def decode_engine(payload: str):
    if not isinstance(payload, str) or not payload:
        raise ValueError("engine_snapshot is required")
    compressed = base64.b64decode(payload.encode("ascii"), validate=True)
    if len(compressed) > MAX_REQUEST_BYTES:
        raise ValueError("compressed engine snapshot is too large")
    raw = gzip.decompress(compressed)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("engine snapshot is too large")
    # This worker accepts snapshots only from the authenticated loopback game
    # process. Never expose the endpoint on a non-loopback interface.
    return pickle.loads(raw)


class LiveDecisionService:
    def __init__(
        self,
        *,
        policy_name: str,
        game_root: str | Path,
        diagnostics_root: str | Path,
        max_sessions: int = 16,
        retention_days: float = 14.0,
        max_diagnostic_bytes: int = 2 * 1024 * 1024 * 1024,
        export_finished: bool = True,
    ) -> None:
        self.policy_name = str(policy_name)
        self.game_root = Path(game_root).resolve()
        # Pickle resolves production classes while decoding the snapshot, before
        # Garden1v1Env can configure its imports. Install the trusted game root
        # as soon as the worker service starts.
        configure_game_imports(self.game_root)
        self.diagnostics_root = Path(diagnostics_root).resolve()
        self.max_sessions = max(1, int(max_sessions))
        self.retention_days = max(0.0, float(retention_days))
        self.max_diagnostic_bytes = max(0, int(max_diagnostic_bytes))
        self.export_finished = bool(export_finished)
        self._policies: OrderedDict[str, Any] = OrderedDict()
        self._recorders: dict[str, DiagnosticSessionRecorder] = {}
        self._lock = threading.RLock()

        self._cleanup_storage()
        # Load large checkpoint weights once. Session policies fork only RNG and
        # search bookkeeping while sharing the immutable inference network.
        self._policy_template = policy_from_name(self.policy_name, seed=0)

    def _policy_for(self, session_id: str, *, seed: int):
        with self._lock:
            policy = self._policies.get(session_id)
            if policy is not None:
                self._policies.move_to_end(session_id)
                return policy
            fork = getattr(self._policy_template, "fork", None)
            policy = (
                fork(seed=int(seed), name=getattr(self._policy_template, "name", None))
                if callable(fork)
                else policy_from_name(self.policy_name, seed=int(seed))
            )
            self._policies[session_id] = policy
            while len(self._policies) > self.max_sessions:
                stale_id, _ = self._policies.popitem(last=False)
                recorder = self._recorders.pop(stale_id, None)
                if recorder is not None:
                    recorder.finish(outcome={"reason": "session_evicted"})
            self._cleanup_storage()
            return policy

    def _cleanup_storage(self) -> dict[str, int]:
        protected = set(self._policies) | set(self._recorders)
        protected.discard("__warmup__")
        return cleanup_diagnostic_storage(
            self.diagnostics_root,
            retention_days=self.retention_days,
            max_bytes=self.max_diagnostic_bytes,
            protected_session_ids=protected,
        )

    def _recorder_for(
        self,
        session_id: str,
        *,
        env: Garden1v1Env,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosticSessionRecorder:
        with self._lock:
            recorder = self._recorders.get(session_id)
            if recorder is None:
                recorder = DiagnosticSessionRecorder(
                    self.diagnostics_root,
                    session_id=session_id,
                    metadata={
                        "source": "local_human_vs_ai",
                        "policy": self.policy_name,
                        "ruleset_fingerprint": env.ruleset_fingerprint,
                        "loadout_fingerprint": env.loadout_fingerprint,
                        **dict(metadata or {}),
                    },
                )
                self._recorders[session_id] = recorder
            return recorder

    def _environment(self, request: dict[str, Any]) -> Garden1v1Env:
        engine = decode_engine(request.get("engine_snapshot"))
        return Garden1v1Env.from_engine(
            engine,
            game_root=self.game_root,
            seed=int(request.get("seed", 0) or 0),
            enabled_mods=request.get("enabled_mods"),
            public_history=request.get("public_history") or (),
        )

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        env = self._environment(request)
        player_id = int(request.get("player_id", env.decision_player(default=-1)))
        if env.decision_player() != player_id:
            raise ValueError("requested player does not own the current decision")
        observation = env.observe(player_id)
        legal_actions = env.legal_actions(player_id)
        if not legal_actions:
            raise ValueError("the current decision has no legal action")
        legal_actions = _filter_unambiguous_bad_targets(observation, legal_actions)
        policy = self._policy_for(session_id, seed=int(request.get("seed", 0) or 0))
        started = time.perf_counter()
        if hasattr(policy, "select_action_with_env"):
            action = policy.select_action_with_env(
                env,
                observation,
                legal_actions,
                player_id,
            )
        else:
            action = policy.select_action(observation, legal_actions)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metadata = getattr(policy, "last_search_metadata", None)
        if request.get("record", True):
            recorder = self._recorder_for(
                session_id,
                env=env,
                metadata=request.get("session_metadata"),
            )
            record = recorder.record_decision(
                actor=player_id,
                actor_kind="ai",
                observation=observation,
                legal_actions=legal_actions,
                action=action,
                elapsed_ms=elapsed_ms,
                policy_metadata=metadata,
                engine=_sanitized_engine_snapshot(env.engine),
            )
            decision_id = record.decision_id
        else:
            decision_id = None
        step_info = None
        next_player = env.decision_player()
        updated_snapshot = None
        if request.get("execute"):
            step_result = env.step(action, player_id)
            step_info = step_result.info
            next_player = env.decision_player()
            updated_snapshot = encode_engine(env.engine)
        return {
            "success": True,
            "action": action.to_dict(),
            "decision_id": decision_id,
            "elapsed_ms": round(elapsed_ms, 3),
            "policy": getattr(policy, "name", type(policy).__name__),
            "search": metadata,
            "legal_action_count": len(legal_actions),
            "executed": bool(request.get("execute")),
            "engine_snapshot": updated_snapshot,
            "next_player": next_player,
            "terminated": bool(getattr(env.engine, "game_over", False)),
            "step_info": step_info,
        }

    def record(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        env = self._environment(request)
        actor = int(request.get("player_id", env.decision_player(default=-1)))
        observation = env.observe(actor)
        try:
            legal_actions = env.legal_actions(actor)
        except Exception:
            legal_actions = []
        raw_action = Action.from_dict(request.get("action") or {})
        action, canonical_reason = _canonical_external_action(env, actor, raw_action)
        recorder = self._recorder_for(
            session_id,
            env=env,
            metadata=request.get("session_metadata"),
        )
        record = recorder.record_decision(
            actor=actor,
            actor_kind=str(request.get("actor_kind") or "human"),
            observation=observation,
            legal_actions=legal_actions,
            action=action,
            elapsed_ms=request.get("elapsed_ms"),
            policy_metadata={
                **dict(request.get("policy_metadata") or {}),
                "external_action_canonicalized": action.key in {item.key for item in legal_actions},
                "external_action_canonical_reason": canonical_reason,
                "external_action_raw": raw_action.to_dict(),
            },
            engine=_sanitized_engine_snapshot(env.engine),
        )
        return {
            "success": True,
            "decision_id": record.decision_id,
            "canonicalized": action.key in {item.key for item in legal_actions},
            "canonical_action": action.to_dict(),
        }

    def mark(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request.get("session_id") or "").strip()
        recorder = self._recorders.get(session_id)
        if recorder is None:
            raise ValueError("diagnostic session is not active")
        marker = recorder.mark(
            request.get("decision_id"),
            label=str(request.get("label") or "review"),
            note=str(request.get("note") or ""),
        )
        return {"success": True, "marker": marker}

    def finish(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request.get("session_id") or "").strip()
        with self._lock:
            recorder = self._recorders.pop(session_id, None)
            self._policies.pop(session_id, None)
        if recorder is None:
            return {"success": True, "export": None}
        recorder.finish(outcome=request.get("outcome"))
        exported = recorder.export() if self.export_finished else None
        self._cleanup_storage()
        return {"success": True, "export": str(exported) if exported else None}


def _sanitized_engine_snapshot(engine):
    """Remove player-controlled identity text from private training snapshots."""

    sanitized = copy.copy(engine)
    player_names = getattr(engine, "player_names", None)
    if isinstance(player_names, (list, tuple)):
        generic = [f"Player {index + 1}" for index in range(len(player_names))]
        sanitized.player_names = tuple(generic) if isinstance(player_names, tuple) else generic
    return sanitized


def _filter_unambiguous_bad_targets(
    observation: dict[str, Any],
    legal_actions: list[Action],
) -> list[Action]:
    """Remove target choices whose direction is unambiguously backwards.

    This intentionally remains a very small allow-list. Most equipment can have
    situationally useful hostile or self-directed plays, so only cards whose
    benefit direction is unconditional belong here.
    """

    seat = int(observation.get("seat", -1))
    hand = {
        int(card.get("slot", -1)): str(card.get("def_id") or "")
        for card in ((observation.get("self") or {}).get("hand") or [])
        if isinstance(card, dict)
    }
    filtered: list[Action] = []
    for action in legal_actions:
        if action.kind != "play_card":
            filtered.append(action)
            continue
        card_id = hand.get(int(action.payload.get("hand_slot", -1)), "")
        choice = action.payload.get("choice")
        target = _target_from_payload(choice if isinstance(choice, dict) else {})
        wrong_target = (
            (card_id == "Pincer" and target == seat)
            or (card_id == "Powder" and target is not None and target != seat)
        )
        if not wrong_target:
            filtered.append(action)
    return filtered or legal_actions


class _WorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, token: str, service: LiveDecisionService):
        super().__init__(address, handler)
        self.token = token
        self.service = service


def _canonical_external_action(
    env: Garden1v1Env,
    actor: int,
    raw_action: Action,
) -> tuple[Action, str]:
    """Map browser/server payloads to the stable slot-based training protocol."""

    try:
        legal = env.legal_actions(actor)
    except Exception as exc:
        return raw_action, f"legal_actions_unavailable:{type(exc).__name__}"
    by_key = {action.key: action for action in legal}
    if raw_action.key in by_key:
        return by_key[raw_action.key], "already_canonical"

    payload = raw_action.payload
    kind = raw_action.kind
    if kind == "end_turn":
        mapped = _first_action(legal, "end_turn")
        if mapped is not None:
            return mapped, "mapped_end_turn"
        return raw_action, "end_turn_not_legal"

    if kind == "play_card":
        slot = _card_slot(
            env.engine.players[actor].hand,
            payload.get("card_instance_id"),
        )
        candidates = [
            action for action in legal
            if action.kind == "play_card" and action.payload.get("hand_slot") == slot
        ]
        target = _target_from_payload(payload)
        if target is not None:
            targeted = [
                action for action in candidates
                if _target_from_payload(action.payload) == target
            ]
            if targeted:
                candidates = targeted
        if len(candidates) == 1:
            return candidates[0], "mapped_card_instance_to_hand_slot"

    if kind == "respond":
        instance_id = payload.get("card_instance_id")
        if instance_id in (None, ""):
            candidates = [
                action for action in legal
                if action.kind == "respond" and action.payload.get("hand_slot") is None
            ]
        else:
            slot = _card_slot(env.engine.players[actor].hand, instance_id)
            candidates = [
                action for action in legal
                if action.kind == "respond" and action.payload.get("hand_slot") == slot
            ]
        if len(candidates) == 1:
            return candidates[0], "mapped_response_instance_to_hand_slot"

    if kind == "use_trigger":
        slot = _equipment_slot(
            env.engine.players[actor].equipment,
            payload.get("equipment_instance_id"),
        )
        target = _target_from_payload(payload)
        candidates = [
            action for action in legal
            if action.kind == "use_trigger"
            and action.payload.get("equipment_slot") == slot
            and (target is None or _target_from_payload(action.payload) == target)
        ]
        if len(candidates) == 1:
            return candidates[0], "mapped_equipment_instance_to_slot"

    if kind == "resolve_choice":
        choice = payload.get("choice") if isinstance(payload.get("choice"), dict) else {}
        instance_id = choice.get("target_instance_id")
        if instance_id is not None:
            env._sync_choice_builder()
            slot = env._choice_slot(int(instance_id))
            candidates = [
                action for action in legal
                if action.kind == "select_choice"
                and action.payload.get("candidate_slot") == slot
            ]
            target = _target_from_payload(choice)
            if target is not None:
                candidates = [
                    action for action in candidates
                    if _target_from_payload(action.payload) == target
                ]
            if len(candidates) == 1:
                return candidates[0], "mapped_single_choice_instance_to_slot"
        normalized_choice = _normalized_choice(choice)
        candidates = [
            action for action in legal
            if action.kind == "resolve_choice"
            and _normalized_choice(action.payload.get("choice") or {}) == normalized_choice
        ]
        if len(candidates) == 1:
            return candidates[0], "mapped_choice_payload"

    if kind == "v2_ui_response" and env.engine.pending_v2_ui is not None:
        pending = env.engine.pending_v2_ui
        component = pending.get("component") if isinstance(pending, dict) else {}
        context = pending.get("context") if isinstance(pending, dict) else {}
        try:
            from mod_runtime_v2 import validate_v2_ui_response

            raw_response = {
                "button": payload.get("button") or payload.get("button_id"),
                "values": payload.get("values") if isinstance(payload.get("values"), dict) else {},
            }
            clean_raw = validate_v2_ui_response(env.engine, context or {}, component or {}, raw_response)
            candidates = []
            for action in legal:
                if action.kind != "v2_ui_response":
                    continue
                clean_action = validate_v2_ui_response(
                    env.engine,
                    context or {},
                    component or {},
                    env._v2_response_payload(action),
                )
                if clean_action == clean_raw:
                    candidates.append(action)
            if len(candidates) == 1:
                return candidates[0], "mapped_v2_values_to_slots"
        except Exception as exc:
            return raw_action, f"v2_mapping_failed:{type(exc).__name__}"

    return raw_action, "no_unique_canonical_mapping"


def _first_action(actions: list[Action], kind: str) -> Action | None:
    return next((action for action in actions if action.kind == kind), None)


def _card_slot(cards, instance_id: Any) -> int:
    try:
        expected = int(instance_id)
    except (TypeError, ValueError):
        return -1
    return next(
        (slot for slot, card in enumerate(cards) if int(card.instance_id) == expected),
        -1,
    )


def _equipment_slot(equipment, instance_id: Any) -> int:
    try:
        expected = int(instance_id)
    except (TypeError, ValueError):
        return -1
    return next((
        slot for slot, item in enumerate(equipment)
        if int(item.card_instance.instance_id) == expected
    ), -1)


def _target_from_payload(payload: dict[str, Any]) -> int | None:
    choice = payload.get("choice") if isinstance(payload.get("choice"), dict) else payload
    for key in ("target_player_id", "target_player", "target_id"):
        try:
            if choice.get(key) is not None:
                return int(choice[key])
        except (TypeError, ValueError):
            continue
    return None


def _normalized_choice(choice: dict[str, Any]) -> dict[str, Any]:
    result = dict(choice)
    target = _target_from_payload(result)
    for key in ("target_player_id", "target_player", "target_id"):
        result.pop(key, None)
    if target is not None:
        result["target_player_id"] = target
    return result


class LiveWorkerHandler(BaseHTTPRequestHandler):
    server_version = "GTNAILiveWorker/1"

    def log_message(self, format: str, *args) -> None:
        return

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return (
            self.client_address[0] in {"127.0.0.1", "::1"}
            and secrets.compare_digest(
                self.headers.get("Authorization", ""),
                f"Bearer {self.server.token}",
            )
        )

    def do_GET(self) -> None:
        if self.path != "/health" or not self._authorized():
            self._reply(404, {"success": False, "error": "not found"})
            return
        self._reply(200, {"success": True, "status": "ready"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._reply(404, {"success": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/shutdown":
                self._reply(200, {"success": True, "status": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            routes = {
                "/v1/live/decide": self.server.service.decide,
                "/v1/live/record": self.server.service.record,
                "/v1/live/mark": self.server.service.mark,
                "/v1/live/finish": self.server.service.finish,
            }
            handler = routes.get(self.path)
            if handler is None:
                self._reply(404, {"success": False, "error": "not found"})
                return
            self._reply(200, handler(request))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._reply(400, {"success": False, "error": str(exc)})
        except Exception as exc:
            self._reply(500, {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loopback-only GTN live AI worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--token", default=os.environ.get("GTN_AI_WORKER_TOKEN", ""))
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--game-root", required=True)
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--policy", default=default_policy_name())
    parser.add_argument("--max-sessions", type=int, default=16)
    parser.add_argument("--retention-days", type=float, default=14.0)
    parser.add_argument(
        "--max-diagnostic-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
    )
    parser.add_argument("--export-finished", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("live worker must bind to loopback")
    token = str(args.token or "")
    if len(token) < 24:
        raise SystemExit("a strong worker token is required")
    service = LiveDecisionService(
        policy_name=args.policy,
        game_root=args.game_root,
        diagnostics_root=args.diagnostics_root,
        max_sessions=args.max_sessions,
        retention_days=args.retention_days,
        max_diagnostic_bytes=args.max_diagnostic_bytes,
        export_finished=args.export_finished,
    )
    server = _WorkerServer((args.host, args.port), LiveWorkerHandler, token=token, service=service)
    ready_file = Path(args.ready_file)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.write_text(
        json.dumps({"port": server.server_port, "pid": os.getpid()}),
        encoding="utf-8",
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        ready_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
