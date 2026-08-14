from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence

from .policies import Policy, policy_from_name
from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, SERVICE_API_VERSION, Action


MAX_REQUEST_BYTES = 2 * 1024 * 1024
class DecisionService:
    def __init__(self, policy: Policy):
        if bool(getattr(policy, "offline_only", False)):
            raise ValueError("offline-only search policies cannot be served")
        self.policy = policy

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        if _as_int(request.get("api_version"), SERVICE_API_VERSION) != SERVICE_API_VERSION:
            raise ValueError("unsupported service api version")
        observation = request.get("observation")
        raw_actions = request.get("legal_actions")
        if not isinstance(observation, dict) or not isinstance(raw_actions, list):
            raise ValueError("request requires observation and legal_actions")
        if _as_int(observation.get("schema_version"), -1) != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported observation schema version")
        actions = [Action.from_dict(item) for item in raw_actions]
        if not actions:
            raise ValueError("legal_actions must not be empty")
        selected = self.policy.select_action(observation, actions)
        legal_by_key = {action.key: action for action in actions}
        if selected.key not in legal_by_key:
            raise RuntimeError("policy returned an action outside the supplied legal set")
        return {
            "api_version": SERVICE_API_VERSION,
            "model": str(self.policy.name),
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "action": selected.to_dict(),
        }


def make_handler(service: DecisionService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GTNAI/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(HTTPStatus.OK, {
                    "ok": True,
                    "api_version": SERVICE_API_VERSION,
                    "model": str(service.policy.name),
                    "ruleset_fingerprint": str(
                        getattr(service.policy, "ruleset_fingerprint", "") or ""
                    ),
                    "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                    "action_schema_version": ACTION_SCHEMA_VERSION,
                })
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/decide":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                body = self.rfile.read(length)
                request = json.loads(body.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request body must be an object")
                response = service.decide(request)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__})
                return
            self._json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args) -> None:
            return

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def serve(*, host: str = "127.0.0.1", port: int = 8767, policy_name: str = "heuristic", seed: int = 0) -> None:
    service = DecisionService(policy_from_name(policy_name, seed=seed, exploration=0.0))
    server = ThreadingHTTPServer((host, int(port)), make_handler(service))
    print(f"GTN AI {service.policy.name} listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a lightweight Garden of Thorn policy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--policy", default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, policy_name=args.policy, seed=args.seed)
    return 0


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


if __name__ == "__main__":
    raise SystemExit(main())
