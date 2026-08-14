from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from .policies import Policy
from .protocol import ACTION_SCHEMA_VERSION, SERVICE_API_VERSION, Action


DEFAULT_ENDPOINT = "http://127.0.0.1:8767/v1/decide"
MAX_RESPONSE_BYTES = 256 * 1024


class InferenceClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionResult:
    action: Action
    source: str
    latency_ms: float
    model: str = ""
    error: str = ""


class InferenceClient:
    """Small fail-closed client suitable for a future game-server adapter."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout_seconds: float = 0.2,
        fallback_policy: Policy | None = None,
    ):
        self.endpoint = str(endpoint)
        self.timeout_seconds = max(0.001, float(timeout_seconds))
        self.fallback_policy = fallback_policy

    def decide(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> DecisionResult:
        actions = list(legal_actions)
        if not actions:
            raise InferenceClientError("cannot request a decision without legal actions")
        legal_by_key = {action.key: action for action in actions}
        started = time.perf_counter()
        try:
            selected, model = self._request(observation, actions)
            canonical = legal_by_key.get(selected.key)
            if canonical is None:
                raise InferenceClientError("sidecar returned an action outside the supplied legal set")
            return DecisionResult(
                action=canonical,
                source="sidecar",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model=model,
            )
        except Exception as exc:
            if self.fallback_policy is None:
                if isinstance(exc, InferenceClientError):
                    raise
                raise InferenceClientError(f"sidecar request failed: {type(exc).__name__}: {exc}") from exc
            fallback = self.fallback_policy.select_action(observation, actions)
            canonical = legal_by_key.get(fallback.key)
            if canonical is None:
                raise InferenceClientError("fallback policy returned an illegal action") from exc
            return DecisionResult(
                action=canonical,
                source="fallback",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model=str(self.fallback_policy.name),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _request(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> tuple[Action, str]:
        payload = json.dumps(
            {
                "api_version": SERVICE_API_VERSION,
                "observation": observation,
                "legal_actions": [action.to_dict() for action in legal_actions],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if int(response.status) != 200:
                    raise InferenceClientError(f"sidecar returned HTTP {response.status}")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise InferenceClientError(f"sidecar returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise InferenceClientError(f"sidecar is unavailable: {exc}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise InferenceClientError("sidecar response exceeds the size limit")
        try:
            response_payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InferenceClientError("sidecar returned invalid JSON") from exc
        if not isinstance(response_payload, dict):
            raise InferenceClientError("sidecar response must be an object")
        if int(response_payload.get("api_version", -1)) != SERVICE_API_VERSION:
            raise InferenceClientError("sidecar API version mismatch")
        if int(response_payload.get("action_schema_version", -1)) != ACTION_SCHEMA_VERSION:
            raise InferenceClientError("sidecar action schema mismatch")
        try:
            action = Action.from_dict(response_payload.get("action") or {})
        except (TypeError, ValueError) as exc:
            raise InferenceClientError("sidecar returned an invalid action") from exc
        return action, str(response_payload.get("model") or "")
