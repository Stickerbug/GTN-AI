from __future__ import annotations

import math
import random
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .neural_model import (
    neural_state_dict_fingerprint,
    require_torch,
    resolve_device,
    torch,
)
from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, Action
from .structured_features import (
    STRUCTURED_FEATURE_SCHEMA_VERSION,
    encode_structured_decision,
)
from .structured_model import (
    STRUCTURED_MODEL_SCHEMA_VERSION,
    StructuredModelConfig,
    StructuredPolicyNetwork,
    collate_structured_decisions,
)


CORRECTION_MODEL_SCHEMA_VERSION = 1

try:
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - covered by dependency checks.
    nn = None


@dataclass(frozen=True)
class CorrectionModelConfig:
    hidden_dim: int = 192
    dropout: float = 0.08
    top_k: int = 3
    residual_scale: float = 1.0
    gate_threshold: float = 0.5

    def __post_init__(self) -> None:
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive")
        if float(self.residual_scale) <= 0.0:
            raise ValueError("residual_scale must be positive")
        if not 0.0 <= float(self.gate_threshold) <= 1.0:
            raise ValueError("gate_threshold must be in [0, 1]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CorrectionModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in allowed if key in value})


if nn is not None:
    class StructuredCorrectionNetwork(nn.Module):
        """Small residual ranker over a frozen structured policy."""

        ACTION_SCALARS = 4
        GATE_SCALARS = 5

        def __init__(
            self,
            base_config: StructuredModelConfig,
            config: CorrectionModelConfig,
        ):
            super().__init__()
            self.base_config = base_config
            self.config = config
            joint_dim = int(base_config.model_dim) * 3
            self.action_head = nn.Sequential(
                nn.LayerNorm(joint_dim + self.ACTION_SCALARS),
                nn.Linear(joint_dim + self.ACTION_SCALARS, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
            self.gate_head = nn.Sequential(
                nn.LayerNorm(int(base_config.model_dim) + self.GATE_SCALARS),
                nn.Linear(
                    int(base_config.model_dim) + self.GATE_SCALARS,
                    config.hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(
            self,
            joint,
            state_summary,
            base_scores,
            batch: dict[str, Any],
            *,
            hard_gate: bool = False,
        ):
            mask = batch["action_mask"]
            masked_scores = base_scores.masked_fill(~mask, -1e9)
            peak = masked_scores.max(dim=1, keepdim=True).values
            relative = (masked_scores - peak).clamp(min=-20.0, max=0.0)
            probabilities = torch.softmax(masked_scores, dim=1).masked_fill(~mask, 0.0)

            order = masked_scores.argsort(dim=1, descending=True)
            ranks = torch.zeros_like(masked_scores)
            ordinal = torch.arange(
                masked_scores.shape[1],
                dtype=masked_scores.dtype,
                device=masked_scores.device,
            ).unsqueeze(0).expand_as(masked_scores)
            ranks.scatter_(1, order, ordinal)
            action_count = mask.sum(dim=1, keepdim=True).clamp_min(1)
            rank_fraction = ranks / action_count.to(masked_scores.dtype)
            phase = batch["phases"].to(masked_scores.dtype).unsqueeze(1)
            action_phase = phase.expand_as(masked_scores)
            action_features = torch.cat(
                (
                    joint,
                    relative.unsqueeze(-1),
                    probabilities.unsqueeze(-1),
                    rank_fraction.unsqueeze(-1),
                    action_phase.unsqueeze(-1),
                ),
                dim=-1,
            )
            residual = self.action_head(action_features).squeeze(-1)

            candidate_count = min(int(self.config.top_k), masked_scores.shape[1])
            candidate_indices = masked_scores.topk(candidate_count, dim=1).indices
            candidate_mask = torch.zeros_like(mask)
            candidate_mask.scatter_(1, candidate_indices, True)
            residual = residual.masked_fill(~candidate_mask, 0.0)
            residual = residual.masked_fill(~mask, 0.0)

            top_values = masked_scores.topk(min(2, masked_scores.shape[1]), dim=1).values
            margin = (
                top_values[:, 0] - top_values[:, 1]
                if top_values.shape[1] > 1
                else torch.zeros_like(top_values[:, 0])
            )
            entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
            entropy_scale = action_count.squeeze(1).to(masked_scores.dtype).log().clamp_min(1.0)
            normalized_entropy = entropy / entropy_scale
            gate_scalars = torch.stack(
                (
                    margin.clamp(0.0, 20.0) / 20.0,
                    normalized_entropy.clamp(0.0, 1.0),
                    action_count.squeeze(1).to(masked_scores.dtype).log1p() / 5.0,
                    torch.tanh(peak.squeeze(1) / 5.0),
                    phase.squeeze(1),
                ),
                dim=1,
            )
            gate_logits = self.gate_head(
                torch.cat((state_summary, gate_scalars), dim=1)
            ).squeeze(1)
            gate_probability = torch.sigmoid(gate_logits)
            gate = (
                (gate_probability >= float(self.config.gate_threshold)).to(
                    masked_scores.dtype
                )
                if hard_gate
                else gate_probability
            )
            corrected = masked_scores + (
                float(self.config.residual_scale) * gate.unsqueeze(1) * residual
            )
            corrected = corrected.masked_fill(~mask, -1e9)
            return corrected, gate_logits, residual
else:
    class StructuredCorrectionNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


class StructuredCorrectionPolicy:
    def __init__(
        self,
        base_model: StructuredPolicyNetwork,
        correction_model: StructuredCorrectionNetwork,
        *,
        base_config: StructuredModelConfig,
        correction_config: CorrectionModelConfig,
        ruleset_fingerprint: str,
        device: str = "cpu",
        seed: int = 0,
        name: str = "structured-correction-v1",
        temperature: float = 0.0,
        record_behavior: bool = False,
        model_fingerprint: str | None = None,
    ):
        require_torch()
        self.base_model = base_model
        self.correction_model = correction_model
        self.base_config = base_config
        self.correction_config = correction_config
        self.ruleset_fingerprint = str(ruleset_fingerprint or "")
        self.device = resolve_device(device)
        self.name = str(name or "structured-correction-v1")
        self.temperature = max(0.0, float(temperature))
        self.record_behavior = bool(record_behavior)
        self._rng = random.Random(int(seed))
        self._inference_lock = threading.Lock()
        self.last_decision_metadata: dict[str, Any] | None = None
        self._decisions = 0
        self._gated_decisions = 0
        self._action_changes = 0
        self.base_model.to(torch.device(self.device)).eval()
        self.correction_model.to(torch.device(self.device)).eval()
        if model_fingerprint:
            self.model_fingerprint = str(model_fingerprint)
        else:
            merged = {
                **{f"base.{key}": value for key, value in self.base_model.state_dict().items()},
                **{
                    f"correction.{key}": value
                    for key, value in self.correction_model.state_dict().items()
                },
            }
            self.model_fingerprint = neural_state_dict_fingerprint(merged)

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        self.last_decision_metadata = None
        if not actions:
            raise ValueError("policy received an empty legal action list")
        logits, state_value, diagnostics = self._evaluate(observation, actions)
        if self.temperature > 0:
            probabilities = torch.softmax(
                torch.tensor(logits) / self.temperature,
                dim=0,
            ).tolist()
            selected_index = self._rng.choices(
                range(len(actions)), weights=probabilities, k=1
            )[0]
        else:
            peak = max(logits)
            best = [
                index for index, value in enumerate(logits)
                if abs(value - peak) <= 1e-8
            ]
            selected_index = self._rng.choice(best)
            probabilities = [
                1.0 / len(best) if index in best else 0.0
                for index in range(len(actions))
            ]
        self._decisions += 1
        self._gated_decisions += int(diagnostics["gated"])
        self._action_changes += int(diagnostics["action_changed"])
        action = actions[selected_index]
        if self.record_behavior:
            probability = max(1e-12, float(probabilities[selected_index]))
            entropy = -sum(
                value * math.log(value) for value in probabilities if value > 0
            )
            self.last_decision_metadata = {
                "action_key": action.key,
                "log_prob": math.log(probability),
                "value": float(state_value),
                "entropy": entropy,
                "temperature": self.temperature if self.temperature > 0 else 1.0,
            }
        return action

    def evaluate_actions(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> tuple[list[float], float]:
        logits, value, _ = self._evaluate(observation, list(legal_actions))
        return logits, value

    def _evaluate(
        self,
        observation: dict[str, Any],
        actions: Sequence[Action],
    ) -> tuple[list[float], float, dict[str, bool]]:
        if not actions:
            raise ValueError("policy received an empty legal action list")
        self._validate_ruleset(observation)
        encoded = encode_structured_decision(
            observation,
            actions,
            config=self.base_config.feature_config,
        )
        batch = collate_structured_decisions(
            [encoded], config=self.base_config, device=self.device
        )
        with self._inference_lock, torch.inference_mode():
            joint, state_summary = self.base_model.encode_features(batch)
            base_scores, values = self.base_model.score_features(
                joint, state_summary, batch
            )
            base_scores = base_scores + batch["action_logit_biases"]
            corrected, gate_logits, _ = self.correction_model(
                joint,
                state_summary,
                base_scores,
                batch,
                hard_gate=True,
            )
        base_index = int(base_scores[0, :len(actions)].argmax().item())
        corrected_index = int(corrected[0, :len(actions)].argmax().item())
        gated = bool(
            torch.sigmoid(gate_logits[0]).item()
            >= float(self.correction_config.gate_threshold)
        )
        return (
            [float(value) for value in corrected[0, :len(actions)].to("cpu").tolist()],
            float(values[0].to("cpu").item()),
            {"gated": gated, "action_changed": corrected_index != base_index},
        )

    def estimate_value(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> float:
        return self.estimate_values([(observation, legal_actions)])[0]

    def estimate_values(
        self,
        decisions: Sequence[tuple[dict[str, Any], Sequence[Action]]],
    ) -> list[float]:
        items = list(decisions)
        if not items:
            return []
        encoded = []
        for observation, actions in items:
            action_list = list(actions)
            if not action_list:
                raise ValueError("policy received an empty legal action list")
            self._validate_ruleset(observation)
            encoded.append(encode_structured_decision(
                observation,
                action_list,
                config=self.base_config.feature_config,
            ))
        batch = collate_structured_decisions(
            encoded, config=self.base_config, device=self.device
        )
        with self._inference_lock, torch.inference_mode():
            _, values = self.base_model(batch)
        return [float(value) for value in values.to("cpu").tolist()]

    def fork(
        self,
        *,
        seed: int,
        temperature: float = 0.0,
        record_behavior: bool = False,
        name: str | None = None,
    ) -> "StructuredCorrectionPolicy":
        return StructuredCorrectionPolicy(
            self.base_model,
            self.correction_model,
            base_config=self.base_config,
            correction_config=self.correction_config,
            ruleset_fingerprint=self.ruleset_fingerprint,
            device=self.device,
            seed=seed,
            name=name or self.name,
            temperature=temperature,
            record_behavior=record_behavior,
            model_fingerprint=self.model_fingerprint,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "decisions": self._decisions,
            "gated_decisions": self._gated_decisions,
            "action_changes": self._action_changes,
        }

    def _validate_ruleset(self, observation: dict[str, Any]) -> None:
        observed = str((observation.get("loadout") or {}).get("ruleset_fingerprint") or "")
        if self.ruleset_fingerprint and observed != self.ruleset_fingerprint:
            raise ValueError("model ruleset fingerprint does not match the current game rules")

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        save_correction_checkpoint(
            path,
            self.base_model,
            self.correction_model,
            base_config=self.base_config,
            correction_config=self.correction_config,
            ruleset_fingerprint=self.ruleset_fingerprint,
            name=self.name,
            metadata=metadata,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        seed: int = 0,
        temperature: float = 0.0,
        record_behavior: bool = False,
    ) -> "StructuredCorrectionPolicy":
        checkpoint = load_correction_checkpoint(path, device=device)
        return cls(
            checkpoint["base_model"],
            checkpoint["correction_model"],
            base_config=checkpoint["base_config"],
            correction_config=checkpoint["correction_config"],
            ruleset_fingerprint=checkpoint["ruleset_fingerprint"],
            device=checkpoint["device"],
            seed=seed,
            name=checkpoint["name"],
            temperature=temperature,
            record_behavior=record_behavior,
        )


def save_correction_checkpoint(
    path: str | Path,
    base_model: StructuredPolicyNetwork,
    correction_model: StructuredCorrectionNetwork,
    *,
    base_config: StructuredModelConfig,
    correction_config: CorrectionModelConfig,
    ruleset_fingerprint: str,
    name: str = "structured-correction-v1",
    metadata: dict[str, Any] | None = None,
) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": CORRECTION_MODEL_SCHEMA_VERSION,
        "base_model_schema_version": STRUCTURED_MODEL_SCHEMA_VERSION,
        "structured_feature_schema_version": STRUCTURED_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "name": str(name),
        "base_config": asdict(base_config),
        "correction_config": asdict(correction_config),
        "ruleset_fingerprint": str(ruleset_fingerprint or ""),
        "base_state_dict": {
            key: value.detach().to("cpu")
            for key, value in base_model.state_dict().items()
        },
        "correction_state_dict": {
            key: value.detach().to("cpu")
            for key, value in correction_model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }, target)


def load_correction_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    require_torch()
    resolved_device = resolve_device(device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("correction checkpoint must contain an object")
    expected = {
        "schema_version": CORRECTION_MODEL_SCHEMA_VERSION,
        "base_model_schema_version": STRUCTURED_MODEL_SCHEMA_VERSION,
        "structured_feature_schema_version": STRUCTURED_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    for key, expected_value in expected.items():
        if int(payload.get(key, -1)) != int(expected_value):
            raise ValueError(f"unsupported correction checkpoint {key}")
    base_config = StructuredModelConfig.from_dict(payload.get("base_config") or {})
    correction_config = CorrectionModelConfig.from_dict(
        payload.get("correction_config") or {}
    )
    base_model = StructuredPolicyNetwork(base_config)
    base_model.load_state_dict(payload.get("base_state_dict") or {}, strict=True)
    correction_model = StructuredCorrectionNetwork(base_config, correction_config)
    correction_model.load_state_dict(
        payload.get("correction_state_dict") or {}, strict=True
    )
    base_model.to(torch.device(resolved_device)).eval()
    correction_model.to(torch.device(resolved_device)).eval()
    return {
        "base_model": base_model,
        "correction_model": correction_model,
        "base_config": base_config,
        "correction_config": correction_config,
        "device": resolved_device,
        "name": str(payload.get("name") or "structured-correction-v1"),
        "ruleset_fingerprint": str(payload.get("ruleset_fingerprint") or ""),
        "metadata": dict(payload.get("metadata") or {}),
    }
