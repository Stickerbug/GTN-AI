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
    torch_available,
)
from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, Action
from .structured_features import (
    STRUCTURED_FEATURE_SCHEMA_VERSION,
    EntityToken,
    StructuredDecision,
    StructuredFeatureConfig,
    TokenType,
    encode_structured_decision,
)


STRUCTURED_MODEL_SCHEMA_VERSION = 2

try:
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - covered by the v1 dependency test.
    nn = None


@dataclass(frozen=True)
class StructuredModelConfig:
    categorical_buckets: int = 1 << 15
    categorical_slots: int = 32
    numeric_buckets: int = 32
    max_state_tokens: int = 192
    max_history_events: int = 32
    model_dim: int = 192
    num_heads: int = 6
    state_layers: int = 4
    action_layers: int = 2
    feedforward_dim: int = 768
    dropout: float = 0.08

    def __post_init__(self) -> None:
        for name in (
            "categorical_buckets",
            "categorical_slots",
            "numeric_buckets",
            "max_state_tokens",
            "max_history_events",
            "model_dim",
            "num_heads",
            "state_layers",
            "action_layers",
            "feedforward_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.model_dim) % int(self.num_heads) != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def feature_config(self) -> StructuredFeatureConfig:
        return StructuredFeatureConfig(
            categorical_buckets=self.categorical_buckets,
            categorical_slots=self.categorical_slots,
            numeric_buckets=self.numeric_buckets,
            max_state_tokens=self.max_state_tokens,
            max_history_events=self.max_history_events,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructuredModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in allowed if key in value})


def collate_structured_decisions(
    examples: Sequence[StructuredDecision],
    *,
    config: StructuredModelConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    require_torch()
    items = list(examples)
    if not items:
        raise ValueError("cannot collate an empty structured batch")
    state_count = max(len(item.state_tokens) for item in items)
    action_count = max(item.action_count for item in items)
    if state_count <= 0 or action_count <= 0:
        raise ValueError("structured decisions require state and action tokens")
    batch_size = len(items)
    cat_slots = config.categorical_slots
    numeric_slots = config.numeric_buckets
    state_categorical = torch.zeros(
        (batch_size, state_count, cat_slots), dtype=torch.long
    )
    state_numeric = torch.zeros(
        (batch_size, state_count, numeric_slots), dtype=torch.float32
    )
    state_types = torch.zeros((batch_size, state_count), dtype=torch.long)
    state_mask = torch.zeros((batch_size, state_count), dtype=torch.bool)
    action_categorical = torch.zeros(
        (batch_size, action_count, cat_slots), dtype=torch.long
    )
    action_numeric = torch.zeros(
        (batch_size, action_count, numeric_slots), dtype=torch.float32
    )
    action_types = torch.zeros((batch_size, action_count), dtype=torch.long)
    action_mask = torch.zeros((batch_size, action_count), dtype=torch.bool)
    action_logit_biases = torch.zeros((batch_size, action_count), dtype=torch.float32)

    for row, item in enumerate(items):
        _copy_tokens(
            item.state_tokens,
            state_categorical[row],
            state_numeric[row],
            state_types[row],
            state_mask[row],
            config=config,
        )
        _copy_tokens(
            item.action_tokens,
            action_categorical[row],
            action_numeric[row],
            action_types[row],
            action_mask[row],
            config=config,
        )
        biases = item.action_logit_biases or (0.0,) * item.action_count
        if len(biases) != item.action_count:
            raise ValueError("action logit bias count does not match legal action count")
        action_logit_biases[row, :item.action_count] = torch.tensor(
            biases, dtype=torch.float32
        )

    tensor_device = torch.device(device)
    return {
        "state_categorical": state_categorical.to(tensor_device),
        "state_numeric": state_numeric.to(tensor_device),
        "state_types": state_types.to(tensor_device),
        "state_mask": state_mask.to(tensor_device),
        "action_categorical": action_categorical.to(tensor_device),
        "action_numeric": action_numeric.to(tensor_device),
        "action_types": action_types.to(tensor_device),
        "action_mask": action_mask.to(tensor_device),
        "action_logit_biases": action_logit_biases.to(tensor_device),
        "phases": torch.tensor(
            [item.phase for item in items], dtype=torch.long, device=tensor_device
        ),
        "selected_indices": torch.tensor(
            [item.selected_index for item in items], dtype=torch.long, device=tensor_device
        ),
    }


def _copy_tokens(
    tokens: Sequence[EntityToken],
    categorical,
    numeric,
    token_types,
    mask,
    *,
    config: StructuredModelConfig,
) -> None:
    for index, token in enumerate(tokens[:categorical.shape[0]]):
        ids = token.categorical_ids[:config.categorical_slots]
        if ids:
            categorical[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        values = token.numeric_values[:config.numeric_buckets]
        if values:
            numeric[index, :len(values)] = torch.tensor(values, dtype=torch.float32)
        token_types[index] = int(token.token_type)
        mask[index] = True


if nn is not None:
    class StructuredPolicyNetwork(nn.Module):
        """Entity-preserving state encoder with legal actions as decoder queries."""

        def __init__(self, config: StructuredModelConfig):
            super().__init__()
            self.config = config
            self.categorical_embedding = nn.Embedding(
                config.categorical_buckets + 1,
                config.model_dim,
                padding_idx=0,
            )
            self.type_embedding = nn.Embedding(
                max(int(value) for value in TokenType) + 1,
                config.model_dim,
                padding_idx=int(TokenType.PAD),
            )
            self.numeric_projection = nn.Linear(config.numeric_buckets, config.model_dim)
            self.input_norm = nn.LayerNorm(config.model_dim)
            self.input_dropout = nn.Dropout(config.dropout)
            state_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.state_encoder = nn.TransformerEncoder(
                state_layer,
                num_layers=config.state_layers,
                norm=nn.LayerNorm(config.model_dim),
                enable_nested_tensor=False,
            )
            action_layer = nn.TransformerDecoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.action_decoder = nn.TransformerDecoder(
                action_layer,
                num_layers=config.action_layers,
                norm=nn.LayerNorm(config.model_dim),
            )
            joint_dim = config.model_dim * 3
            self.pregame_policy_head = self._policy_head(joint_dim)
            self.combat_policy_head = self._policy_head(joint_dim)
            self.pregame_value_head = self._value_head()
            self.combat_value_head = self._value_head()
            self.reset_parameters()

        def _policy_head(self, input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, self.config.model_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.model_dim, 1),
            )

        def _value_head(self) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.config.model_dim, self.config.model_dim),
                nn.GELU(),
                nn.Linear(self.config.model_dim, 1),
                nn.Tanh(),
            )

        def reset_parameters(self) -> None:
            nn.init.normal_(self.categorical_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.numeric_projection.weight)
            nn.init.zeros_(self.numeric_projection.bias)
            with torch.no_grad():
                self.categorical_embedding.weight[0].zero_()
                self.type_embedding.weight[int(TokenType.PAD)].zero_()

        def forward(self, batch: dict[str, Any]):
            joint, state_summary = self.encode_features(batch)
            return self.score_features(joint, state_summary, batch)

        def encode_features(self, batch: dict[str, Any]):
            """Return frozen state/action representations for auxiliary policies."""

            state = self._embed(
                batch["state_categorical"],
                batch["state_numeric"],
                batch["state_types"],
            )
            state = self.state_encoder(
                state,
                src_key_padding_mask=~batch["state_mask"],
            )
            actions = self._embed(
                batch["action_categorical"],
                batch["action_numeric"],
                batch["action_types"],
            )
            actions = self.action_decoder(
                actions,
                state,
                tgt_key_padding_mask=~batch["action_mask"],
                memory_key_padding_mask=~batch["state_mask"],
            )
            state_summary = state[:, 0]
            expanded_state = state_summary.unsqueeze(1).expand_as(actions)
            joint = torch.cat(
                (actions, expanded_state, actions * expanded_state), dim=-1
            )
            return joint, state_summary

        def score_features(self, joint, state_summary, batch: dict[str, Any]):
            pregame_scores = self.pregame_policy_head(joint).squeeze(-1)
            combat_scores = self.combat_policy_head(joint).squeeze(-1)
            scores = torch.where(
                batch["phases"].unsqueeze(1) == 0,
                pregame_scores,
                combat_scores,
            )
            scores = scores.masked_fill(~batch["action_mask"], -1e9)
            pregame_values = self.pregame_value_head(state_summary).squeeze(-1)
            combat_values = self.combat_value_head(state_summary).squeeze(-1)
            values = torch.where(
                batch["phases"] == 0,
                pregame_values,
                combat_values,
            )
            return scores, values

        def _embed(self, categorical, numeric, token_types):
            embedded = self.categorical_embedding(categorical)
            categorical_mask = (categorical != 0).unsqueeze(-1)
            count = categorical_mask.sum(dim=-2).clamp_min(1)
            categorical_mean = (embedded * categorical_mask).sum(dim=-2) / count
            combined = (
                categorical_mean
                + self.numeric_projection(numeric)
                + self.type_embedding(token_types)
            )
            return self.input_dropout(self.input_norm(combined))
else:
    class StructuredPolicyNetwork:  # pragma: no cover
        def __init__(self, config: StructuredModelConfig):
            require_torch()


class StructuredPolicy:
    def __init__(
        self,
        model: StructuredPolicyNetwork,
        *,
        config: StructuredModelConfig,
        ruleset_fingerprint: str,
        device: str = "cpu",
        seed: int = 0,
        name: str = "structured-v2",
        temperature: float = 0.0,
        record_behavior: bool = False,
        model_fingerprint: str | None = None,
    ):
        require_torch()
        self.model = model
        self.config = config
        self.ruleset_fingerprint = str(ruleset_fingerprint or "")
        self.device = resolve_device(device)
        self.name = str(name or "structured-v2")
        self.temperature = max(0.0, float(temperature))
        self.record_behavior = bool(record_behavior)
        self.model_fingerprint = str(
            model_fingerprint or neural_state_dict_fingerprint(self.model.state_dict())
        )
        self.last_decision_metadata: dict[str, Any] | None = None
        self._rng = random.Random(int(seed))
        self._inference_lock = threading.Lock()
        self.model.to(torch.device(self.device))
        self.model.eval()

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        self.last_decision_metadata = None
        if not actions:
            raise ValueError("policy received an empty legal action list")
        logits, state_value = self.evaluate_actions(observation, actions)
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
        """Return policy logits and the acting player's state value."""

        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        self._validate_ruleset(observation)
        encoded = encode_structured_decision(
            observation,
            actions,
            config=self.config.feature_config,
        )
        batch = collate_structured_decisions(
            [encoded], config=self.config, device=self.device
        )
        with self._inference_lock, torch.inference_mode():
            scores, state_value = self.model(batch)
        logits = (
            scores[0, :len(actions)] + batch["action_logit_biases"][0, :len(actions)]
        ).detach().to("cpu").tolist()
        return [float(value) for value in logits], float(
            state_value[0].detach().to("cpu").item()
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
        """Batch leaf values so rollout search pays one model call per root."""

        items = list(decisions)
        if not items:
            return []
        encoded = []
        for observation, legal_actions in items:
            actions = list(legal_actions)
            if not actions:
                raise ValueError("policy received an empty legal action list")
            self._validate_ruleset(observation)
            encoded.append(encode_structured_decision(
                observation,
                actions,
                config=self.config.feature_config,
            ))
        batch = collate_structured_decisions(
            encoded, config=self.config, device=self.device
        )
        with self._inference_lock, torch.inference_mode():
            _, values = self.model(batch)
        return [float(value) for value in values.detach().to("cpu").tolist()]

    def fork(
        self,
        *,
        seed: int,
        temperature: float = 0.0,
        record_behavior: bool = False,
        name: str | None = None,
    ) -> "StructuredPolicy":
        return StructuredPolicy(
            self.model,
            config=self.config,
            ruleset_fingerprint=self.ruleset_fingerprint,
            device=self.device,
            seed=seed,
            name=name or self.name,
            temperature=temperature,
            record_behavior=record_behavior,
            model_fingerprint=self.model_fingerprint,
        )

    def _validate_ruleset(self, observation: dict[str, Any]) -> None:
        observed = str((observation.get("loadout") or {}).get("ruleset_fingerprint") or "")
        if self.ruleset_fingerprint and observed != self.ruleset_fingerprint:
            raise ValueError("model ruleset fingerprint does not match the current game rules")

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        save_structured_checkpoint(
            path,
            self.model,
            config=self.config,
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
    ) -> "StructuredPolicy":
        checkpoint = load_structured_checkpoint(path, device=device)
        return cls(
            checkpoint["model"],
            config=checkpoint["config"],
            ruleset_fingerprint=checkpoint["ruleset_fingerprint"],
            device=checkpoint["device"],
            seed=seed,
            name=checkpoint["name"],
            temperature=temperature,
            record_behavior=record_behavior,
        )


class StructuredEnsemblePolicy:
    """Combine compatible structured policies using normalized log probabilities."""

    def __init__(self, policies: Sequence[StructuredPolicy], *, seed: int = 0):
        members = list(policies)
        if len(members) < 2:
            raise ValueError("a structured ensemble requires at least two policies")
        config = members[0].config
        ruleset = members[0].ruleset_fingerprint
        device = members[0].device
        if any(member.config != config for member in members[1:]):
            raise ValueError("ensemble policies must use the same structured architecture")
        if any(member.ruleset_fingerprint != ruleset for member in members[1:]):
            raise ValueError("ensemble policies must use the same game ruleset")
        if any(member.device != device for member in members[1:]):
            raise ValueError("ensemble policies must use the same device")
        self.policies = members
        self.config = config
        self.ruleset_fingerprint = ruleset
        self.device = device
        self.name = "structured-ensemble[" + ",".join(member.name for member in members) + "]"
        self.model_fingerprint = "ensemble:" + "|".join(
            member.model_fingerprint for member in members
        )
        self.last_decision_metadata = None
        self._rng = random.Random(int(seed))

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
            if abs(value - peak) <= 1e-8
        ]
        return actions[self._rng.choice(best)]

    def evaluate_actions(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> tuple[list[float], float]:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        evaluations = [
            member.evaluate_actions(observation, actions)
            for member in self.policies
        ]
        normalized = [
            torch.log_softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
            for logits, _ in evaluations
        ]
        averaged_logits = torch.stack(normalized).mean(dim=0).tolist()
        averaged_value = sum(value for _, value in evaluations) / len(evaluations)
        return [float(value) for value in averaged_logits], float(averaged_value)

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
        estimates = [member.estimate_values(items) for member in self.policies]
        return [
            sum(member_values[index] for member_values in estimates) / len(estimates)
            for index in range(len(items))
        ]


def save_structured_checkpoint(
    path: str | Path,
    model: StructuredPolicyNetwork,
    *,
    config: StructuredModelConfig,
    ruleset_fingerprint: str,
    name: str = "structured-v2",
    metadata: dict[str, Any] | None = None,
) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": STRUCTURED_MODEL_SCHEMA_VERSION,
        "structured_feature_schema_version": STRUCTURED_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "name": str(name),
        "config": asdict(config),
        "ruleset_fingerprint": str(ruleset_fingerprint or ""),
        "state_dict": {
            key: value.detach().to("cpu")
            for key, value in model.state_dict().items()
        },
        "metadata": dict(metadata or {}),
    }, target)


def load_structured_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    require_torch()
    resolved_device = resolve_device(device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("structured checkpoint must contain an object")
    expected = {
        "schema_version": STRUCTURED_MODEL_SCHEMA_VERSION,
        "structured_feature_schema_version": STRUCTURED_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if int(payload.get(key, -1)) != int(value):
            raise ValueError(f"unsupported structured checkpoint {key}")
    config = StructuredModelConfig.from_dict(payload.get("config") or {})
    model = StructuredPolicyNetwork(config)
    model.load_state_dict(payload.get("state_dict") or {}, strict=True)
    model.to(torch.device(resolved_device))
    model.eval()
    return {
        "model": model,
        "config": config,
        "device": resolved_device,
        "name": str(payload.get("name") or "structured-v2"),
        "ruleset_fingerprint": str(payload.get("ruleset_fingerprint") or ""),
        "metadata": dict(payload.get("metadata") or {}),
    }
