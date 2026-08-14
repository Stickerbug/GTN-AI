from __future__ import annotations

import hashlib
import random
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .features import (
    NEURAL_FEATURE_SCHEMA_VERSION,
    hashed_action_only_features,
    hashed_observation_features,
    history_event_token_ids,
)
from .protocol import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, Action


NEURAL_MODEL_SCHEMA_VERSION = 1

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # The production game server does not need training dependencies.
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None and nn is not None


def require_torch() -> None:
    if not torch_available():
        raise RuntimeError(
            "the neural policy requires PyTorch; use the training extra in a Python 3.12 environment"
        )


def resolve_device(requested: str = "auto") -> str:
    require_torch()
    choice = str(requested or "auto").strip().lower()
    if choice == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if choice == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("the requested Intel XPU is not available")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is not available")
    if choice == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("the requested MPS device is not available")
    if choice not in {"cpu", "xpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    return choice


@dataclass(frozen=True)
class NeuralModelConfig:
    observation_buckets: int = 1 << 15
    action_buckets: int = 1 << 14
    history_buckets: int = 1 << 13
    observation_embedding_dim: int = 64
    action_embedding_dim: int = 64
    history_embedding_dim: int = 32
    hidden_dim: int = 128
    max_history_events: int = 32
    max_history_tokens_per_event: int = 12
    dropout: float = 0.08

    def __post_init__(self) -> None:
        integer_fields = (
            "observation_buckets",
            "action_buckets",
            "history_buckets",
            "observation_embedding_dim",
            "action_embedding_dim",
            "history_embedding_dim",
            "hidden_dim",
            "max_history_events",
            "max_history_tokens_per_event",
        )
        for field_name in integer_fields:
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if int(self.hidden_dim) < 2:
            raise ValueError("hidden_dim must be at least 2")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NeuralModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in allowed if key in value})


@dataclass(frozen=True)
class EncodedDecision:
    observation_indices: tuple[int, ...]
    observation_values: tuple[float, ...]
    history_tokens: tuple[tuple[int, ...], ...]
    action_indices: tuple[tuple[int, ...], ...]
    action_values: tuple[tuple[float, ...], ...]
    phase: int
    selected_index: int = -1
    value_target: float = 0.0
    policy_weight: float = 1.0
    action_logit_biases: tuple[float, ...] = ()
    behavior_log_prob: float = 0.0
    behavior_value: float = 0.0
    behavior_temperature: float = 1.0
    advantage_target: float = 0.0

    @property
    def action_count(self) -> int:
        return len(self.action_indices)


def encode_decision(
    observation: dict[str, Any],
    legal_actions: Sequence[Action],
    *,
    config: NeuralModelConfig,
    selected_index: int = -1,
    value_target: float = 0.0,
    policy_weight: float = 1.0,
    behavior_log_prob: float = 0.0,
    behavior_value: float = 0.0,
    behavior_temperature: float = 1.0,
    advantage_target: float | None = None,
) -> EncodedDecision:
    actions = list(legal_actions)
    if not actions:
        raise ValueError("cannot encode a decision without legal actions")
    if selected_index < -1:
        raise ValueError("selected action index must be -1 or a legal action index")
    if selected_index >= len(actions):
        raise ValueError("selected action index is outside the legal action set")
    observation_features = hashed_observation_features(
        observation,
        buckets=config.observation_buckets,
    )
    action_features = [
        hashed_action_only_features(observation, action, buckets=config.action_buckets)
        for action in actions
    ]
    history = history_event_token_ids(
        observation,
        buckets=config.history_buckets,
        max_events=config.max_history_events,
        max_tokens_per_event=config.max_history_tokens_per_event,
    )
    return EncodedDecision(
        observation_indices=tuple(sorted(observation_features)),
        observation_values=tuple(
            float(observation_features[index]) for index in sorted(observation_features)
        ),
        history_tokens=tuple(tuple(tokens) for tokens in history),
        action_indices=tuple(tuple(sorted(features)) for features in action_features),
        action_values=tuple(
            tuple(float(features[index]) for index in sorted(features))
            for features in action_features
        ),
        phase=0 if observation.get("phase") == "pregame" else 1,
        selected_index=int(selected_index),
        value_target=max(-1.0, min(1.0, float(value_target))),
        policy_weight=max(0.0, float(policy_weight)),
        action_logit_biases=tuple(_progress_prior_biases(observation, actions)),
        behavior_log_prob=float(behavior_log_prob),
        behavior_value=max(-1.0, min(1.0, float(behavior_value))),
        behavior_temperature=max(1e-6, float(behavior_temperature)),
        advantage_target=float(
            value_target - behavior_value
            if advantage_target is None
            else advantage_target
        ),
    )


def collate_decisions(
    examples: Sequence[EncodedDecision],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    require_torch()
    items = list(examples)
    if not items:
        raise ValueError("cannot collate an empty decision batch")
    observation_indices: list[int] = []
    observation_values: list[float] = []
    observation_offsets = [0]
    action_indices: list[int] = []
    action_values: list[float] = []
    action_embedding_offsets = [0]
    action_owners: list[int] = []
    action_set_offsets = [0]
    selected_absolute: list[int] = []
    action_logit_biases: list[float] = []

    max_events = max(1, max(len(item.history_tokens) for item in items))
    max_tokens = max(
        1,
        max(
            (len(tokens) for item in items for tokens in item.history_tokens),
            default=0,
        ),
    )
    history = torch.zeros((len(items), max_events, max_tokens), dtype=torch.long)
    history_lengths: list[int] = []

    for owner, item in enumerate(items):
        observation_indices.extend(item.observation_indices)
        observation_values.extend(item.observation_values)
        observation_offsets.append(len(observation_indices))
        history_lengths.append(len(item.history_tokens))
        for event_index, tokens in enumerate(item.history_tokens):
            if tokens:
                history[owner, event_index, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        action_start = len(action_owners)
        biases = item.action_logit_biases or (0.0,) * item.action_count
        if len(biases) != item.action_count:
            raise ValueError("action logit bias count does not match legal action count")
        for indices, values, bias in zip(item.action_indices, item.action_values, biases):
            action_indices.extend(indices)
            action_values.extend(values)
            action_embedding_offsets.append(len(action_indices))
            action_owners.append(owner)
            action_logit_biases.append(float(bias))
        action_set_offsets.append(len(action_owners))
        selected_absolute.append(
            action_start + item.selected_index if item.selected_index >= 0 else -1
        )

    tensor_device = torch.device(device)
    return {
        "observation_indices": torch.tensor(observation_indices, dtype=torch.long, device=tensor_device),
        "observation_values": torch.tensor(observation_values, dtype=torch.float32, device=tensor_device),
        "observation_offsets": torch.tensor(observation_offsets, dtype=torch.long, device=tensor_device),
        "history_tokens": history.to(tensor_device),
        "history_lengths": torch.tensor(history_lengths, dtype=torch.long, device=tensor_device),
        "action_indices": torch.tensor(action_indices, dtype=torch.long, device=tensor_device),
        "action_values": torch.tensor(action_values, dtype=torch.float32, device=tensor_device),
        "action_embedding_offsets": torch.tensor(action_embedding_offsets, dtype=torch.long, device=tensor_device),
        "action_owners": torch.tensor(action_owners, dtype=torch.long, device=tensor_device),
        "action_set_offsets": torch.tensor(action_set_offsets, dtype=torch.long, device=tensor_device),
        "selected_absolute": torch.tensor(selected_absolute, dtype=torch.long, device=tensor_device),
        "action_logit_biases": torch.tensor(
            action_logit_biases, dtype=torch.float32, device=tensor_device
        ),
        "phases": torch.tensor([item.phase for item in items], dtype=torch.long, device=tensor_device),
        "value_targets": torch.tensor(
            [item.value_target for item in items], dtype=torch.float32, device=tensor_device
        ),
        "policy_weights": torch.tensor(
            [item.policy_weight for item in items], dtype=torch.float32, device=tensor_device
        ),
        "behavior_log_probs": torch.tensor(
            [item.behavior_log_prob for item in items], dtype=torch.float32, device=tensor_device
        ),
        "behavior_values": torch.tensor(
            [item.behavior_value for item in items], dtype=torch.float32, device=tensor_device
        ),
        "behavior_temperatures": torch.tensor(
            [item.behavior_temperature for item in items], dtype=torch.float32, device=tensor_device
        ),
        "advantage_targets": torch.tensor(
            [item.advantage_target for item in items], dtype=torch.float32, device=tensor_device
        ),
    }


if nn is not None:
    class VariableActionNetwork(nn.Module):
        """Shared information-set encoder with independent pregame/combat heads."""

        def __init__(self, config: NeuralModelConfig):
            super().__init__()
            self.config = config
            self.observation_embedding = nn.EmbeddingBag(
                config.observation_buckets,
                config.observation_embedding_dim,
                mode="sum",
                include_last_offset=True,
            )
            self.action_embedding = nn.EmbeddingBag(
                config.action_buckets,
                config.action_embedding_dim,
                mode="sum",
                include_last_offset=True,
            )
            self.history_embedding = nn.Embedding(
                config.history_buckets + 1,
                config.history_embedding_dim,
                padding_idx=0,
            )
            self.history_encoder = nn.GRU(
                config.history_embedding_dim,
                config.history_embedding_dim,
                batch_first=True,
            )
            self.state_encoder = nn.Sequential(
                nn.Linear(
                    config.observation_embedding_dim + config.history_embedding_dim,
                    config.hidden_dim,
                ),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(config.action_embedding_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
            )
            joint_dim = config.hidden_dim * 3
            self.pregame_policy_head = self._policy_head(joint_dim)
            self.combat_policy_head = self._policy_head(joint_dim)
            self.pregame_value_head = self._value_head()
            self.combat_value_head = self._value_head()
            self.reset_parameters()

        def _policy_head(self, input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, 1),
            )

        def _value_head(self) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim // 2, 1),
                nn.Tanh(),
            )

        def reset_parameters(self) -> None:
            nn.init.normal_(self.observation_embedding.weight, mean=0.0, std=0.025)
            nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.025)
            nn.init.normal_(self.history_embedding.weight, mean=0.0, std=0.025)
            with torch.no_grad():
                self.history_embedding.weight[0].zero_()

        def forward(self, batch: dict[str, Any]):
            observation = self.observation_embedding(
                batch["observation_indices"],
                batch["observation_offsets"],
                per_sample_weights=batch["observation_values"],
            )
            actions = self.action_embedding(
                batch["action_indices"],
                batch["action_embedding_offsets"],
                per_sample_weights=batch["action_values"],
            )
            history = self._encode_history(
                batch["history_tokens"],
                batch["history_lengths"],
            )
            state = self.state_encoder(torch.cat((observation, history), dim=-1))
            action_state = self.action_encoder(actions)
            owned_state = state.index_select(0, batch["action_owners"])
            joint = torch.cat((owned_state, action_state, owned_state * action_state), dim=-1)
            pregame_scores = self.pregame_policy_head(joint).squeeze(-1)
            combat_scores = self.combat_policy_head(joint).squeeze(-1)
            action_phases = batch["phases"].index_select(0, batch["action_owners"])
            scores = torch.where(action_phases == 0, pregame_scores, combat_scores)
            pregame_values = self.pregame_value_head(state).squeeze(-1)
            combat_values = self.combat_value_head(state).squeeze(-1)
            values = torch.where(batch["phases"] == 0, pregame_values, combat_values)
            return scores, values

        def _encode_history(self, tokens, lengths):
            embedded = self.history_embedding(tokens)
            token_mask = (tokens != 0).unsqueeze(-1)
            token_count = token_mask.sum(dim=2).clamp_min(1)
            events = (embedded * token_mask).sum(dim=2) / token_count
            encoded, _ = self.history_encoder(events)
            final_indices = (lengths - 1).clamp_min(0)
            row_indices = torch.arange(encoded.shape[0], device=encoded.device)
            final = encoded[row_indices, final_indices]
            return torch.where((lengths > 0).unsqueeze(-1), final, torch.zeros_like(final))
else:
    class VariableActionNetwork:  # pragma: no cover - exercised by the dependency error test.
        def __init__(self, config: NeuralModelConfig):
            require_torch()


class NeuralPolicy:
    def __init__(
        self,
        model: VariableActionNetwork,
        *,
        config: NeuralModelConfig,
        ruleset_fingerprint: str,
        device: str = "cpu",
        seed: int = 0,
        name: str = "variable-action-v1",
        temperature: float = 0.0,
        epsilon: float = 0.0,
        record_behavior: bool = False,
        model_fingerprint: str | None = None,
    ):
        require_torch()
        self.model = model
        self.config = config
        self.ruleset_fingerprint = str(ruleset_fingerprint or "")
        self.device = resolve_device(device)
        self.name = str(name or "variable-action-v1")
        self.temperature = max(0.0, float(temperature))
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.record_behavior = bool(record_behavior)
        if self.record_behavior and self.epsilon > 0:
            raise ValueError("recorded on-policy sampling does not support epsilon exploration")
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
        self._validate_ruleset(observation)
        encoded = encode_decision(observation, actions, config=self.config)
        batch = collate_decisions([encoded], device=self.device)
        with self._inference_lock, torch.inference_mode():
            scores, state_value = self.model(batch)
        values = scores.detach().to("cpu").tolist()
        values = _apply_progress_prior(observation, actions, values)
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            selected_index = self._rng.randrange(len(actions))
            return actions[selected_index]
        if self.temperature > 0:
            peak = max(values)
            weights = [
                math.exp(max(-60.0, (float(value) - peak) / self.temperature))
                for value in values
            ]
            selected_index = self._rng.choices(range(len(actions)), weights=weights, k=1)[0]
            probabilities = [weight / sum(weights) for weight in weights]
        else:
            peak = max(values)
            best = [index for index, value in enumerate(values) if abs(value - peak) <= 1e-8]
            selected_index = self._rng.choice(best)
            probabilities = [1.0 / len(best) if index in best else 0.0 for index in range(len(actions))]
        action = actions[selected_index]
        if self.record_behavior:
            probability = max(1e-12, probabilities[selected_index])
            entropy = -sum(
                probability_value * math.log(probability_value)
                for probability_value in probabilities
                if probability_value > 0
            )
            self.last_decision_metadata = {
                "action_key": action.key,
                "log_prob": math.log(probability),
                "value": float(state_value[0].detach().to("cpu").item()),
                "entropy": entropy,
                "temperature": self.temperature if self.temperature > 0 else 1.0,
            }
        return action

    def estimate_value(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> float:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        self._validate_ruleset(observation)
        batch = collate_decisions(
            [encode_decision(observation, actions, config=self.config)],
            device=self.device,
        )
        with self._inference_lock, torch.inference_mode():
            _, value = self.model(batch)
        return float(value[0].detach().to("cpu").item())

    def fork(
        self,
        *,
        seed: int,
        temperature: float = 0.0,
        epsilon: float = 0.0,
        record_behavior: bool = False,
        name: str | None = None,
    ) -> "NeuralPolicy":
        """Create an independently seeded policy while sharing read-only model weights."""

        return NeuralPolicy(
            self.model,
            config=self.config,
            ruleset_fingerprint=self.ruleset_fingerprint,
            device=self.device,
            seed=seed,
            name=name or self.name,
            temperature=temperature,
            epsilon=epsilon,
            record_behavior=record_behavior,
            model_fingerprint=self.model_fingerprint,
        )

    def _validate_ruleset(self, observation: dict[str, Any]) -> None:
        observed = str((observation.get("loadout") or {}).get("ruleset_fingerprint") or "")
        if self.ruleset_fingerprint and observed != self.ruleset_fingerprint:
            raise ValueError("model ruleset fingerprint does not match the current game rules")

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        save_neural_checkpoint(
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
        epsilon: float = 0.0,
        record_behavior: bool = False,
    ) -> "NeuralPolicy":
        checkpoint = load_neural_checkpoint(path, device=device)
        return cls(
            checkpoint["model"],
            config=checkpoint["config"],
            ruleset_fingerprint=checkpoint["ruleset_fingerprint"],
            device=checkpoint["device"],
            seed=seed,
            name=checkpoint["name"],
            temperature=temperature,
            epsilon=epsilon,
            record_behavior=record_behavior,
        )


class NeuralEnsemblePolicy:
    """Average compatible policy logits without exposing additional game information."""

    def __init__(self, policies: Sequence[NeuralPolicy], *, seed: int = 0):
        members = list(policies)
        if len(members) < 2:
            raise ValueError("a neural ensemble requires at least two policies")
        config = members[0].config
        ruleset = members[0].ruleset_fingerprint
        device = members[0].device
        if any(member.config != config for member in members[1:]):
            raise ValueError("ensemble policies must use the same neural architecture")
        if any(member.ruleset_fingerprint != ruleset for member in members[1:]):
            raise ValueError("ensemble policies must use the same game ruleset")
        if any(member.device != device for member in members[1:]):
            raise ValueError("ensemble policies must use the same device")
        self.policies = members
        self.config = config
        self.ruleset_fingerprint = ruleset
        self.device = device
        self.name = "ensemble[" + ",".join(member.name for member in members) + "]"
        self._rng = random.Random(int(seed))

    def select_action(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        self._validate_ruleset(observation)
        batch = collate_decisions(
            [encode_decision(observation, actions, config=self.config)],
            device=self.device,
        )
        score_sets = []
        for member in self.policies:
            with member._inference_lock, torch.inference_mode():
                scores, _ = member.model(batch)
            score_sets.append(scores)
        averaged = torch.stack(score_sets).mean(dim=0).detach().to("cpu").tolist()
        values = _apply_progress_prior(observation, actions, averaged)
        peak = max(values)
        best = [index for index, value in enumerate(values) if abs(value - peak) <= 1e-8]
        return actions[self._rng.choice(best)]

    def estimate_value(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> float:
        actions = list(legal_actions)
        if not actions:
            raise ValueError("policy received an empty legal action list")
        self._validate_ruleset(observation)
        batch = collate_decisions(
            [encode_decision(observation, actions, config=self.config)],
            device=self.device,
        )
        estimates = []
        for member in self.policies:
            with member._inference_lock, torch.inference_mode():
                _, value = member.model(batch)
            estimates.append(value[0])
        return float(torch.stack(estimates).mean().detach().to("cpu").item())

    def _validate_ruleset(self, observation: dict[str, Any]) -> None:
        observed = str((observation.get("loadout") or {}).get("ruleset_fingerprint") or "")
        if self.ruleset_fingerprint and observed != self.ruleset_fingerprint:
            raise ValueError("model ruleset fingerprint does not match the current game rules")


def _apply_progress_prior(
    observation: dict[str, Any],
    actions: Sequence[Action],
    scores: Sequence[float],
) -> list[float]:
    """Discourage immediately undoing a partial multi-select decision."""

    return [
        float(score) + bias
        for score, bias in zip(scores, _progress_prior_biases(observation, actions))
    ]


def _progress_prior_biases(
    observation: dict[str, Any],
    actions: Sequence[Action],
) -> list[float]:
    biases = [0.0] * len(actions)
    pending = observation.get("pending") or {}
    own = observation.get("self") or {}
    pregame = own.get("sub_choice") or {}
    for index, action in enumerate(actions):
        if action.kind == "toggle_choice":
            selection = pending.get("selection") or {}
        elif action.kind == "toggle_pregame_choice":
            selection = pregame.get("selection") or {}
        else:
            continue
        selected = {_slot_key(slot) for slot in selection.get("selected_slots") or []}
        if _slot_key(action.payload.get("candidate_slot")) in selected:
            biases[index] -= 8.0
    return biases


def _slot_key(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def save_neural_checkpoint(
    path: str | Path,
    model: VariableActionNetwork,
    *,
    config: NeuralModelConfig,
    ruleset_fingerprint: str,
    name: str = "variable-action-v1",
    metadata: dict[str, Any] | None = None,
) -> None:
    require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": NEURAL_MODEL_SCHEMA_VERSION,
        "neural_feature_schema_version": NEURAL_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "name": str(name),
        "config": asdict(config),
        "ruleset_fingerprint": str(ruleset_fingerprint or ""),
        "state_dict": {key: value.detach().to("cpu") for key, value in model.state_dict().items()},
        "metadata": dict(metadata or {}),
    }, target)


def neural_state_dict_fingerprint(state_dict: dict[str, Any]) -> str:
    """Return a stable digest for exact actor-policy provenance."""

    require_torch()
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_neural_checkpoint(path: str | Path, *, device: str = "auto") -> dict[str, Any]:
    require_torch()
    resolved_device = resolve_device(device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("neural checkpoint must contain an object")
    expected = {
        "schema_version": NEURAL_MODEL_SCHEMA_VERSION,
        "neural_feature_schema_version": NEURAL_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if int(payload.get(key, -1)) != int(value):
            raise ValueError(f"unsupported neural checkpoint {key}")
    config = NeuralModelConfig.from_dict(payload.get("config") or {})
    model = VariableActionNetwork(config)
    model.load_state_dict(payload.get("state_dict") or {}, strict=True)
    model.to(torch.device(resolved_device))
    model.eval()
    return {
        "model": model,
        "config": config,
        "device": resolved_device,
        "name": str(payload.get("name") or "variable-action-v1"),
        "ruleset_fingerprint": str(payload.get("ruleset_fingerprint") or ""),
        "metadata": dict(payload.get("metadata") or {}),
    }
