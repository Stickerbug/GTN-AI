from __future__ import annotations

import math
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

from .neural_model import require_torch, resolve_device, torch
from .progress import ProgressReporter
from .structured_cache import (
    StructuredDistillationExample,
    cached_split_count,
    iter_cached_batches,
    iter_cached_examples,
    load_cache_manifest,
)
from .structured_model import (
    StructuredModelConfig,
    StructuredPolicy,
    StructuredPolicyNetwork,
    collate_structured_decisions,
    load_structured_checkpoint,
)


def train_structured_distillation(
    cache_dir: str | Path,
    *,
    output_checkpoint: str | Path,
    epochs: int = 4,
    batch_size: int = 256,
    shuffle_buffer: int = 2048,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    temperature: float = 1.5,
    soft_policy_weight: float = 1.0,
    hard_policy_weight: float = 0.1,
    value_loss_weight: float = 0.25,
    max_grad_norm: float = 1.0,
    validation_fraction: float = 0.05,
    initial_checkpoint: str | Path | None = None,
    replay_cache_dir: str | Path | None = None,
    replay_ratio: float = 0.0,
    trainable_scope: str = "all",
    device: str = "auto",
    seed: int = 1,
    progress_interval: float = 10.0,
    show_progress: bool = True,
) -> tuple[StructuredPolicy, dict[str, Any]]:
    require_torch()
    manifest = load_cache_manifest(cache_dir)
    config = StructuredModelConfig.from_dict(manifest.get("structured_config") or {})
    normalized_replay_ratio = max(0.0, float(replay_ratio))
    if normalized_replay_ratio > 0.0 and replay_cache_dir is None:
        raise ValueError("replay ratio requires a replay cache")
    replay_manifest = (
        load_cache_manifest(replay_cache_dir)
        if replay_cache_dir is not None and normalized_replay_ratio > 0.0
        else None
    )
    if replay_manifest is not None:
        replay_config = StructuredModelConfig.from_dict(
            replay_manifest.get("structured_config") or {}
        )
        if replay_config != config:
            raise ValueError("replay cache structured config does not match primary cache")
        if replay_manifest["ruleset_fingerprint"] != manifest["ruleset_fingerprint"]:
            raise ValueError("replay cache ruleset does not match primary cache")
    resolved_device = resolve_device(device)
    torch.manual_seed(int(seed))
    if resolved_device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(int(seed))
    elif resolved_device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if initial_checkpoint:
        initial = load_structured_checkpoint(initial_checkpoint, device=resolved_device)
        if initial["config"] != config:
            raise ValueError("initial structured checkpoint config does not match cache")
        if initial["ruleset_fingerprint"] != manifest["ruleset_fingerprint"]:
            raise ValueError("initial structured checkpoint ruleset does not match cache")
        model = initial["model"]
    else:
        model = StructuredPolicyNetwork(config).to(torch.device(resolved_device))
    trainable_parameters = _configure_trainable_parameters(model, trainable_scope)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    started = time.perf_counter()
    epoch_metrics: list[dict[str, Any]] = []
    primary_train_count = cached_split_count(
        manifest,
        split="train" if validation_fraction > 0 else "all",
        validation_fraction=validation_fraction,
    )
    replay_examples_per_epoch = 0
    if replay_manifest is not None:
        replay_available = cached_split_count(
            replay_manifest,
            split="train" if validation_fraction > 0 else "all",
            validation_fraction=validation_fraction,
        )
        replay_examples_per_epoch = min(
            replay_available,
            max(0, int(round(primary_train_count * normalized_replay_ratio))),
        )
    train_example_count = primary_train_count + replay_examples_per_epoch
    for epoch in range(max(0, int(epochs))):
        model.train()
        training = _DistillationAccumulator()
        progress = ProgressReporter(
            f"distill {epoch + 1}/{max(0, int(epochs))}",
            total=train_example_count or None,
            interval=progress_interval,
            enabled=show_progress,
        )
        progress.update(0, force=True, stage="loading first bucket")
        primary_examples = iter_cached_examples(
            cache_dir,
            split="train" if validation_fraction > 0 else "all",
            validation_fraction=validation_fraction,
            shuffle_shards=True,
            seed=int(seed) + epoch * 104729,
        )
        if replay_examples_per_epoch:
            replay_examples = islice(
                iter_cached_examples(
                    replay_cache_dir,
                    split="train" if validation_fraction > 0 else "all",
                    validation_fraction=validation_fraction,
                    shuffle_shards=True,
                    seed=int(seed) + epoch * 15485863,
                ),
                replay_examples_per_epoch,
            )
            examples = _interleave_example_streams(
                primary_examples,
                replay_examples,
                primary_count=primary_train_count,
                replay_count=replay_examples_per_epoch,
                rng=random.Random(int(seed) + epoch * 32452843),
            )
        else:
            examples = primary_examples
        loaded_examples = 0

        def tracked_examples():
            nonlocal loaded_examples
            for example in examples:
                loaded_examples += 1
                if loaded_examples % 128 == 0:
                    progress.update(
                        training.examples,
                        stage=f"buffering {loaded_examples - training.examples:,}",
                    )
                yield example

        batches = iter_cached_batches(
            tracked_examples(),
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            rng=random.Random(int(seed) + epoch * 130363),
        )
        for examples_batch in batches:
            batch = collate_distillation_examples(
                examples_batch, config=config, device=resolved_device
            )
            optimizer.zero_grad(set_to_none=True)
            scores, values = model(batch)
            loss, stats = _distillation_loss(
                scores,
                values,
                batch,
                temperature=temperature,
                soft_policy_weight=soft_policy_weight,
                hard_policy_weight=hard_policy_weight,
                value_loss_weight=value_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("structured distillation produced a non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(max_grad_norm)
            )
            optimizer.step()
            training.add(
                len(examples_batch),
                stats,
                float(gradient_norm.detach().to("cpu").item()),
            )
            progress.update(
                training.examples,
                loss=f"{stats['loss']:.4f}",
                agree=f"{stats['teacher_agreement'] * 100:.1f}%",
            )
        train_summary = training.summary()
        progress.finish(
            training.examples,
            loss=f"{train_summary['loss']:.4f}",
            agree=f"{train_summary['teacher_agreement'] * 100:.1f}%",
        )
        if train_summary["examples"] == 0:
            raise ValueError("distillation training split contains no examples")
        validation = evaluate_structured_distillation(
            model,
            cache_dir,
            config=config,
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            temperature=temperature,
            soft_policy_weight=soft_policy_weight,
            hard_policy_weight=hard_policy_weight,
            value_loss_weight=value_loss_weight,
            device=resolved_device,
        ) if validation_fraction > 0 else {}
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train": train_summary,
            "validation": validation,
        })
        snapshot_metrics = _training_metadata(
            manifest,
            cache_dir=cache_dir,
            initial_checkpoint=initial_checkpoint,
            replay_manifest=replay_manifest,
            replay_cache_dir=replay_cache_dir,
            replay_ratio=normalized_replay_ratio,
            replay_examples_per_epoch=replay_examples_per_epoch,
            trainable_scope=trainable_scope,
            device=resolved_device,
            epochs=max(0, int(epochs)),
            seconds=time.perf_counter() - started,
            epoch_metrics=epoch_metrics,
            model=model,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            temperature=temperature,
            soft_policy_weight=soft_policy_weight,
            hard_policy_weight=hard_policy_weight,
            value_loss_weight=value_loss_weight,
            max_grad_norm=max_grad_norm,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        snapshot = StructuredPolicy(
            model,
            config=config,
            ruleset_fingerprint=manifest["ruleset_fingerprint"],
            device=resolved_device,
            seed=seed,
            name=f"{Path(output_checkpoint).stem}-epoch-{epoch + 1:02d}",
        )
        snapshot_path = Path(output_checkpoint).with_name(
            f"{Path(output_checkpoint).stem}.epoch-{epoch + 1:02d}{Path(output_checkpoint).suffix}"
        )
        snapshot.save(snapshot_path, metadata=snapshot_metrics)

    elapsed = time.perf_counter() - started
    policy = StructuredPolicy(
        model,
        config=config,
        ruleset_fingerprint=manifest["ruleset_fingerprint"],
        device=resolved_device,
        seed=seed,
        name=Path(output_checkpoint).stem,
    )
    metrics = _training_metadata(
        manifest,
        cache_dir=cache_dir,
        initial_checkpoint=initial_checkpoint,
        replay_manifest=replay_manifest,
        replay_cache_dir=replay_cache_dir,
        replay_ratio=normalized_replay_ratio,
        replay_examples_per_epoch=replay_examples_per_epoch,
        trainable_scope=trainable_scope,
        device=resolved_device,
        epochs=max(0, int(epochs)),
        seconds=elapsed,
        epoch_metrics=epoch_metrics,
        model=model,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        soft_policy_weight=soft_policy_weight,
        hard_policy_weight=hard_policy_weight,
        value_loss_weight=value_loss_weight,
        max_grad_norm=max_grad_norm,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    policy.save(output_checkpoint, metadata=metrics)
    return policy, metrics


def _training_metadata(
    manifest: dict[str, Any],
    *,
    cache_dir: str | Path,
    initial_checkpoint: str | Path | None,
    replay_manifest: dict[str, Any] | None,
    replay_cache_dir: str | Path | None,
    replay_ratio: float,
    replay_examples_per_epoch: int,
    trainable_scope: str,
    device: str,
    epochs: int,
    seconds: float,
    epoch_metrics: list[dict[str, Any]],
    model: StructuredPolicyNetwork,
    batch_size: int,
    shuffle_buffer: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    soft_policy_weight: float,
    hard_policy_weight: float,
    value_loss_weight: float,
    max_grad_norm: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "objective": "v8_teacher_structured_policy_distillation",
        "device": device,
        "epochs_requested": int(epochs),
        "epochs_completed": len(epoch_metrics),
        "seconds": round(float(seconds), 3),
        "cache_dir": str(Path(cache_dir).resolve()),
        "cache_fingerprint": manifest["cache_fingerprint"],
        "teacher_fingerprint": manifest["teacher_fingerprint"],
        "teacher_name": manifest.get("teacher_name"),
        "examples": int((manifest.get("counters") or {}).get("examples", 0)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "initial_checkpoint": (
            str(Path(initial_checkpoint).resolve()) if initial_checkpoint else None
        ),
        "replay_cache": (
            {
                "cache_dir": str(Path(replay_cache_dir).resolve()),
                "cache_fingerprint": replay_manifest["cache_fingerprint"],
                "teacher_fingerprint": replay_manifest["teacher_fingerprint"],
                "examples_per_epoch": int(replay_examples_per_epoch),
            }
            if replay_manifest is not None and replay_cache_dir is not None
            else None
        ),
        "epoch_metrics": epoch_metrics,
        "hyperparameters": {
            "batch_size": int(batch_size),
            "shuffle_buffer": int(shuffle_buffer),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "temperature": float(temperature),
            "soft_policy_weight": float(soft_policy_weight),
            "hard_policy_weight": float(hard_policy_weight),
            "value_loss_weight": float(value_loss_weight),
            "max_grad_norm": float(max_grad_norm),
            "validation_fraction": float(validation_fraction),
            "replay_ratio": float(replay_ratio),
            "trainable_scope": str(trainable_scope),
            "seed": int(seed),
        },
    }


def _configure_trainable_parameters(
    model: StructuredPolicyNetwork,
    scope: str,
) -> list[Any]:
    normalized = str(scope or "all").strip().lower()
    allowed = {"all", "policy-heads", "combat-policy-head"}
    if normalized not in allowed:
        raise ValueError(f"unsupported trainable scope: {scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(normalized == "all")
    if normalized == "policy-heads":
        modules = (model.pregame_policy_head, model.combat_policy_head)
    elif normalized == "combat-policy-head":
        modules = (model.combat_policy_head,)
    else:
        modules = ()
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("trainable scope selected no parameters")
    return parameters


def _interleave_example_streams(
    primary_examples,
    replay_examples,
    *,
    primary_count: int,
    replay_count: int,
    rng: random.Random,
):
    schedule = [0] * max(0, int(primary_count)) + [1] * max(0, int(replay_count))
    rng.shuffle(schedule)
    sources = (iter(primary_examples), iter(replay_examples))
    for source_index in schedule:
        try:
            yield next(sources[source_index])
        except StopIteration as exc:
            source_name = "primary" if source_index == 0 else "replay"
            raise ValueError(f"{source_name} cache yielded fewer examples than expected") from exc


def collate_distillation_examples(
    examples: Sequence[StructuredDistillationExample],
    *,
    config: StructuredModelConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    items = list(examples)
    if not items:
        raise ValueError("cannot collate an empty distillation batch")
    batch = collate_structured_decisions(
        [item.decision for item in items], config=config, device=device
    )
    action_count = batch["action_mask"].shape[1]
    tensor_device = torch.device(device)
    teacher_logits = torch.full(
        (len(items), action_count), -1e9, dtype=torch.float32, device=tensor_device
    )
    for row, item in enumerate(items):
        if len(item.teacher_logits) != item.decision.action_count:
            raise ValueError("teacher logit count does not match legal action count")
        teacher_logits[row, :len(item.teacher_logits)] = torch.tensor(
            item.teacher_logits, dtype=torch.float32, device=tensor_device
        )
    batch["teacher_logits"] = teacher_logits
    batch["teacher_values"] = torch.tensor(
        [item.teacher_value for item in items],
        dtype=torch.float32,
        device=tensor_device,
    )
    return batch


def evaluate_structured_distillation(
    model: StructuredPolicyNetwork,
    cache_dir: str | Path,
    *,
    config: StructuredModelConfig,
    batch_size: int,
    validation_fraction: float,
    temperature: float,
    soft_policy_weight: float,
    hard_policy_weight: float,
    value_loss_weight: float,
    device: str,
) -> dict[str, Any]:
    model.eval()
    accumulator = _DistillationAccumulator()
    examples = iter_cached_examples(
        cache_dir,
        split="validation",
        validation_fraction=validation_fraction,
    )
    batches = iter_cached_batches(
        examples,
        batch_size=batch_size,
        shuffle_buffer=batch_size,
        rng=random.Random(0),
    )
    with torch.inference_mode():
        for examples_batch in batches:
            batch = collate_distillation_examples(
                examples_batch, config=config, device=device
            )
            scores, values = model(batch)
            _, stats = _distillation_loss(
                scores,
                values,
                batch,
                temperature=temperature,
                soft_policy_weight=soft_policy_weight,
                hard_policy_weight=hard_policy_weight,
                value_loss_weight=value_loss_weight,
            )
            accumulator.add(len(examples_batch), stats, 0.0)
    return accumulator.summary()


def _distillation_loss(
    scores,
    values,
    batch,
    *,
    temperature: float,
    soft_policy_weight: float,
    hard_policy_weight: float,
    value_loss_weight: float,
):
    scale = max(1e-6, float(temperature))
    mask = batch["action_mask"]
    student_logits = scores + batch["action_logit_biases"]
    teacher_logits = batch["teacher_logits"]
    student_scaled = (student_logits / scale).masked_fill(~mask, -1e9)
    teacher_scaled = (teacher_logits / scale).masked_fill(~mask, -1e9)
    student_log_probs = torch.nn.functional.log_softmax(student_scaled, dim=1)
    teacher_log_probs = torch.nn.functional.log_softmax(teacher_scaled, dim=1)
    teacher_probs = teacher_log_probs.exp()
    soft_loss = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).masked_fill(~mask, 0.0).sum(dim=1).mean() * (scale * scale)
    teacher_targets = teacher_logits.masked_fill(~mask, -1e9).argmax(dim=1)
    hard_loss = torch.nn.functional.cross_entropy(
        student_logits.masked_fill(~mask, -1e9), teacher_targets
    )
    value_loss = torch.nn.functional.mse_loss(values, batch["teacher_values"])
    total = (
        float(soft_policy_weight) * soft_loss
        + float(hard_policy_weight) * hard_loss
        + float(value_loss_weight) * value_loss
    )
    student_targets = student_logits.masked_fill(~mask, -1e9).argmax(dim=1)
    agreement = (student_targets == teacher_targets).float().mean()
    teacher_entropy = -(
        teacher_probs * teacher_log_probs
    ).masked_fill(~mask, 0.0).sum(dim=1).mean()
    value_rmse = value_loss.sqrt()
    return total, {
        "loss": float(total.detach().to("cpu").item()),
        "soft_policy_loss": float(soft_loss.detach().to("cpu").item()),
        "hard_policy_loss": float(hard_loss.detach().to("cpu").item()),
        "value_loss": float(value_loss.detach().to("cpu").item()),
        "value_rmse": float(value_rmse.detach().to("cpu").item()),
        "teacher_agreement": float(agreement.detach().to("cpu").item()),
        "teacher_entropy": float(teacher_entropy.detach().to("cpu").item()),
    }


class _DistillationAccumulator:
    def __init__(self) -> None:
        self.examples = 0
        self.batches = 0
        self.totals = {
            "loss": 0.0,
            "soft_policy_loss": 0.0,
            "hard_policy_loss": 0.0,
            "value_loss": 0.0,
            "value_rmse": 0.0,
            "teacher_agreement": 0.0,
            "teacher_entropy": 0.0,
            "gradient_norm": 0.0,
        }

    def add(self, count: int, stats: dict[str, float], gradient_norm: float) -> None:
        self.examples += int(count)
        self.batches += 1
        for key in self.totals:
            value = gradient_norm if key == "gradient_norm" else stats[key]
            self.totals[key] += float(value) * int(count)

    def summary(self) -> dict[str, Any]:
        divisor = max(1, self.examples)
        return {
            "examples": self.examples,
            "batches": self.batches,
            **{
                key: round(value / divisor, 7)
                for key, value in self.totals.items()
            },
        }
