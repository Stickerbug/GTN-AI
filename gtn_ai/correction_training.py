from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from .correction_model import (
    CorrectionModelConfig,
    StructuredCorrectionNetwork,
    StructuredCorrectionPolicy,
)
from .neural_model import (
    neural_state_dict_fingerprint,
    require_torch,
    resolve_device,
    torch,
)
from .progress import ProgressReporter
from .structured_cache import (
    StructuredDistillationExample,
    cached_split_count,
    iter_cached_batches,
    iter_cached_examples,
    load_cache_manifest,
)
from .structured_distillation import collate_distillation_examples
from .structured_model import load_structured_checkpoint


def train_structured_correction(
    cache_dir: str | Path,
    *,
    base_checkpoint: str | Path,
    output_checkpoint: str | Path,
    correction_config: CorrectionModelConfig | None = None,
    epochs: int = 6,
    batch_size: int = 128,
    shuffle_buffer: int = 2048,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    rank_loss_weight: float = 1.0,
    gate_loss_weight: float = 0.4,
    pair_loss_weight: float = 0.25,
    anchor_loss_weight: float = 0.1,
    residual_loss_weight: float = 1e-4,
    max_grad_norm: float = 1.0,
    validation_fraction: float = 0.1,
    device: str = "auto",
    seed: int = 1,
    progress_interval: float = 10.0,
    show_progress: bool = True,
) -> tuple[StructuredCorrectionPolicy, dict[str, Any]]:
    require_torch()
    started = time.perf_counter()
    manifest = load_cache_manifest(cache_dir)
    resolved_device = resolve_device(device)
    base = load_structured_checkpoint(base_checkpoint, device=resolved_device)
    config = correction_config or CorrectionModelConfig()
    base_config = base["config"]
    cache_config = type(base_config).from_dict(manifest.get("structured_config") or {})
    if cache_config != base_config:
        raise ValueError("correction cache config does not match the base checkpoint")
    if str(manifest.get("ruleset_fingerprint") or "") != base["ruleset_fingerprint"]:
        raise ValueError("correction cache ruleset does not match the base checkpoint")

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    base_model = base["model"]
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    model = StructuredCorrectionNetwork(base_config, config)
    model.to(torch.device(resolved_device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_count = cached_split_count(
        manifest,
        split="train",
        validation_fraction=validation_fraction,
    )
    epoch_metrics: list[dict[str, Any]] = []
    best_validation = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch = 0

    for epoch in range(max(0, int(epochs))):
        model.train()
        accumulator = _CorrectionAccumulator()
        progress = ProgressReporter(
            f"correction {epoch + 1}/{max(1, int(epochs))}",
            total=train_count or None,
            interval=progress_interval,
            enabled=show_progress,
        )
        examples = iter_cached_examples(
            cache_dir,
            split="train",
            validation_fraction=validation_fraction,
            shuffle_shards=True,
            seed=int(seed) + epoch,
        )
        batches = iter_cached_batches(
            examples,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            rng=random.Random(int(seed) + epoch * 104729),
        )
        for examples_batch in batches:
            batch = collate_distillation_examples(
                examples_batch,
                config=base_config,
                device=resolved_device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                joint, state_summary = base_model.encode_features(batch)
                base_scores, _ = base_model.score_features(
                    joint, state_summary, batch
                )
                base_scores = base_scores + batch["action_logit_biases"]
            corrected, gate_logits, residual = model(
                joint,
                state_summary,
                base_scores,
                batch,
                hard_gate=False,
            )
            loss, stats = _correction_loss(
                base_scores,
                corrected,
                gate_logits,
                residual,
                batch,
                top_k=config.top_k,
                rank_loss_weight=rank_loss_weight,
                gate_loss_weight=gate_loss_weight,
                pair_loss_weight=pair_loss_weight,
                anchor_loss_weight=anchor_loss_weight,
                residual_loss_weight=residual_loss_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("correction training produced a non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(max_grad_norm)
            )
            optimizer.step()
            accumulator.add(
                len(examples_batch),
                stats,
                float(gradient_norm.detach().to("cpu").item()),
            )
            progress.update(
                accumulator.examples,
                loss=f"{stats['loss']:.4f}",
                base=f"{stats['base_agreement'] * 100:.1f}%",
                soft=f"{stats['soft_agreement'] * 100:.1f}%",
            )
        train_summary = accumulator.summary()
        progress.finish(
            accumulator.examples,
            loss=f"{train_summary['loss']:.4f}",
            soft=f"{train_summary['soft_agreement'] * 100:.1f}%",
        )
        if train_summary["examples"] == 0:
            raise ValueError("correction training split contains no examples")
        validation = evaluate_structured_correction(
            base_model,
            model,
            cache_dir,
            base_config=base_config,
            correction_config=config,
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            device=resolved_device,
        ) if validation_fraction > 0 else {}
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train": train_summary,
            "validation": validation,
        })
        score = float(validation.get("best_teacher_agreement", 0.0))
        if score > best_validation:
            best_validation = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().to("cpu").clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    final_validation = evaluate_structured_correction(
        base_model,
        model,
        cache_dir,
        base_config=base_config,
        correction_config=config,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        device=resolved_device,
    ) if validation_fraction > 0 else {}
    tuned_threshold = float(
        final_validation.get("best_gate_threshold", config.gate_threshold)
    )
    tuned_config = replace(config, gate_threshold=tuned_threshold)
    elapsed = time.perf_counter() - started
    metadata = {
        "objective": "frozen_base_preference_correction",
        "device": resolved_device,
        "seconds": round(elapsed, 3),
        "epochs_requested": int(epochs),
        "epochs_completed": len(epoch_metrics),
        "best_epoch": best_epoch,
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_fingerprint": neural_state_dict_fingerprint(base_model.state_dict()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "cache_fingerprint": manifest["cache_fingerprint"],
        "teacher_fingerprint": manifest["teacher_fingerprint"],
        "teacher_name": manifest.get("teacher_name"),
        "examples": int((manifest.get("counters") or {}).get("examples", 0)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "correction_config": asdict(tuned_config),
        "epoch_metrics": epoch_metrics,
        "final_validation": final_validation,
        "hyperparameters": {
            "batch_size": int(batch_size),
            "shuffle_buffer": int(shuffle_buffer),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "rank_loss_weight": float(rank_loss_weight),
            "gate_loss_weight": float(gate_loss_weight),
            "pair_loss_weight": float(pair_loss_weight),
            "anchor_loss_weight": float(anchor_loss_weight),
            "residual_loss_weight": float(residual_loss_weight),
            "max_grad_norm": float(max_grad_norm),
            "validation_fraction": float(validation_fraction),
            "seed": int(seed),
        },
    }
    policy = StructuredCorrectionPolicy(
        base_model,
        model,
        base_config=base_config,
        correction_config=tuned_config,
        ruleset_fingerprint=base["ruleset_fingerprint"],
        device=resolved_device,
        seed=seed,
        name=Path(output_checkpoint).stem,
    )
    policy.save(output_checkpoint, metadata=metadata)
    return policy, metadata


def evaluate_structured_correction(
    base_model,
    model,
    cache_dir: str | Path,
    *,
    base_config,
    correction_config: CorrectionModelConfig,
    batch_size: int,
    validation_fraction: float,
    device: str,
) -> dict[str, Any]:
    model.eval()
    records: list[tuple[float, int, int, int, bool]] = []
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
                examples_batch, config=base_config, device=device
            )
            joint, state_summary = base_model.encode_features(batch)
            base_scores, _ = base_model.score_features(joint, state_summary, batch)
            base_scores = base_scores + batch["action_logit_biases"]
            _, gate_logits, residual = model(
                joint,
                state_summary,
                base_scores,
                batch,
                hard_gate=False,
            )
            mask = batch["action_mask"]
            teacher = batch["teacher_logits"].masked_fill(~mask, -1e9).argmax(dim=1)
            base = base_scores.masked_fill(~mask, -1e9).argmax(dim=1)
            reranked_scores = base_scores + (
                float(correction_config.residual_scale) * residual
            )
            reranked = reranked_scores.masked_fill(~mask, -1e9).argmax(dim=1)
            candidate_count = min(int(correction_config.top_k), mask.shape[1])
            candidates = base_scores.masked_fill(~mask, -1e9).topk(
                candidate_count, dim=1
            ).indices
            eligible = (candidates == teacher.unsqueeze(1)).any(dim=1)
            probabilities = torch.sigmoid(gate_logits)
            for index in range(len(examples_batch)):
                records.append((
                    float(probabilities[index].to("cpu").item()),
                    int(base[index].to("cpu").item()),
                    int(reranked[index].to("cpu").item()),
                    int(teacher[index].to("cpu").item()),
                    bool(eligible[index].to("cpu").item()),
                ))
    if not records:
        return {"examples": 0}
    thresholds = [index / 100.0 for index in range(5, 100, 5)]
    candidates = []
    for threshold in thresholds:
        correct = 0
        changed = 0
        true_positive = 0
        false_positive = 0
        correction_targets = 0
        for probability, base, reranked, teacher, eligible in records:
            needs_correction = eligible and base != teacher
            use_reranker = eligible and probability >= threshold
            prediction = reranked if use_reranker else base
            correct += int(prediction == teacher)
            changed += int(prediction != base)
            correction_targets += int(needs_correction)
            true_positive += int(use_reranker and needs_correction)
            false_positive += int(use_reranker and not needs_correction)
        candidates.append({
            "threshold": threshold,
            "teacher_agreement": correct / len(records),
            "changed_fraction": changed / len(records),
            "gate_recall": true_positive / max(1, correction_targets),
            "gate_precision": true_positive / max(1, true_positive + false_positive),
        })
    best = max(
        candidates,
        key=lambda item: (
            item["teacher_agreement"],
            -item["changed_fraction"],
            item["threshold"],
        ),
    )
    base_agreement = sum(base == teacher for _, base, _, teacher, _ in records) / len(records)
    eligible_fraction = sum(eligible for *_, eligible in records) / len(records)
    return {
        "examples": len(records),
        "base_teacher_agreement": round(base_agreement, 7),
        "teacher_in_top_k": round(eligible_fraction, 7),
        "best_gate_threshold": float(best["threshold"]),
        "best_teacher_agreement": round(float(best["teacher_agreement"]), 7),
        "changed_fraction": round(float(best["changed_fraction"]), 7),
        "gate_recall": round(float(best["gate_recall"]), 7),
        "gate_precision": round(float(best["gate_precision"]), 7),
    }


def _correction_loss(
    base_scores,
    corrected_scores,
    gate_logits,
    residual,
    batch,
    *,
    top_k: int,
    rank_loss_weight: float,
    gate_loss_weight: float,
    pair_loss_weight: float,
    anchor_loss_weight: float,
    residual_loss_weight: float,
):
    mask = batch["action_mask"]
    teacher_logits = batch["teacher_logits"].masked_fill(~mask, -1e9)
    teacher_targets = teacher_logits.argmax(dim=1)
    base_masked = base_scores.masked_fill(~mask, -1e9)
    base_targets = base_masked.argmax(dim=1)
    candidate_count = min(max(1, int(top_k)), mask.shape[1])
    candidates = base_masked.topk(candidate_count, dim=1).indices
    eligible = (candidates == teacher_targets.unsqueeze(1)).any(dim=1)
    needs_correction = eligible & (teacher_targets != base_targets)

    top_teacher = teacher_logits.topk(min(2, teacher_logits.shape[1]), dim=1).values
    teacher_margin = (
        top_teacher[:, 0] - top_teacher[:, 1]
        if top_teacher.shape[1] > 1
        else torch.ones_like(top_teacher[:, 0])
    ).clamp_min(0.0)
    confidence = 0.5 + 0.5 * (1.0 - torch.exp(-teacher_margin / 0.05))
    rank_targets = torch.where(eligible, teacher_targets, base_targets)
    rank_per_item = torch.nn.functional.cross_entropy(
        corrected_scores.masked_fill(~mask, -1e9),
        rank_targets,
        reduction="none",
    )
    rank_weights = confidence * torch.where(
        needs_correction,
        torch.full_like(confidence, 2.0),
        torch.ones_like(confidence),
    )
    rank_loss = (rank_per_item * rank_weights).sum() / rank_weights.sum().clamp_min(1.0)

    gate_targets = needs_correction.to(gate_logits.dtype)
    positives = gate_targets.sum()
    negatives = gate_targets.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 4.0)
    gate_per_item = torch.nn.functional.binary_cross_entropy_with_logits(
        gate_logits,
        gate_targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    gate_loss = (gate_per_item * confidence).sum() / confidence.sum().clamp_min(1.0)

    teacher_scores = corrected_scores.gather(1, teacher_targets.unsqueeze(1)).squeeze(1)
    base_choice_scores = corrected_scores.gather(1, base_targets.unsqueeze(1)).squeeze(1)
    pair_values = torch.nn.functional.softplus(
        -(teacher_scores - base_choice_scores)
    )
    pair_weights = (confidence * needs_correction.to(confidence.dtype))
    pair_loss = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)

    base_log_probs = torch.log_softmax(base_masked, dim=1)
    corrected_log_probs = torch.log_softmax(
        corrected_scores.masked_fill(~mask, -1e9), dim=1
    )
    base_probs = base_log_probs.exp()
    anchor_values = (
        base_probs * (base_log_probs - corrected_log_probs)
    ).masked_fill(~mask, 0.0).sum(dim=1)
    anchor_weights = (~needs_correction).to(anchor_values.dtype)
    anchor_loss = (anchor_values * anchor_weights).sum() / anchor_weights.sum().clamp_min(1.0)
    residual_loss = residual.square().masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)

    total = (
        float(rank_loss_weight) * rank_loss
        + float(gate_loss_weight) * gate_loss
        + float(pair_loss_weight) * pair_loss
        + float(anchor_loss_weight) * anchor_loss
        + float(residual_loss_weight) * residual_loss
    )
    soft_targets = corrected_scores.masked_fill(~mask, -1e9).argmax(dim=1)
    gate_predictions = torch.sigmoid(gate_logits) >= 0.5
    return total, {
        "loss": float(total.detach().to("cpu").item()),
        "rank_loss": float(rank_loss.detach().to("cpu").item()),
        "gate_loss": float(gate_loss.detach().to("cpu").item()),
        "pair_loss": float(pair_loss.detach().to("cpu").item()),
        "anchor_loss": float(anchor_loss.detach().to("cpu").item()),
        "residual_loss": float(residual_loss.detach().to("cpu").item()),
        "base_agreement": float((base_targets == teacher_targets).float().mean().to("cpu").item()),
        "soft_agreement": float((soft_targets == teacher_targets).float().mean().to("cpu").item()),
        "correction_fraction": float(needs_correction.float().mean().to("cpu").item()),
        "teacher_in_top_k": float(eligible.float().mean().to("cpu").item()),
        "gate_accuracy": float((gate_predictions == needs_correction).float().mean().to("cpu").item()),
    }


class _CorrectionAccumulator:
    def __init__(self) -> None:
        self.examples = 0
        self.batches = 0
        self.totals = {
            "loss": 0.0,
            "rank_loss": 0.0,
            "gate_loss": 0.0,
            "pair_loss": 0.0,
            "anchor_loss": 0.0,
            "residual_loss": 0.0,
            "base_agreement": 0.0,
            "soft_agreement": 0.0,
            "correction_fraction": 0.0,
            "teacher_in_top_k": 0.0,
            "gate_accuracy": 0.0,
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
