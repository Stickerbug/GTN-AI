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
    correction_proposal_indices,
    load_correction_checkpoint,
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
from .structured_model import (
    load_structured_checkpoint,
    structured_feature_layout_compatible,
)


def train_structured_correction(
    cache_dir: str | Path,
    *,
    base_checkpoint: str | Path,
    output_checkpoint: str | Path,
    validation_cache_dir: str | Path | None = None,
    context_checkpoint: str | Path | None = None,
    init_correction_checkpoint: str | Path | None = None,
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
    minimum_correction_margin: float = 0.05,
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
    validation_manifest = (
        load_cache_manifest(validation_cache_dir)
        if validation_cache_dir is not None
        else None
    )
    resolved_device = resolve_device(device)
    base = load_structured_checkpoint(base_checkpoint, device=resolved_device)
    config = correction_config or CorrectionModelConfig()
    base_config = base["config"]
    cache_config = type(base_config).from_dict(manifest.get("structured_config") or {})
    context_model = None
    context_config = None
    if config.contextual_value_features:
        if context_checkpoint is None:
            raise ValueError(
                "contextual correction training requires --context-checkpoint"
            )
        if not cache_config.contextual_value_features:
            raise ValueError("contextual correction requires a contextual cache")
        if not bool(manifest.get("paired_base_features")):
            raise ValueError(
                "contextual correction cache must include paired base features"
            )
        if not structured_feature_layout_compatible(base_config, cache_config):
            raise ValueError("context cache tensor layout does not match the base model")
        context = load_structured_checkpoint(
            context_checkpoint, device=resolved_device
        )
        context_config = context["config"]
        if context_config != cache_config:
            raise ValueError("context checkpoint config does not match the cache")
        if context["ruleset_fingerprint"] != base["ruleset_fingerprint"]:
            raise ValueError("context checkpoint ruleset does not match the base")
        context_model = context["model"]
    elif cache_config != base_config:
        raise ValueError("correction cache config does not match the base checkpoint")
    if str(manifest.get("ruleset_fingerprint") or "") != base["ruleset_fingerprint"]:
        raise ValueError("correction cache ruleset does not match the base checkpoint")
    if validation_manifest is not None:
        validation_cache_config = type(base_config).from_dict(
            validation_manifest.get("structured_config") or {}
        )
        if validation_cache_config != cache_config:
            raise ValueError("validation cache config does not match training cache")
        if bool(validation_manifest.get("paired_base_features")) != bool(
            manifest.get("paired_base_features")
        ):
            raise ValueError("validation cache paired features do not match training cache")
        if str(validation_manifest.get("ruleset_fingerprint") or "") != base[
            "ruleset_fingerprint"
        ]:
            raise ValueError("validation cache ruleset does not match the base checkpoint")

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    base_model = base["model"]
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    if context_model is not None:
        context_model.eval()
        for parameter in context_model.parameters():
            parameter.requires_grad_(False)
    model = StructuredCorrectionNetwork(base_config, config, context_config)
    if init_correction_checkpoint is not None:
        initialized = load_correction_checkpoint(
            init_correction_checkpoint,
            device="cpu",
        )
        if initialized["base_config"] != base_config:
            raise ValueError("initial correction base config does not match")
        if initialized["ruleset_fingerprint"] != base["ruleset_fingerprint"]:
            raise ValueError("initial correction ruleset does not match")
        initialized_config = initialized["correction_config"]
        for field in (
            "hidden_dim",
            "dropout",
            "top_k",
            "residual_scale",
            "combat_only",
            "contextual_value_features",
        ):
            if getattr(initialized_config, field) != getattr(config, field):
                raise ValueError(
                    f"initial correction {field} does not match training config"
                )
        initialized_base_fingerprint = neural_state_dict_fingerprint(
            initialized["base_model"].state_dict()
        )
        if initialized_base_fingerprint != neural_state_dict_fingerprint(
            base_model.state_dict()
        ):
            raise ValueError("initial correction base weights do not match")
        if config.contextual_value_features:
            if initialized["context_config"] != context_config:
                raise ValueError("initial correction context config does not match")
            if neural_state_dict_fingerprint(
                initialized["context_model"].state_dict()
            ) != neural_state_dict_fingerprint(context_model.state_dict()):
                raise ValueError("initial correction context weights do not match")
        model.load_state_dict(
            initialized["correction_model"].state_dict(),
            strict=True,
        )
    model.to(torch.device(resolved_device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    external_validation = validation_manifest is not None
    training_validation_fraction = 0.0 if external_validation else validation_fraction
    train_count = cached_split_count(
        manifest,
        split="all" if external_validation else "train",
        validation_fraction=training_validation_fraction,
    )
    validation_source = validation_cache_dir if external_validation else cache_dir
    validation_split = "all" if external_validation else "validation"
    has_validation = external_validation or validation_fraction > 0
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
            split="all" if external_validation else "train",
            validation_fraction=training_validation_fraction,
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
            batch, context_batch = _collate_correction_examples(
                examples_batch,
                base_config=base_config,
                context_config=context_config,
                device=resolved_device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                joint, state_summary = base_model.encode_features(batch)
                base_scores, _ = base_model.score_features(
                    joint, state_summary, batch
                )
                base_scores = base_scores + batch["action_logit_biases"]
                context_joint = None
                context_state_summary = None
                if context_batch is not None:
                    context_joint, context_state_summary = (
                        context_model.encode_features(context_batch)
                    )
            correction_logits, candidate_mask, base_indices = model(
                joint,
                state_summary,
                base_scores,
                batch,
                context_joint,
                context_state_summary,
            )
            loss, stats = _correction_loss(
                base_scores,
                correction_logits,
                candidate_mask,
                base_indices,
                batch,
                top_k=config.top_k,
                rank_loss_weight=rank_loss_weight,
                gate_loss_weight=gate_loss_weight,
                pair_loss_weight=pair_loss_weight,
                anchor_loss_weight=anchor_loss_weight,
                residual_loss_weight=residual_loss_weight,
                minimum_correction_margin=minimum_correction_margin,
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
            validation_source,
            base_config=base_config,
            context_model=context_model,
            context_config=context_config,
            correction_config=config,
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            split=validation_split,
            minimum_correction_margin=minimum_correction_margin,
            device=resolved_device,
        ) if has_validation else {}
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
        validation_source,
        base_config=base_config,
        context_model=context_model,
        context_config=context_config,
        correction_config=config,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        split=validation_split,
        minimum_correction_margin=minimum_correction_margin,
        device=resolved_device,
    ) if has_validation else {}
    tuned_threshold = float(
        final_validation.get("best_gate_threshold", config.gate_threshold)
    )
    tuned_config = replace(config, gate_threshold=tuned_threshold)
    elapsed = time.perf_counter() - started
    metadata = {
        "objective": (
            "frozen_base_contextual_topk_correction_v1"
            if config.contextual_value_features
            else "frozen_base_topk_correction_v2"
        ),
        "device": resolved_device,
        "seconds": round(elapsed, 3),
        "epochs_requested": int(epochs),
        "epochs_completed": len(epoch_metrics),
        "best_epoch": best_epoch,
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "context_checkpoint": (
            str(Path(context_checkpoint).resolve())
            if context_checkpoint is not None
            else None
        ),
        "init_correction_checkpoint": (
            str(Path(init_correction_checkpoint).resolve())
            if init_correction_checkpoint is not None
            else None
        ),
        "base_fingerprint": neural_state_dict_fingerprint(base_model.state_dict()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "cache_fingerprint": manifest["cache_fingerprint"],
        "validation_cache_dir": (
            str(Path(validation_cache_dir).resolve())
            if validation_cache_dir is not None
            else None
        ),
        "validation_cache_fingerprint": (
            validation_manifest["cache_fingerprint"]
            if validation_manifest is not None
            else None
        ),
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
            "minimum_correction_margin": float(minimum_correction_margin),
            "max_grad_norm": float(max_grad_norm),
            "validation_fraction": float(validation_fraction),
            "external_validation_cache": external_validation,
            "seed": int(seed),
        },
    }
    policy = StructuredCorrectionPolicy(
        base_model,
        model,
        base_config=base_config,
        correction_config=tuned_config,
        context_model=context_model,
        context_config=context_config,
        ruleset_fingerprint=base["ruleset_fingerprint"],
        device=resolved_device,
        seed=seed,
        name=Path(output_checkpoint).stem,
    )
    policy.save(output_checkpoint, metadata=metadata)
    return policy, metadata


def _collate_correction_examples(
    examples: Sequence[StructuredDistillationExample],
    *,
    base_config,
    context_config,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if context_config is None:
        return (
            collate_distillation_examples(
                examples, config=base_config, device=device
            ),
            None,
        )
    base_examples = []
    for example in examples:
        if example.base_decision is None:
            raise ValueError(
                "contextual correction example is missing paired base features"
            )
        base_examples.append(replace(
            example,
            decision=example.base_decision,
            base_decision=None,
        ))
    base_batch = collate_distillation_examples(
        base_examples, config=base_config, device=device
    )
    context_batch = collate_distillation_examples(
        examples, config=context_config, device=device
    )
    if not torch.equal(base_batch["action_mask"], context_batch["action_mask"]):
        raise ValueError("paired correction action masks do not align")
    return base_batch, context_batch


def evaluate_structured_correction(
    base_model,
    model,
    cache_dir: str | Path,
    *,
    base_config,
    context_model,
    context_config,
    correction_config: CorrectionModelConfig,
    batch_size: int,
    validation_fraction: float,
    minimum_correction_margin: float,
    device: str,
    split: str = "validation",
) -> dict[str, Any]:
    model.eval()
    records: list[tuple[float, int, int, int, bool, float, float]] = []
    examples = iter_cached_examples(
        cache_dir,
        split=split,
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
            batch, context_batch = _collate_correction_examples(
                examples_batch,
                base_config=base_config,
                context_config=context_config,
                device=device,
            )
            joint, state_summary = base_model.encode_features(batch)
            base_scores, _ = base_model.score_features(joint, state_summary, batch)
            base_scores = base_scores + batch["action_logit_biases"]
            context_joint = None
            context_state_summary = None
            if context_batch is not None:
                context_joint, context_state_summary = context_model.encode_features(
                    context_batch
                )
            correction_logits, candidate_mask, base_indices = model(
                joint,
                state_summary,
                base_scores,
                batch,
                context_joint,
                context_state_summary,
            )
            mask = batch["action_mask"]
            teacher = batch["teacher_logits"].masked_fill(~mask, -1e9).argmax(dim=1)
            teacher_top = batch["teacher_logits"].masked_fill(
                ~mask, -1e9
            ).topk(min(2, mask.shape[1]), dim=1).values
            teacher_margins = (
                teacher_top[:, 0] - teacher_top[:, 1]
                if teacher_top.shape[1] > 1
                else torch.full_like(teacher_top[:, 0], float("inf"))
            ).clamp_min(0.0)
            base = base_indices
            proposal = correction_proposal_indices(
                correction_logits, base_indices
            )
            probabilities = torch.softmax(correction_logits, dim=1)
            proposal_probabilities = probabilities.gather(
                1, proposal.unsqueeze(1)
            ).squeeze(1)
            eligible = candidate_mask.gather(
                1, teacher.unsqueeze(1)
            ).squeeze(1)
            example_weights = batch.get(
                "example_weights",
                torch.ones_like(teacher_margins),
            )
            for index in range(len(examples_batch)):
                records.append((
                    float(proposal_probabilities[index].to("cpu").item()),
                    int(base[index].to("cpu").item()),
                    int(proposal[index].to("cpu").item()),
                    int(teacher[index].to("cpu").item()),
                    bool(eligible[index].to("cpu").item()),
                    float(teacher_margins[index].to("cpu").item()),
                    float(example_weights[index].to("cpu").item()),
                ))
    if not records:
        return {"examples": 0}
    thresholds = [index / 100.0 for index in range(20, 100, 5)]
    candidates = []
    for threshold in thresholds:
        correct = 0.0
        changed = 0.0
        true_positive = 0.0
        false_positive = 0.0
        correction_targets = 0.0
        total_weight = sum(record[6] for record in records)
        for probability, base, reranked, teacher, eligible, margin, weight in records:
            needs_correction = (
                eligible
                and base != teacher
                and margin >= float(minimum_correction_margin)
            )
            target = teacher if needs_correction else base
            use_correction = reranked != base and probability >= threshold
            prediction = reranked if use_correction else base
            correct += weight * int(prediction == target)
            changed += weight * int(prediction != base)
            correction_targets += weight * int(needs_correction)
            true_positive += weight * int(use_correction and prediction == target)
            false_positive += weight * int(use_correction and prediction != target)
        candidates.append({
            "threshold": threshold,
            "teacher_agreement": correct / max(1e-6, total_weight),
            "changed_fraction": changed / max(1e-6, total_weight),
            "gate_recall": true_positive / max(1e-6, correction_targets),
            "gate_precision": true_positive / max(
                1e-6, true_positive + false_positive
            ),
        })
    best = max(
        candidates,
        key=lambda item: (
            item["teacher_agreement"],
            -item["changed_fraction"],
            item["threshold"],
        ),
    )
    total_weight = sum(record[6] for record in records)
    raw_base_agreement = sum(
        weight * int(base == teacher)
        for _, base, _, teacher, _, _, weight in records
    ) / max(1e-6, total_weight)
    target_base_agreement = sum(
        weight * int(not (
            eligible
            and base != teacher
            and margin >= float(minimum_correction_margin)
        ))
        for _, base, _, teacher, eligible, margin, weight in records
    ) / max(1e-6, total_weight)
    proposal_agreement = sum(
        weight * int(proposal == teacher)
        for _, _, proposal, teacher, _, _, weight in records
    ) / max(1e-6, total_weight)
    eligible_fraction = sum(
        record[6] * int(record[4]) for record in records
    ) / max(1e-6, total_weight)
    return {
        "examples": len(records),
        "base_teacher_agreement": round(raw_base_agreement, 7),
        "base_target_agreement": round(target_base_agreement, 7),
        "proposal_teacher_agreement": round(proposal_agreement, 7),
        "teacher_in_top_k": round(eligible_fraction, 7),
        "best_gate_threshold": float(best["threshold"]),
        "best_teacher_agreement": round(float(best["teacher_agreement"]), 7),
        "best_target_agreement": round(float(best["teacher_agreement"]), 7),
        "changed_fraction": round(float(best["changed_fraction"]), 7),
        "gate_recall": round(float(best["gate_recall"]), 7),
        "gate_precision": round(float(best["gate_precision"]), 7),
        "threshold_curve": [
            {
                key: round(float(value), 7)
                for key, value in item.items()
            }
            for item in candidates
        ],
    }


def _correction_loss(
    base_scores,
    correction_logits,
    candidate_mask,
    base_indices,
    batch,
    *,
    top_k: int,
    rank_loss_weight: float,
    gate_loss_weight: float,
    pair_loss_weight: float,
    anchor_loss_weight: float,
    residual_loss_weight: float,
    minimum_correction_margin: float,
):
    mask = batch["action_mask"]
    teacher_logits = batch["teacher_logits"].masked_fill(~mask, -1e9)
    teacher_targets = teacher_logits.argmax(dim=1)
    base_targets = base_indices
    eligible = candidate_mask.gather(
        1, teacher_targets.unsqueeze(1)
    ).squeeze(1)
    top_teacher = teacher_logits.topk(min(2, teacher_logits.shape[1]), dim=1).values
    teacher_margin = (
        top_teacher[:, 0] - top_teacher[:, 1]
        if top_teacher.shape[1] > 1
        else torch.ones_like(top_teacher[:, 0])
    ).clamp_min(0.0)
    confident_teacher = teacher_margin >= float(minimum_correction_margin)
    needs_correction = (
        eligible & confident_teacher & (teacher_targets != base_targets)
    )
    confidence = 0.5 + 0.5 * (1.0 - torch.exp(-teacher_margin / 0.05))
    sample_weights = batch.get(
        "example_weights",
        torch.ones_like(confidence),
    ).to(confidence.dtype)
    # A low-margin teacher disagreement is deliberately a keep-base example.
    # Letting it supervise the ranking head would contradict the gate/anchor
    # targets and teach search noise to the residual policy.
    rank_targets = torch.where(needs_correction, teacher_targets, base_targets)
    rank_per_item = torch.nn.functional.cross_entropy(
        correction_logits,
        rank_targets,
        reduction="none",
    )
    positives = (sample_weights * needs_correction.to(sample_weights.dtype)).sum()
    negatives = (sample_weights * (~needs_correction).to(sample_weights.dtype)).sum()
    positive_weight = (negatives / positives.clamp_min(1e-6)).clamp(1.0, 4.0)
    rank_weights = confidence * sample_weights * torch.where(
        needs_correction,
        torch.ones_like(confidence) * positive_weight.to(confidence.dtype),
        torch.ones_like(confidence),
    )
    rank_loss = (rank_per_item * rank_weights).sum() / rank_weights.sum().clamp_min(1.0)

    base_mask = torch.zeros_like(candidate_mask)
    base_mask.scatter_(1, base_targets.unsqueeze(1), True)
    alternative_mask = candidate_mask & ~base_mask
    alternative_logits = correction_logits.masked_fill(~alternative_mask, -1e9)
    correction_mass_logits = torch.logsumexp(alternative_logits, dim=1)
    correction_mass_logits = torch.where(
        alternative_mask.any(dim=1),
        correction_mass_logits,
        torch.full_like(correction_mass_logits, -20.0),
    )
    gate_targets = needs_correction.to(correction_mass_logits.dtype)
    gate_per_item = torch.nn.functional.binary_cross_entropy_with_logits(
        correction_mass_logits,
        gate_targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    gate_weights = confidence * sample_weights
    gate_loss = (gate_per_item * gate_weights).sum() / gate_weights.sum().clamp_min(1.0)

    teacher_scores = correction_logits.gather(
        1, teacher_targets.unsqueeze(1)
    ).squeeze(1)
    base_choice_scores = correction_logits.gather(
        1, base_targets.unsqueeze(1)
    ).squeeze(1)
    pair_values = torch.nn.functional.softplus(
        0.25 - (teacher_scores - base_choice_scores)
    )
    pair_weights = (
        confidence * sample_weights * needs_correction.to(confidence.dtype)
    )
    pair_loss = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)

    strongest_alternative = alternative_logits.max(dim=1).values
    strongest_alternative = torch.where(
        alternative_mask.any(dim=1),
        strongest_alternative,
        torch.full_like(strongest_alternative, -20.0),
    )
    anchor_values = torch.nn.functional.softplus(strongest_alternative)
    anchor_weights = (
        sample_weights * (~needs_correction).to(anchor_values.dtype)
    )
    anchor_loss = (anchor_values * anchor_weights).sum() / anchor_weights.sum().clamp_min(1.0)
    regularized_logits = torch.where(
        alternative_mask,
        correction_logits,
        torch.zeros_like(correction_logits),
    )
    residual_per_example = regularized_logits.square().sum(dim=1) / (
        alternative_mask.sum(dim=1).clamp_min(1)
    )
    residual_loss = (
        residual_per_example * sample_weights
    ).sum() / sample_weights.sum().clamp_min(1.0)

    total = (
        float(rank_loss_weight) * rank_loss
        + float(gate_loss_weight) * gate_loss
        + float(pair_loss_weight) * pair_loss
        + float(anchor_loss_weight) * anchor_loss
        + float(residual_loss_weight) * residual_loss
    )
    soft_targets = correction_proposal_indices(correction_logits, base_targets)
    gate_predictions = soft_targets != base_targets
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
