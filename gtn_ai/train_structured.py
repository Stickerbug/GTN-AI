from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .neural_model import require_torch
from .structured_distillation import train_structured_distillation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Distill the v8 teacher into the structured GTN policy",
    )
    parser.add_argument("cache", help="Versioned cache directory")
    parser.add_argument("--output", default="models/structured-v2.pt")
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--model-state-layers",
        type=int,
        help="Override state Transformer depth; an initial checkpoint is expanded safely",
    )
    parser.add_argument(
        "--model-action-layers",
        type=int,
        help="Override action Transformer depth; an initial checkpoint is expanded safely",
    )
    parser.add_argument("--replay-cache", help="Optional broad cache mixed in to prevent forgetting")
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.0,
        help="Replay examples per primary cache example in each epoch",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "policy-heads", "combat-policy-head"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--soft-policy-weight", type=float, default=1.0)
    parser.add_argument("--hard-policy-weight", type=float, default=0.1)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--teacher-margin-weight-power",
        type=float,
        default=0.0,
        help="Upweight decisive teacher labels by their top-two logit margin; 0 disables",
    )
    parser.add_argument("--teacher-margin-weight-floor", type=float, default=1.0)
    parser.add_argument("--teacher-margin-weight-reference", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "xpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    require_torch()
    _, metrics = train_structured_distillation(
        args.cache,
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        soft_policy_weight=args.soft_policy_weight,
        hard_policy_weight=args.hard_policy_weight,
        value_loss_weight=args.value_loss_weight,
        teacher_margin_weight_power=args.teacher_margin_weight_power,
        teacher_margin_weight_floor=args.teacher_margin_weight_floor,
        teacher_margin_weight_reference=args.teacher_margin_weight_reference,
        max_grad_norm=args.max_grad_norm,
        validation_fraction=args.validation_fraction,
        initial_checkpoint=args.init_checkpoint,
        model_state_layers=args.model_state_layers,
        model_action_layers=args.model_action_layers,
        replay_cache_dir=args.replay_cache,
        replay_ratio=args.replay_ratio,
        trainable_scope=args.trainable_scope,
        device=args.device,
        seed=args.seed,
        progress_interval=args.progress_interval,
        show_progress=not args.quiet,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        **metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
