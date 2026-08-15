from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .correction_model import CorrectionModelConfig
from .correction_training import train_structured_correction
from .neural_model import require_torch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a gated residual ranker over a frozen structured policy",
    )
    parser.add_argument("cache", help="Recorded search-teacher cache directory")
    parser.add_argument("--base", required=True, help="Frozen structured checkpoint")
    parser.add_argument(
        "--context-checkpoint",
        help="Frozen contextual encoder checkpoint used only by the correction head",
    )
    parser.add_argument(
        "--init-correction",
        help="Continue from a compatible correction checkpoint",
    )
    parser.add_argument("--output", default="models/structured-correction-v1.pt")
    parser.add_argument(
        "--validation-cache",
        help="Independent whole-session validation cache; disables per-example splitting",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--contextual-value-features",
        action="store_true",
        help="Use paired contextual card-value features without changing base logits",
    )
    parser.add_argument(
        "--include-pregame-corrections",
        action="store_true",
        help="Allow the correction head to alter pregame selection decisions",
    )
    parser.add_argument("--rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--gate-loss-weight", type=float, default=0.4)
    parser.add_argument("--pair-loss-weight", type=float, default=0.25)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.1)
    parser.add_argument("--residual-loss-weight", type=float, default=1e-4)
    parser.add_argument(
        "--minimum-correction-margin",
        type=float,
        default=0.05,
        help="Treat lower-margin teacher disagreements as keep-base examples",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "xpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    require_torch()
    config = CorrectionModelConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        top_k=args.top_k,
        residual_scale=args.residual_scale,
        gate_threshold=args.gate_threshold,
        combat_only=not args.include_pregame_corrections,
        contextual_value_features=args.contextual_value_features,
    )
    _, metrics = train_structured_correction(
        args.cache,
        base_checkpoint=args.base,
        output_checkpoint=args.output,
        validation_cache_dir=args.validation_cache,
        context_checkpoint=args.context_checkpoint,
        init_correction_checkpoint=args.init_correction,
        correction_config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_loss_weight=args.rank_loss_weight,
        gate_loss_weight=args.gate_loss_weight,
        pair_loss_weight=args.pair_loss_weight,
        anchor_loss_weight=args.anchor_loss_weight,
        residual_loss_weight=args.residual_loss_weight,
        minimum_correction_margin=args.minimum_correction_margin,
        max_grad_norm=args.max_grad_norm,
        validation_fraction=args.validation_fraction,
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
