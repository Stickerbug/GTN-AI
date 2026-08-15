from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .neural_model import require_torch
from .structured_cache import build_distillation_cache
from .structured_model import StructuredModelConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Encode GTN trajectories and cache v8 teacher predictions",
    )
    parser.add_argument("input", nargs="+", help="Trajectory JSONL or JSONL.gz files")
    parser.add_argument("--teacher", default="models/champion.pt")
    parser.add_argument(
        "--deck-prior",
        help="Optional anonymous population deck prior added to structured inputs",
    )
    parser.add_argument(
        "--dynamic-deck-belief",
        action="store_true",
        help="Condition deck-prior tokens on public opponent card evidence",
    )
    parser.add_argument("--output", default="datasets/structured-v2-cache")
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument(
        "--expected-decisions",
        type=int,
        default=0,
        help="Optional progress total only; it does not limit cache generation",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "xpu", "cuda", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include-recovered-episodes", action="store_true")
    _add_model_arguments(parser)
    args = parser.parse_args(argv)
    require_torch()
    config = _config_from_args(args)
    manifest = build_distillation_cache(
        args.input,
        teacher_checkpoint=args.teacher,
        output_dir=args.output,
        config=config,
        deck_prior_path=args.deck_prior,
        dynamic_deck_belief=args.dynamic_deck_belief,
        teacher_batch_size=args.teacher_batch_size,
        shard_size=args.shard_size,
        device=args.device,
        max_decisions=args.max_decisions,
        expected_decisions=args.expected_decisions,
        skip_recovered_episodes=not args.include_recovered_episodes,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
        show_progress=not args.quiet,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        **manifest,
    }, ensure_ascii=False, indent=2))
    return 0


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = StructuredModelConfig()
    parser.add_argument("--categorical-buckets", type=int, default=defaults.categorical_buckets)
    parser.add_argument("--categorical-slots", type=int, default=defaults.categorical_slots)
    parser.add_argument("--numeric-buckets", type=int, default=defaults.numeric_buckets)
    parser.add_argument("--max-state-tokens", type=int, default=defaults.max_state_tokens)
    parser.add_argument("--max-history-events", type=int, default=defaults.max_history_events)
    parser.add_argument("--model-dim", type=int, default=defaults.model_dim)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--state-layers", type=int, default=defaults.state_layers)
    parser.add_argument("--action-layers", type=int, default=defaults.action_layers)
    parser.add_argument("--feedforward-dim", type=int, default=defaults.feedforward_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument(
        "--contextual-value-features",
        action="store_true",
        default=defaults.contextual_value_features,
        help=(
            "Encode card-instance mutations, rule operations, resource demand, "
            "and deck context without a fixed scalar card value"
        ),
    )


def _config_from_args(args) -> StructuredModelConfig:
    return StructuredModelConfig(
        categorical_buckets=args.categorical_buckets,
        categorical_slots=args.categorical_slots,
        numeric_buckets=args.numeric_buckets,
        max_state_tokens=args.max_state_tokens,
        max_history_events=args.max_history_events,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        state_layers=args.state_layers,
        action_layers=args.action_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        contextual_value_features=args.contextual_value_features,
    )


if __name__ == "__main__":
    raise SystemExit(main())
