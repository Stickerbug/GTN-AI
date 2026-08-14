from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .neural_model import require_torch
from .structured_cache import build_recorded_teacher_cache
from .structured_model import load_structured_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Encode recorded offline-search labels into structured cache shards",
    )
    parser.add_argument("input", nargs="+", help="Trajectory JSONL or JSONL.gz files")
    parser.add_argument(
        "--config-checkpoint",
        default="models/structured-v2-distilled.pt",
        help="Structured checkpoint whose feature/model configuration should be reused",
    )
    parser.add_argument("--output", default="datasets/structured-search-teacher")
    parser.add_argument(
        "--deck-prior",
        help="Optional anonymous population deck prior added to structured inputs",
    )
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--expected-decisions", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include-recovered-episodes", action="store_true")
    parser.add_argument(
        "--min-teacher-margin",
        type=float,
        default=0.0,
        help="Keep only labels whose raw search score margin meets this threshold",
    )
    args = parser.parse_args(argv)
    require_torch()
    checkpoint = load_structured_checkpoint(args.config_checkpoint, device="cpu")
    manifest = build_recorded_teacher_cache(
        args.input,
        output_dir=args.output,
        config=checkpoint["config"],
        deck_prior_path=args.deck_prior,
        shard_size=args.shard_size,
        max_decisions=args.max_decisions,
        expected_decisions=args.expected_decisions,
        skip_recovered_episodes=not args.include_recovered_episodes,
        min_teacher_margin=args.min_teacher_margin,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
        show_progress=not args.quiet,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        **manifest,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
