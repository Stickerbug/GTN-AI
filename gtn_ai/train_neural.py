from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .neural_model import NeuralModelConfig, require_torch
from .neural_training import train_neural_behavior_policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the GTN variable-action policy/value network",
    )
    parser.add_argument("input", nargs="+", help="Self-play JSONL or JSONL.gz files")
    parser.add_argument("--output", default="models/variable-action-v1.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--device", choices=("auto", "cpu", "xpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-checkpoint", help="Continue training from a compatible checkpoint")
    parser.add_argument("--winner-policy-weight", type=float, default=1.0)
    parser.add_argument("--loser-policy-weight", type=float, default=1.0)
    parser.add_argument("--draw-policy-weight", type=float, default=1.0)
    parser.add_argument(
        "--include-recovered-episodes",
        action="store_true",
        help="Train on episodes that needed loop recovery (excluded by default)",
    )
    parser.add_argument("--observation-buckets", type=int, default=1 << 15)
    parser.add_argument("--action-buckets", type=int, default=1 << 14)
    parser.add_argument("--history-buckets", type=int, default=1 << 13)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--history-events", type=int, default=32)
    args = parser.parse_args(argv)
    require_torch()
    config = None if args.init_checkpoint else NeuralModelConfig(
        observation_buckets=args.observation_buckets,
        action_buckets=args.action_buckets,
        history_buckets=args.history_buckets,
        hidden_dim=args.hidden_dim,
        max_history_events=args.history_events,
    )
    policy, metrics = train_neural_behavior_policy(
        args.input,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        validation_fraction=args.validation_fraction,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        seed=args.seed,
        initial_checkpoint=args.init_checkpoint,
        winner_policy_weight=args.winner_policy_weight,
        loser_policy_weight=args.loser_policy_weight,
        draw_policy_weight=args.draw_policy_weight,
        skip_recovered_episodes=not args.include_recovered_episodes,
    )
    policy.name = Path(args.output).stem
    policy.save(args.output, metadata=metrics)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        **metrics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
