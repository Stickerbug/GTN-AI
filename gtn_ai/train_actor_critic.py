from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .actor_critic_training import train_actor_critic_policy
from .neural_model import require_torch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Improve a GTN policy directly from on-policy game outcomes",
    )
    parser.add_argument("input", nargs="+", help="On-policy self-play JSONL or JSONL.gz files")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output", default="models/actor-critic-v1.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.15)
    parser.add_argument("--value-clip", type=float, default=0.15)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--entropy-weight", type=float, default=0.004)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=1.0,
        help="GAE trace decay; 1.0 reproduces terminal Monte Carlo advantages",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "xpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--include-recovered-episodes",
        action="store_true",
        help="Include episodes that needed loop recovery (not valid on-policy by default)",
    )
    args = parser.parse_args(argv)
    require_torch()
    policy, metrics = train_actor_critic_policy(
        args.input,
        initial_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_ratio=args.clip_ratio,
        value_clip=args.value_clip,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=args.device,
        seed=args.seed,
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
