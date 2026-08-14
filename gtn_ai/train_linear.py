from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .linear_model import train_behavior_cloning, train_monte_carlo
from .trajectory import TRAJECTORY_SCHEMA_VERSION


def read_episodes(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    episodes = []
    for raw_path in paths:
        path = Path(raw_path)
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    version = int(value.get("schema_version", -1))
                    if version != TRAJECTORY_SCHEMA_VERSION:
                        raise ValueError(
                            f"{path}: unsupported trajectory schema {version}; "
                            f"expected {TRAJECTORY_SCHEMA_VERSION}"
                        )
                    episodes.append(value)
    return episodes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the dependency-free GTN action-value baseline")
    parser.add_argument("input", nargs="+", help="Self-play JSONL or JSONL.gz files")
    parser.add_argument("--output", default="models/hashed-linear-v1.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--buckets", type=int, default=1 << 16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--objective",
        choices=("behavior", "monte-carlo"),
        default="behavior",
        help="Clone recorded legal-set choices, or regress selected actions to terminal outcome",
    )
    args = parser.parse_args(argv)
    episodes = read_episodes(args.input)
    trainer = train_behavior_cloning if args.objective == "behavior" else train_monte_carlo
    policy, metrics = trainer(
        episodes,
        buckets=args.buckets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    policy.save(args.output, metadata=metrics)
    print(json.dumps({"output": str(Path(args.output).resolve()), **metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
