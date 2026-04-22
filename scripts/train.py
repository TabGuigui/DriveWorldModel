"""Train a DriveWorldModel experiment from config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIFFUSERS = ROOT / "diffusers" / "src"
for path in (ROOT, LOCAL_DIFFUSERS):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import drivewm.data  # noqa: F401 - register dataset adapters
from drivewm.config import load_config
from drivewm.registry import DATASETS
from drivewm.training import DriveWorldTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="DriveWorldModel YAML config")
    parser.add_argument("--mixed-precision", default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--deepspeed-config", default=None, help="Path to a DeepSpeed json config")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--validation-prompt", default=None)
    parser.add_argument("--num-validation-videos", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    if args.mixed_precision is not None:
        config.training.mixed_precision = args.mixed_precision

    dataset_cls = DATASETS.get(config.dataset.name)

    trainer = DriveWorldTrainer(
        config=config,
        args=args,
        dataset_cls=dataset_cls,
    )
    trainer.train()


if __name__ == "__main__":
    main()
