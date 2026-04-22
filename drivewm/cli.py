"""Command-line interface for DriveWorldModel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from drivewm.config import load_config
from drivewm.pipeline import DriveWorldPipeline
from drivewm.registry import DATASETS, MODEL_ADAPTERS


def main() -> None:
    parser = argparse.ArgumentParser(prog="drivewm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Run video generation")
    generate.add_argument("--config", required=True, help="Path to YAML config")

    train = subparsers.add_parser("train", help="Run diffusers-native training")
    train.add_argument("--config", required=True, help="Path to YAML config")

    subparsers.add_parser("list", help="List registered datasets and model adapters")

    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps({"datasets": DATASETS.names(), "models": MODEL_ADAPTERS.names()}, indent=2))
        return

    config = load_config(args.config)

    if args.command == "train":
        from drivewm.training import DiffusersTrainer

        result = DiffusersTrainer(config).run()
        print(json.dumps(result, indent=2, default=str))
        return

    pipeline = DriveWorldPipeline(config)
    results = pipeline.run()
    print(json.dumps([asdict(result) for result in results], indent=2, default=str))


if __name__ == "__main__":
    main()
