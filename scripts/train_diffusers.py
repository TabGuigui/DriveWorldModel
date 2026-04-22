"""Train DriveWorldModel adapters with a diffusers-native loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivewm.config import load_config
from drivewm.training import DiffusersTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    result = DiffusersTrainer(config).run()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
