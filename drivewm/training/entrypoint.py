"""Config-driven training dispatch."""

from __future__ import annotations

from drivewm.config import ExperimentConfig
from drivewm.registry import MODEL_TRAINERS

import drivewm.models  # noqa: F401 - register trainers


def train_from_config(config: ExperimentConfig, args=None) -> None:
    trainer_cls = MODEL_TRAINERS.get(config.model.family)
    trainer = trainer_cls(config=config, args=args)
    trainer.train()
