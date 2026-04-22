"""Training dataset wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import drivewm.data  # noqa: F401 - register dataset adapters
from drivewm.config import ExperimentConfig
from drivewm.conditions import ConditionBuilder
from drivewm.registry import DATASETS


class DriveWorldTrainingDataset:
    """Map-style dataset backed by the existing driving dataset adapters."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        dataset_cls = DATASETS.get(config.dataset.name)
        self.dataset = dataset_cls(config.dataset)
        self.condition_builder = ConditionBuilder(config.conditioning)
        self.samples = list(self.dataset.iter_samples())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        conditions = self.condition_builder.build(sample)
        target_video = _resolve_target_video(sample.metadata, self.config)
        return {
            "sample": sample,
            "sample_id": sample.sample_id,
            "prompt": self.config.generation.prompt or sample.prompt,
            "history_images": conditions.history_images,
            "future_trajectory": conditions.future_trajectory,
            "target_video": target_video,
        }


def collate_training_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": [item["sample"] for item in items],
        "sample_ids": [item["sample_id"] for item in items],
        "prompts": [item["prompt"] for item in items],
        "history_images": [item["history_images"] for item in items],
        "future_trajectory": [item["future_trajectory"] for item in items],
        "target_videos": [item["target_video"] for item in items],
    }


def _resolve_target_video(metadata: dict[str, Any], config: ExperimentConfig) -> Path:
    key = config.training.target_video_key
    raw_path = metadata.get(key) or metadata.get("video") or metadata.get("future_video")
    if raw_path is None:
        raise KeyError(
            f"Training sample is missing target video metadata. Expected '{key}', "
            "'video', or 'future_video' in the manifest line."
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(config.dataset.root) / path
    return path
