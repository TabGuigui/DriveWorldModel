"""Video training datasets built from DriveWorldModel dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import drivewm.data  # noqa: F401 - register dataset adapters
from drivewm.config import ExperimentConfig
from drivewm.registry import DATASETS


@dataclass
class VideoTrainingRecord:
    sample_id: str
    prompt: str
    target_video: Path
    metadata: dict[str, Any]


def load_video_training_records(config: ExperimentConfig) -> list[VideoTrainingRecord]:
    dataset_cls = DATASETS.get(config.dataset.name)
    dataset = dataset_cls(config.dataset)
    records: list[VideoTrainingRecord] = []
    for sample in dataset.iter_samples():
        target_video = _target_video_path(sample.metadata, config)
        records.append(
            VideoTrainingRecord(
                sample_id=sample.sample_id,
                prompt=config.generation.prompt or sample.prompt,
                target_video=target_video,
                metadata={
                    **sample.metadata,
                    "scene_id": sample.scene_id,
                    "dataset": sample.dataset,
                    "timestamp": sample.timestamp,
                },
            )
        )
    return records


def _target_video_path(metadata: dict[str, Any], config: ExperimentConfig) -> Path:
    raw_path = (
        metadata.get(config.training.target_video_key)
        or metadata.get("target_video")
        or metadata.get("video")
        or metadata.get("future_video")
    )
    if raw_path is None:
        raise KeyError(
            f"Training sample missing '{config.training.target_video_key}', "
            "'target_video', 'video', or 'future_video'."
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(config.dataset.root) / path
    return path
