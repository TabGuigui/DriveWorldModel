"""Dataset adapter protocol and manifest-backed implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from drivewm.config import DatasetConfig
from drivewm.data.types import CameraFrame, SceneSample, TrajectoryPoint


class DrivingDataset(ABC):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.root = Path(config.root)

    @abstractmethod
    def iter_samples(self) -> Iterator[SceneSample]:
        """Yield scene samples ready for condition assembly."""


class ManifestDataset(DrivingDataset):
    dataset_name = "manifest"

    def iter_samples(self) -> Iterator[SceneSample]:
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Create a jsonl manifest or set dataset.manifest_path."
            )

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield self._record_to_sample(record, line_number)

    def _manifest_path(self) -> Path:
        if self.config.manifest_path:
            return Path(self.config.manifest_path)
        return self.root / "manifests" / f"{self.config.split}.jsonl"

    def _record_to_sample(self, record: dict, line_number: int) -> SceneSample:
        images = self._parse_history_images(record)
        trajectory = [
            TrajectoryPoint(
                x=float(point["x"]),
                y=float(point["y"]),
                yaw=_maybe_float(point.get("yaw")),
                t=_maybe_float(point.get("t")),
                velocity=_maybe_float(point.get("velocity")),
            )
            for point in record.get("future_trajectory", [])
        ]
        return SceneSample(
            dataset=self.dataset_name,
            scene_id=str(record.get("scene_id", record.get("scene_token", "unknown"))),
            sample_id=str(record.get("sample_id", record.get("sample_token", line_number))),
            timestamp=record.get("timestamp"),
            history_images=images,
            future_trajectory=trajectory,
            prompt=record.get("prompt", ""),
            metadata={key: value for key, value in record.items() if key not in _KNOWN_KEYS},
        )

    def _parse_history_images(self, record: dict) -> list[CameraFrame]:
        history = record.get("history_images", [])
        frames: list[CameraFrame] = []
        for item in history:
            if isinstance(item, str):
                path = item
                camera_name = "CAM_FRONT"
                timestamp = None
                intrinsics = None
                extrinsics = None
            else:
                path = item["path"]
                camera_name = item.get("camera_name", item.get("camera", "CAM_FRONT"))
                timestamp = item.get("timestamp")
                intrinsics = item.get("intrinsics")
                extrinsics = item.get("extrinsics")

            if camera_name not in self.config.camera_names:
                continue
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = self.root / resolved
            frames.append(
                CameraFrame(
                    path=resolved,
                    camera_name=camera_name,
                    timestamp=timestamp,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                )
            )
        return frames[-self.config.history_frames :]


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


_KNOWN_KEYS = {
    "scene_id",
    "scene_token",
    "sample_id",
    "sample_token",
    "timestamp",
    "history_images",
    "future_trajectory",
    "prompt",
}
