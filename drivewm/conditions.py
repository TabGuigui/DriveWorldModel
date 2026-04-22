"""Condition assembly for driving video generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drivewm.config import ConditioningConfig
from drivewm.data.types import SceneSample, TrajectoryPoint


@dataclass
class ConditionBundle:
    history_images: list[Path] = field(default_factory=list)
    future_trajectory: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> list[str]:
        modalities: list[str] = []
        if self.history_images:
            modalities.append("history_images")
        if self.future_trajectory:
            modalities.append("future_trajectory")
        return modalities


class ConditionBuilder:
    def __init__(self, config: ConditioningConfig) -> None:
        self.config = config

    def build(self, sample: SceneSample) -> ConditionBundle:
        history_images = []
        if self.config.use_history_images:
            history_images = [frame.path for frame in sample.history_images]

        future_trajectory: list[dict[str, float]] = []
        if self.config.use_future_trajectory:
            future_trajectory = [
                self._trajectory_point_to_dict(point)
                for point in sample.future_trajectory
            ]
            if self.config.normalize_trajectory:
                future_trajectory = self._normalize_trajectory(future_trajectory)

        return ConditionBundle(
            history_images=history_images,
            future_trajectory=future_trajectory,
            metadata={
                "trajectory_frame": self.config.trajectory_frame,
                "trajectory_fields": self.config.trajectory_fields,
                "sample_id": sample.sample_id,
                "scene_id": sample.scene_id,
                "dataset": sample.dataset,
            },
        )

    def _trajectory_point_to_dict(self, point: TrajectoryPoint) -> dict[str, float]:
        values = {
            "x": point.x,
            "y": point.y,
            "yaw": point.yaw,
            "t": point.t,
            "velocity": point.velocity,
        }
        return {
            field: float(values[field])
            for field in self.config.trajectory_fields
            if values.get(field) is not None
        }

    @staticmethod
    def _normalize_trajectory(points: list[dict[str, float]]) -> list[dict[str, float]]:
        if not points:
            return []
        origin_x = points[0].get("x", 0.0)
        origin_y = points[0].get("y", 0.0)
        normalized: list[dict[str, float]] = []
        for point in points:
            item = dict(point)
            if "x" in item:
                item["x"] -= origin_x
            if "y" in item:
                item["y"] -= origin_y
            normalized.append(item)
        return normalized
