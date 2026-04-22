"""Shared data structures for driving video generation samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraFrame:
    path: Path
    camera_name: str
    timestamp: int | float | None = None
    intrinsics: list[list[float]] | None = None
    extrinsics: list[list[float]] | None = None


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw: float | None = None
    t: float | None = None
    velocity: float | None = None


@dataclass
class SceneSample:
    dataset: str
    scene_id: str
    sample_id: str
    timestamp: int | float | None = None
    history_images: list[CameraFrame] = field(default_factory=list)
    future_trajectory: list[TrajectoryPoint] = field(default_factory=list)
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
