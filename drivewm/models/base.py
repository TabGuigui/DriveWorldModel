"""Model adapter interface for video world models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drivewm.conditions import ConditionBundle
from drivewm.config import GenerationConfig, ModelConfig
from drivewm.data.types import SceneSample


@dataclass
class GenerationRequest:
    sample: SceneSample
    conditions: ConditionBundle
    generation: GenerationConfig
    model: ModelConfig


@dataclass
class GenerationResult:
    sample_id: str
    output_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoModelAdapter(ABC):
    family = "base"
    supported_condition_sets: tuple[tuple[str, ...], ...] = (
        ("history_images",),
        ("history_images", "future_trajectory"),
    )

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def validate_request(self, request: GenerationRequest) -> None:
        modalities = tuple(request.conditions.modalities)
        if modalities not in self.supported_condition_sets:
            supported = ["+".join(item) for item in self.supported_condition_sets]
            current = "+".join(modalities) or "<none>"
            raise ValueError(
                f"{self.family} does not support condition set {current}. "
                f"Supported: {', '.join(supported)}"
            )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.validate_request(request)
        return self._generate(request)

    def output_path_for(self, request: GenerationRequest) -> Path:
        safe_family = self.family.replace("/", "-")
        filename = f"{request.sample.dataset}_{request.sample.sample_id}_{safe_family}.mp4"
        return Path(request.generation.output_dir) / filename

    @abstractmethod
    def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a real backend implementation."""
