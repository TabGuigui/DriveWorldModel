"""End-to-end orchestration for dataset, conditions, and model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import drivewm.data  # noqa: F401 - import registers dataset adapters
import drivewm.models  # noqa: F401 - import registers model adapters
from drivewm.config import ExperimentConfig
from drivewm.conditions import ConditionBuilder
from drivewm.models.base import GenerationRequest, GenerationResult
from drivewm.registry import DATASETS, MODEL_ADAPTERS


class DriveWorldPipeline:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.dataset = DATASETS.get(config.dataset.name)(config.dataset)
        self.condition_builder = ConditionBuilder(config.conditioning)
        self.model = MODEL_ADAPTERS.get(config.model.family)(config.model)

    def run(self) -> list[GenerationResult]:
        Path(self.config.generation.output_dir).mkdir(parents=True, exist_ok=True)
        results: list[GenerationResult] = []
        for index, request in enumerate(self.iter_requests()):
            if index >= self.config.generation.max_samples:
                break
            results.append(self.model.generate(request))
        return results

    def iter_requests(self) -> Iterator[GenerationRequest]:
        for sample in self.dataset.iter_samples():
            conditions = self.condition_builder.build(sample)
            prompt = self.config.generation.prompt or sample.prompt
            generation = self.config.generation
            if prompt != generation.prompt:
                generation = _generation_with_prompt(generation, prompt)
            yield GenerationRequest(
                sample=sample,
                conditions=conditions,
                generation=generation,
                model=self.config.model,
            )


def _generation_with_prompt(generation, prompt: str):
    from dataclasses import replace

    return replace(generation, prompt=prompt)
