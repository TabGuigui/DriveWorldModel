"""Hugging Face V-JEPA model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoVideoProcessor, VJEPA2Model

from drivewm.config import ExperimentConfig


@dataclass
class VJEPAComponents:
    model: VJEPA2Model
    video_processor: AutoVideoProcessor
    frames_per_clip: int
    patch_size: int
    tubelet_size: int
    crop_size: int
    num_tokens: int


def load_vjepa_components(config: ExperimentConfig, torch_dtype: torch.dtype) -> VJEPAComponents:
    pretrained = (
        config.model.extra.get("pretrained_model_name_or_path")
        or config.model.checkpoint
        or config.model.variant
    )
    if not pretrained:
        raise ValueError("V-JEPA training requires model.variant, model.checkpoint, or model.extra.pretrained_model_name_or_path.")

    revision = config.model.extra.get("revision")
    attn_implementation = config.model.extra.get("attn_implementation")
    video_processor = AutoVideoProcessor.from_pretrained(pretrained, revision=revision)
    model = VJEPA2Model.from_pretrained(
        pretrained,
        revision=revision,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    frames_per_clip = int(model.config.frames_per_clip)
    patch_size = int(model.config.patch_size)
    tubelet_size = int(model.config.tubelet_size)
    crop_size = int(model.config.crop_size)
    num_tokens = (frames_per_clip // tubelet_size) * (crop_size // patch_size) * (crop_size // patch_size)
    return VJEPAComponents(
        model=model,
        video_processor=video_processor,
        frames_per_clip=frames_per_clip,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        crop_size=crop_size,
        num_tokens=num_tokens,
    )
