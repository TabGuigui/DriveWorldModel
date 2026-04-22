"""Model adapter registration."""

from drivewm.models.base import GenerationRequest, GenerationResult, VideoModelAdapter
from drivewm.models.cogvideox import CogVideoXAdapter, CogVideoXLoRATrainer
from drivewm.models.diffusers_backend import DiffusersVideoAdapter
from drivewm.models.hunyuan.adapter import HunyuanVideoAdapter
from drivewm.models.wan.adapter import WanAdapter

__all__ = [
    "CogVideoXAdapter",
    "CogVideoXLoRATrainer",
    "GenerationRequest",
    "GenerationResult",
    "HunyuanVideoAdapter",
    "DiffusersVideoAdapter",
    "VideoModelAdapter",
    "WanAdapter",
]
