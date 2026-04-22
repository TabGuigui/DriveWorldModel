"""Diffusers-backed model adapter utilities."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from drivewm.models.base import GenerationRequest, GenerationResult, VideoModelAdapter


class DiffusersVideoAdapter(VideoModelAdapter):
    """Base adapter for Hugging Face diffusers video pipelines.

    Subclasses provide a default pipeline class. Config can override it with:

    model:
      extra:
        pipeline_class: diffusers:WanVideoToVideoPipeline
        pretrained_model_name_or_path: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
        history_input_key: video
        trajectory_kwarg: driving_trajectory
        load_kwargs: {}
        call_kwargs: {}
    """

    default_pipeline_class = "diffusers:DiffusionPipeline"
    default_history_input_key = "video"

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        pipe = self._load_pipeline()
        kwargs = self._build_call_kwargs(request, pipe)
        output = pipe(**kwargs)
        frames = _extract_frames(output)
        output_path = self.output_path_for(request)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _export_to_video(frames, output_path, request.generation.fps)
        return GenerationResult(
            sample_id=request.sample.sample_id,
            output_path=output_path,
            metadata={
                "model_family": self.family,
                "model_variant": self.config.variant,
                "pipeline_class": self.pipeline_class_path,
                "modalities": request.conditions.modalities,
            },
        )

    @property
    def pipeline_class_path(self) -> str:
        return self.config.extra.get("pipeline_class", self.default_pipeline_class)

    def _load_pipeline(self):
        pipeline_cls = _resolve_object(self.pipeline_class_path)
        pretrained = (
            self.config.extra.get("pretrained_model_name_or_path")
            or self.config.checkpoint
            or self.config.variant
        )
        load_kwargs = dict(self.config.extra.get("load_kwargs", {}))
        torch_dtype = _torch_dtype(self.config.precision)
        if torch_dtype is not None and "torch_dtype" not in load_kwargs:
            load_kwargs["torch_dtype"] = torch_dtype

        pipe = pipeline_cls.from_pretrained(pretrained, **load_kwargs)
        if self.config.extra.get("enable_model_cpu_offload", False):
            pipe.enable_model_cpu_offload()
        elif self.config.device:
            pipe = pipe.to(self.config.device)

        if self.config.extra.get("enable_vae_tiling", True) and hasattr(pipe, "vae"):
            vae = getattr(pipe, "vae")
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
        return pipe

    def _build_call_kwargs(self, request: GenerationRequest, pipe) -> dict[str, Any]:
        generation = request.generation
        kwargs: dict[str, Any] = {
            "prompt": generation.prompt,
            "negative_prompt": generation.negative_prompt,
            "height": generation.height,
            "width": generation.width,
            "num_frames": generation.num_frames,
        }
        kwargs.update(generation.extra)
        kwargs.update(self.config.extra.get("call_kwargs", {}))

        if generation.seed is not None:
            generator = _torch_generator(self.config.device, generation.seed)
            if generator is not None:
                kwargs["generator"] = generator

        if request.conditions.history_images:
            frames = _load_history_images(request.conditions.history_images)
            history_key = self.config.extra.get(
                "history_input_key", self.default_history_input_key
            )
            if history_key == "image":
                kwargs[history_key] = frames[-1]
            else:
                kwargs[history_key] = frames

        trajectory_kwarg = self.config.extra.get("trajectory_kwarg")
        if trajectory_kwarg and request.conditions.future_trajectory:
            kwargs[trajectory_kwarg] = request.conditions.future_trajectory

        return _filter_kwargs_for_pipeline(pipe, kwargs)


def _resolve_object(path: str):
    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = "diffusers", path
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Real generation requires diffusers. Install with "
            "`pip install -e '.[diffusers]'`."
        ) from exc
    return getattr(module, object_name)


def _torch_dtype(precision: str):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Real generation requires torch and diffusers. Install with "
            "`pip install -e '.[diffusers]'`."
        ) from exc

    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(precision.lower())


def _torch_generator(device: str, seed: int):
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def _load_history_images(paths: list[Path]):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "History-image conditioning requires Pillow. Install with "
            "`pip install -e '.[diffusers]'`."
        ) from exc

    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    return frames


def _filter_kwargs_for_pipeline(pipe, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(pipe.__call__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _extract_frames(output):
    if hasattr(output, "frames"):
        frames = output.frames
    elif isinstance(output, tuple):
        frames = output[0]
    else:
        raise TypeError(f"Cannot extract frames from output type {type(output)!r}")
    if frames and isinstance(frames, list) and frames and isinstance(frames[0], list):
        return frames[0]
    return frames


def _export_to_video(frames, output_path: Path, fps: int) -> None:
    try:
        from diffusers.utils import export_to_video
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Video export requires diffusers. Install with "
            "`pip install -e '.[diffusers]'`."
        ) from exc
    export_to_video(frames, str(output_path), fps=fps)
