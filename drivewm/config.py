"""Configuration schema for driving world-model experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    yaml = None


@dataclass
class DatasetConfig:
    name: str
    root: str
    split: str = "train"
    manifest_path: str | None = None
    camera_names: list[str] = field(default_factory=lambda: ["CAM_FRONT"])
    history_frames: int = 4
    future_steps: int = 8
    sample_stride: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditioningConfig:
    use_history_images: bool = True
    use_future_trajectory: bool = False
    trajectory_frame: str = "ego"
    trajectory_fields: list[str] = field(default_factory=lambda: ["x", "y", "yaw"])
    normalize_trajectory: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    family: str
    variant: str
    checkpoint: str | None = None
    precision: str = "bf16"
    device: str = "cuda"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    output_dir: str = "outputs"
    num_frames: int = 16
    fps: int = 8
    width: int = 832
    height: int = 480
    seed: int | None = None
    prompt: str = ""
    negative_prompt: str = ""
    max_samples: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/train"
    train_batch_size: int = 1
    num_train_epochs: int = 1
    max_train_steps: int | None = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int | None = None
    dataloader_num_workers: int = 0
    checkpointing_steps: int = 500
    mode: str = "full"
    trainable_module: str = "transformer"
    target_video_key: str = "target_video"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    conditioning: ConditioningConfig
    model: ModelConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    name: str = "drivewm-experiment"


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _load_yaml(handle.read(), config_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at top level of {config_path}")
    return parse_config(raw)


def _load_yaml(text: str, config_path: Path) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _parse_simple_yaml(text, config_path)


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    required = {"dataset", "conditioning", "model"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(sorted(missing))}")

    dataset = _build_dataclass(DatasetConfig, raw["dataset"])
    conditioning = _build_dataclass(ConditioningConfig, raw["conditioning"])
    model = _build_dataclass(ModelConfig, raw["model"])
    generation = _build_dataclass(GenerationConfig, raw.get("generation", {}))
    training = _build_dataclass(TrainingConfig, raw.get("training", {}))
    return ExperimentConfig(
        name=raw.get("name", "drivewm-experiment"),
        dataset=dataset,
        conditioning=conditioning,
        model=model,
        generation=generation,
        training=training,
    )


def _build_dataclass(cls: type, values: dict[str, Any]) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"Expected mapping for {cls.__name__}")

    field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    known = {key: value for key, value in values.items() if key in field_names}
    extra = {key: value for key, value in values.items() if key not in field_names}
    if "extra" in field_names:
        known["extra"] = {**known.get("extra", {}), **extra}
    return cls(**known)


def _parse_simple_yaml(text: str, config_path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by the example configs.

    This keeps basic config parsing usable before optional dependencies are
    installed. Full YAML features still require PyYAML.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"{config_path}:{line_number}: expected 'key: value'")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{config_path}:{line_number}: invalid indentation")

        parent = stack[-1][1]
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
