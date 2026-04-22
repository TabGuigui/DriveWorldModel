# DriveWorldModel

DriveWorldModel is an integration framework for driving video world models. It
keeps datasets, conditioning signals, and video backends separated so the same
experiment can target Wan, CogVideoX, and Hunyuan-style generators on nuScenes
or nuPlan.

The first scaffold supports two conditioning modes:

- `history_images`: past camera frames as the visual condition.
- `history_images + future_trajectory`: past camera frames plus planned future
  ego trajectory.

## Layout

```text
drivewm/
  config.py              # YAML schema for dataset/model/conditioning/generation
  conditions.py          # Builds condition bundles from dataset samples
  pipeline.py            # Orchestrates dataset -> conditions -> model adapter
  data/
    base.py              # Manifest-backed dataset adapter
    nuscenes.py          # nuScenes registration
    nuplan.py            # nuPlan registration
  models/
    base.py              # VideoModelAdapter interface
    diffusers_backend.py # Shared diffusers adapter utilities
    wan/                 # Wan model family
    cogvideox/           # CogVideoX model family
    hunyuan/             # HunyuanVideo model family
configs/                 # Example experiment configs grouped by model family
examples/                # Tiny manifest examples for dry runs
```

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

For the tested CUDA 12.1 training stack:

```bash
pip install -r requirements-cu121.txt
pip install -e .
```

If you clone this repository elsewhere, initialize the local diffusers
submodule before training:

```bash
git submodule update --init --recursive
```

This requirements file targets PyTorch `2.3.0+cu121` and CogVideoX LoRA
training dependencies. Install DeepSpeed separately on Linux CUDA machines if
you need it:

```bash
DS_BUILD_OPS=0 pip install "deepspeed>=0.14.4,<0.16"
```

The built-in config loader supports the simple YAML used by the example files.
Install `pip install -e ".[yaml]"` if you want full YAML support.

List registered integrations:

```bash
drivewm list
```

Run generation:

```bash
drivewm generate \
  --config configs/wan/nuplan_history_traj.yaml
```

Run CogVideoX LoRA training:

```bash
scripts/train.sh
```

Run on multiple GPUs with `torchrun`:

```bash
GPU_IDS=0,1,2,3 NPROC_PER_NODE=4 scripts/train.sh
```

Override the port or precision when needed:

```bash
MASTER_PORT=29501 \
MIXED_PRECISION=bf16 \
scripts/train.sh
```

There is also a generic experimental training script:

```bash
python3 scripts/train_diffusers.py \
  --config configs/cogvideox/nuscenes_history_traj_train.yaml
```

The Wan, CogVideoX, and Hunyuan adapters are implemented as diffusers-backed
adapters. For inference, install:

```bash
pip install -e ".[diffusers]"
```

The default diffusers classes are:

- Wan: `diffusers:WanVideoToVideoPipeline`
- CogVideoX: `diffusers:CogVideoXVideoToVideoPipeline`
- Hunyuan: `diffusers:HunyuanVideo15ImageToVideoPipeline`

You can override the pipeline class per experiment:

```yaml
model:
  family: wan
  variant: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
  checkpoint: null
  precision: bf16
  device: cuda
  extra:
    pipeline_class: diffusers:WanVideoToVideoPipeline
    history_input_key: video
    trajectory_kwarg: driving_trajectory
    enable_model_cpu_offload: true
    enable_vae_tiling: true
```

`trajectory_kwarg` is intentionally configurable. Official text/video
diffusers pipelines do not all accept future trajectory tensors, but a
fine-tuned driving pipeline can expose a kwarg such as `driving_trajectory`.
The adapter inspects the pipeline call signature and only passes supported
kwargs.

## Dataset Manifest

Both nuScenes and nuPlan adapters currently consume a normalized jsonl manifest.
This lets the framework work without forcing the official SDKs into the core
abstraction. Each line is one sample:

```json
{
  "scene_id": "scene-0001",
  "sample_id": "sample-0001",
  "timestamp": 0,
  "history_images": [
    {"path": "samples/CAM_FRONT/000001.jpg", "camera_name": "CAM_FRONT", "timestamp": -3}
  ],
  "future_trajectory": [
    {"x": 0.0, "y": 0.0, "yaw": 0.0, "t": 0.0},
    {"x": 1.8, "y": 0.1, "yaw": 0.02, "t": 0.5}
  ],
  "prompt": "Clear daytime urban driving video."
}
```

Default manifest locations:

- nuScenes: `DATA_ROOT/manifests/SPLIT.jsonl`
- nuPlan: `DATA_ROOT/manifests/SPLIT.jsonl`

You can override this with `dataset.manifest_path` in a config.

## Configuration

Example for future trajectory conditioning:

```yaml
conditioning:
  use_history_images: true
  use_future_trajectory: true
  trajectory_frame: ego
  trajectory_fields: [x, y, yaw]
  normalize_trajectory: true
```

Example model selection:

```yaml
model:
  family: cogvideox
  variant: cogvideox-5b-driving
  checkpoint: /path/to/checkpoint
  precision: bf16
  device: cuda
```

Registered model families:

- `wan`
- `cogvideox`
- `hunyuan`

## Extension Points

To add a new dataset adapter, implement `DrivingDataset.iter_samples()` and
register it with `DATASETS.register("name")`.

To add a real model backend, implement `_generate()` in the relevant
`VideoModelAdapter`, or subclass `DiffusersVideoAdapter` for a new diffusers
pipeline. The adapter receives a `GenerationRequest` containing:

- `sample`: dataset metadata and IDs.
- `conditions.history_images`: resolved image paths.
- `conditions.future_trajectory`: normalized trajectory dictionaries.
- `generation`: video size, frame count, prompt, seed, and output directory.

## Training

Training is controlled by config. The shell calls one Python script, which reads
`dataset.name` from the dataset config, builds the dataset adapter, and then
uses one `DriveWorldTrainer`. The trainer reads `model.family` and currently
supports the official diffusers CogVideoX LoRA path:
`accelerate`, PyTorch `DataLoader`, `CogVideoXTransformer3DModel`, T5 prompt
encoding, CogVideoX VAE latent encoding, scheduler noise sampling, LoRA injected
into transformer attention projections, and a visible training loop. The first
script is:

```text
scripts/train.py
```

It loads:

- dataset samples through the configured `DATASETS` adapter.
- target videos from the DriveWorldModel jsonl manifest.
- `AutoencoderKLCogVideoX`, `CogVideoXTransformer3DModel`,
  `CogVideoXDPMScheduler`, and T5 components from the configured checkpoint.
- LoRA weights into transformer attention modules: `to_k`, `to_q`, `to_v`,
  `to_out.0`.

Manifest rows used for training should include a target video path:

```json
{"target_video": "videos/scene-0001/sample-0001.mp4"}
```

The CogVideoX LoRA settings live under `training.extra`:

```yaml
training:
  extra:
    rank: 64
    lora_alpha: 64
    target_modules: [to_k, to_q, to_v, to_out.0]
```

The current script is text/video CogVideoX LoRA fine-tuning. It does not yet
inject trajectory tensors into the transformer; that should be added after the
baseline nuScenes video LoRA path is stable.

Install training dependencies with:

```bash
pip install -e ".[training]"
```

## Validation

Run the lightweight scaffold validation:

```bash
python scripts/validate_framework.py
```
