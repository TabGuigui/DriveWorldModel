"""CogVideoX LoRA training implementation."""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Optional

from drivewm.config import ExperimentConfig
from drivewm.registry import MODEL_TRAINERS
from drivewm.training.video_dataset import load_video_training_records

logger = logging.getLogger(__name__)


@MODEL_TRAINERS.register("cogvideox")
@MODEL_TRAINERS.register("cogvideox-1.5")
class CogVideoXLoRATrainer:
    def __init__(self, config: ExperimentConfig, args=None) -> None:
        self.config = config
        self.args = args

    def train(self) -> None:
        import numpy as np
        import torch
        import torchvision.transforms as TT
        import transformers
        from accelerate import Accelerator
        from accelerate.logging import get_logger
        from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
        from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
        from torch.utils.data import DataLoader
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize
        from tqdm.auto import tqdm
        from transformers import AutoTokenizer, T5EncoderModel

        import diffusers
        from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
        from diffusers.models.embeddings import get_3d_rotary_pos_embed
        from diffusers.optimization import get_scheduler
        from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
        from diffusers.training_utils import cast_training_params, free_memory
        from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video
        from diffusers.utils.torch_utils import is_compiled_module

        config = self.config
        training = config.training
        generation = config.generation
        train_extra = training.extra
        model_extra = config.model.extra
        hf_logger = get_logger(__name__)

        output_dir = Path(training.output_dir)
        logging_dir = output_dir / train_extra.get("logging_dir", "logs")
        accelerator_project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            mixed_precision=training.mixed_precision,
            log_with=getattr(self.args, "report_to", None) or train_extra.get("report_to"),
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        hf_logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if training.seed is not None:
            set_seed(training.seed)
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)

        pretrained = pretrained_model_name_or_path(config)
        revision = model_extra.get("revision")
        variant = model_extra.get("variant")

        tokenizer = AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer", revision=revision)
        text_encoder = T5EncoderModel.from_pretrained(pretrained, subfolder="text_encoder", revision=revision)
        load_dtype = torch.bfloat16 if "5b" in pretrained.lower() else torch.float16
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            pretrained,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=revision,
            variant=variant,
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(pretrained, subfolder="vae", revision=revision, variant=variant)
        scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

        if train_extra.get("enable_slicing", False):
            vae.enable_slicing()
        if model_extra.get("enable_vae_tiling", True) or train_extra.get("enable_tiling", False):
            vae.enable_tiling()

        text_encoder.requires_grad_(False)
        transformer.requires_grad_(False)
        vae.requires_grad_(False)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            raise ValueError("MPS does not support bf16 mixed precision; use fp16 or fp32.")

        text_encoder.to(accelerator.device, dtype=weight_dtype)
        transformer.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)

        if train_extra.get("gradient_checkpointing", False):
            transformer.enable_gradient_checkpointing()

        lora_rank = int(train_extra.get("rank", 64))
        lora_alpha = float(train_extra.get("lora_alpha", lora_rank))
        target_modules = train_extra.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"])
        transformer.add_adapter(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=True,
                target_modules=target_modules,
            )
        )

        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            return model._orig_mod if is_compiled_module(model) else model

        def save_model_hook(models, weights, save_dir):
            if accelerator.is_main_process:
                transformer_lora_layers = None
                for model in models:
                    if isinstance(model, type(unwrap_model(transformer))):
                        transformer_lora_layers = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")
                    weights.pop()
                CogVideoXPipeline.save_lora_weights(save_dir, transformer_lora_layers=transformer_lora_layers)

        def load_model_hook(models, input_dir):
            transformer_ = None
            while models:
                model = models.pop()
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"Unexpected load model: {model.__class__}")
            lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)
            transformer_state_dict = {
                key.replace("transformer.", ""): value
                for key, value in lora_state_dict.items()
                if key.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if training.mixed_precision == "fp16":
                cast_training_params([transformer_])

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        if train_extra.get("allow_tf32", False) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        if training.mixed_precision == "fp16":
            cast_training_params([transformer], dtype=torch.float32)

        params_to_optimize = [
            {"params": [param for param in transformer.parameters() if param.requires_grad], "lr": training.learning_rate}
        ]
        use_deepspeed_optimizer = (
            accelerator.state.deepspeed_plugin is not None
            and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
        )
        use_deepspeed_scheduler = (
            accelerator.state.deepspeed_plugin is not None
            and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
        )
        if use_deepspeed_optimizer:
            from accelerate.utils import DummyOptim

            optimizer = DummyOptim(
                params_to_optimize,
                lr=training.learning_rate,
                betas=(float(train_extra.get("adam_beta1", 0.9)), float(train_extra.get("adam_beta2", 0.95))),
                eps=float(train_extra.get("adam_epsilon", 1e-8)),
                weight_decay=float(train_extra.get("adam_weight_decay", training.weight_decay)),
            )
        else:
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                betas=(float(train_extra.get("adam_beta1", 0.9)), float(train_extra.get("adam_beta2", 0.95))),
                eps=float(train_extra.get("adam_epsilon", 1e-8)),
                weight_decay=float(train_extra.get("adam_weight_decay", training.weight_decay)),
            )

        train_dataset = CogVideoXManifestDataset(
            config=config,
            torch=torch,
            np=np,
            transforms=TT,
            resize_fn=resize,
            interpolation_mode=InterpolationMode,
            tqdm=tqdm,
        )

        def encode_video(video):
            video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
            video = video.permute(0, 2, 1, 3, 4)
            return vae.encode(video).latent_dist

        encoded_videos = [None] * len(train_dataset.instance_videos)
        progress_encode_bar = tqdm(
            range(len(train_dataset.instance_videos)),
            desc="Encoding videos",
            disable=not accelerator.is_local_main_process,
        )
        for index, video in enumerate(train_dataset.instance_videos):
            encoded_videos[index] = encode_video(video)
            progress_encode_bar.update(1)
        progress_encode_bar.close()
        train_dataset.instance_videos = encoded_videos

        def collate_fn(examples):
            videos = [example["instance_video"].sample() * vae.config.scaling_factor for example in examples]
            prompts = [example["instance_prompt"] for example in examples]
            videos = torch.cat(videos)
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos.to(memory_format=torch.contiguous_format).float()
            return {"videos": videos, "prompts": prompts}

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=training.dataloader_num_workers,
        )

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training.gradient_accumulation_steps)
        max_train_steps = training.max_train_steps or training.num_train_epochs * num_update_steps_per_epoch
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=train_extra.get("lr_scheduler", "constant"),
                optimizer=optimizer,
                total_num_steps=max_train_steps * accelerator.num_processes,
                num_warmup_steps=int(train_extra.get("lr_warmup_steps", 0)) * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                train_extra.get("lr_scheduler", "constant"),
                optimizer=optimizer,
                num_warmup_steps=int(train_extra.get("lr_warmup_steps", 0)) * accelerator.num_processes,
                num_training_steps=max_train_steps * accelerator.num_processes,
                num_cycles=int(train_extra.get("lr_num_cycles", 1)),
                power=float(train_extra.get("lr_power", 1.0)),
            )

        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

        if accelerator.is_main_process:
            accelerator.init_trackers(train_extra.get("tracker_name", "drivewm-cogvideox-lora"), config=flatten_config(config))

        total_batch_size = training.train_batch_size * accelerator.num_processes * training.gradient_accumulation_steps
        num_trainable_parameters = sum(param.numel() for group in params_to_optimize for param in group["params"])
        hf_logger.info("***** Running CogVideoX LoRA training *****")
        hf_logger.info(f"  Dataset adapter = {config.dataset.name}")
        hf_logger.info(f"  Model family = {config.model.family}")
        hf_logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
        hf_logger.info(f"  Num examples = {len(train_dataset)}")
        hf_logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        hf_logger.info(f"  Num epochs = {num_train_epochs}")
        hf_logger.info(f"  Total train batch size = {total_batch_size}")
        hf_logger.info(f"  Total optimization steps = {max_train_steps}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        resume_from_checkpoint = getattr(self.args, "resume_from_checkpoint", None) or train_extra.get("resume_from_checkpoint")
        if resume_from_checkpoint:
            path = resolve_checkpoint(output_dir, resume_from_checkpoint)
            if path is not None:
                accelerator.print(f"Resuming from checkpoint {path.name}")
                accelerator.load_state(str(path))
                global_step = int(path.name.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch

        progress_bar = tqdm(
            range(max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not accelerator.is_local_main_process,
        )
        vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
        model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

        for _ in range(first_epoch, num_train_epochs):
            transformer.train()
            for batch in train_dataloader:
                with accelerator.accumulate(transformer):
                    model_input = batch["videos"].to(dtype=weight_dtype)
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        batch["prompts"],
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                    )

                    noise = torch.randn_like(model_input)
                    batch_size, num_frames, _, _, _ = model_input.shape
                    timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                    ).long()
                    noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                    image_rotary_emb = (
                        prepare_rotary_positional_embeddings(
                            height=generation.height,
                            width=generation.width,
                            num_frames=num_frames,
                            vae_scale_factor_spatial=vae_scale_factor_spatial,
                            patch_size=model_config.patch_size,
                            attention_head_dim=model_config.attention_head_dim,
                            device=accelerator.device,
                            get_3d_rotary_pos_embed=get_3d_rotary_pos_embed,
                            get_resize_crop_region_for_grid=get_resize_crop_region_for_grid,
                        )
                        if model_config.use_rotary_positional_embeddings
                        else None
                    )

                    model_output = transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                    model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                    alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                    weights = 1 / (1 - alphas_cumprod)
                    while len(weights.shape) < len(model_pred.shape):
                        weights = weights.unsqueeze(-1)

                    loss = torch.mean((weights * (model_pred - model_input) ** 2).reshape(batch_size, -1), dim=1).mean()
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), training.max_grad_norm)
                    if accelerator.state.deepspeed_plugin is None:
                        optimizer.step()
                        optimizer.zero_grad()
                    lr_scheduler.step()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % training.checkpointing_steps == 0:
                        save_checkpoint(
                            accelerator,
                            output_dir,
                            global_step,
                            int(train_extra.get("checkpoints_total_limit", 0)),
                        )

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if global_step >= max_train_steps:
                    break
            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            transformer = unwrap_model(transformer)
            dtype = (
                torch.float16
                if training.mixed_precision == "fp16"
                else torch.bfloat16
                if training.mixed_precision == "bf16"
                else torch.float32
            )
            transformer = transformer.to(dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)
            CogVideoXPipeline.save_lora_weights(
                save_directory=str(output_dir),
                transformer_lora_layers=transformer_lora_layers,
            )

            validation_prompt = getattr(self.args, "validation_prompt", None) or train_extra.get("validation_prompt")
            num_validation_videos = int(getattr(self.args, "num_validation_videos", 0) or 0)
            if validation_prompt and num_validation_videos > 0:
                del transformer
                free_memory()
                pipe = CogVideoXPipeline.from_pretrained(
                    pretrained,
                    revision=revision,
                    variant=variant,
                    torch_dtype=weight_dtype,
                )
                pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
                pipe.load_lora_weights(str(output_dir), adapter_name="cogvideox-lora")
                pipe.set_adapters(["cogvideox-lora"], [lora_alpha / lora_rank])
                pipe.to(accelerator.device)
                for index in range(num_validation_videos):
                    frames = pipe(
                        prompt=validation_prompt,
                        guidance_scale=float(train_extra.get("guidance_scale", 6.0)),
                        use_dynamic_cfg=bool(train_extra.get("use_dynamic_cfg", False)),
                        height=generation.height,
                        width=generation.width,
                        num_frames=generation.num_frames,
                    ).frames[0]
                    export_to_video(frames, str(output_dir / f"validation_{index}.mp4"), fps=generation.fps)

        accelerator.end_training()


class CogVideoXManifestDataset:
    def __init__(
        self,
        config: ExperimentConfig,
        torch,
        np,
        transforms,
        resize_fn,
        interpolation_mode,
        tqdm,
    ) -> None:
        self.config = config
        self.torch = torch
        self.np = np
        self.transforms = transforms
        self.resize_fn = resize_fn
        self.interpolation_mode = interpolation_mode
        self.height = config.generation.height
        self.width = config.generation.width
        self.video_reshape_mode = config.training.extra.get("video_reshape_mode", "center")
        self.max_num_frames = config.generation.num_frames
        self.skip_frames_start = int(config.training.extra.get("skip_frames_start", 0))
        self.skip_frames_end = int(config.training.extra.get("skip_frames_end", 0))
        self.id_token = config.training.extra.get("id_token", "")
        self.records = load_video_training_records(config)
        self.instance_prompts = [self.id_token + record.prompt for record in self.records]
        self.instance_video_paths = [record.target_video for record in self.records]
        self.instance_videos = self._preprocess_data(tqdm)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {
            "instance_prompt": self.instance_prompts[index],
            "instance_video": self.instance_videos[index],
        }

    def _resize_for_rectangle_crop(self, arr):
        image_size = self.height, self.width
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = self.resize_fn(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=self.interpolation_mode.BICUBIC,
            )
        else:
            arr = self.resize_fn(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=self.interpolation_mode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)
        delta_h = h - image_size[0]
        delta_w = w - image_size[1]
        if self.video_reshape_mode == "random":
            top = self.np.random.randint(0, delta_h + 1)
            left = self.np.random.randint(0, delta_w + 1)
        elif self.video_reshape_mode in {"center", "none"}:
            top, left = delta_h // 2, delta_w // 2
        else:
            raise ValueError(f"Unknown video_reshape_mode: {self.video_reshape_mode}")
        return self.transforms.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])

    def _preprocess_data(self, tqdm):
        try:
            import decord
        except ImportError as exc:
            raise ImportError("Install decord for CogVideoX training: `pip install decord`.") from exc

        decord.bridge.set_bridge("torch")
        videos = []
        progress = tqdm(range(len(self.instance_video_paths)), desc="Loading and resizing videos")
        for filename in self.instance_video_paths:
            if not filename.is_file():
                raise FileNotFoundError(f"Target video not found: {filename}")
            video_reader = decord.VideoReader(uri=filename.as_posix())
            video_num_frames = len(video_reader)
            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            else:
                step = max(1, (end_frame - start_frame) // self.max_num_frames)
                frames = video_reader.get_batch(list(range(start_frame, end_frame, step)))

            frames = frames[: self.max_num_frames]
            remainder = (3 + (frames.shape[0] % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            if frames.shape[0] == 0 or (frames.shape[0] - 1) % 4 != 0:
                raise ValueError(f"Video {filename} could not be trimmed to 4k+1 frames.")

            frames = (frames - 127.5) / 127.5
            frames = frames.permute(0, 3, 1, 2)
            videos.append(self._resize_for_rectangle_crop(frames).contiguous())
            progress.update(1)
        progress.close()
        return videos


def pretrained_model_name_or_path(config: ExperimentConfig) -> str:
    return (
        config.model.extra.get("pretrained_model_name_or_path")
        or config.model.checkpoint
        or config.model.variant
    )


def compute_prompt_embeddings(tokenizer, text_encoder, prompt, max_sequence_length, device, dtype):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    return prompt_embeds.view(len(prompt), seq_len, -1)


def prepare_rotary_positional_embeddings(
    height,
    width,
    num_frames,
    vae_scale_factor_spatial,
    patch_size,
    attention_head_dim,
    device,
    get_3d_rotary_pos_embed,
    get_resize_crop_region_for_grid,
):
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = 720 // (vae_scale_factor_spatial * patch_size)
    base_size_height = 480 // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    return get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        device=device,
    )


def save_checkpoint(accelerator, output_dir: Path, global_step: int, checkpoints_total_limit: int) -> None:
    if accelerator.is_main_process:
        if checkpoints_total_limit:
            checkpoints = sorted(
                [path for path in output_dir.iterdir() if path.name.startswith("checkpoint-")],
                key=lambda path: int(path.name.split("-")[1]),
            )
            if len(checkpoints) >= checkpoints_total_limit:
                for removing_checkpoint in checkpoints[: len(checkpoints) - checkpoints_total_limit + 1]:
                    shutil.rmtree(removing_checkpoint)
        save_path = output_dir / f"checkpoint-{global_step}"
        accelerator.save_state(str(save_path))
        logger.info("Saved state to %s", save_path)


def resolve_checkpoint(output_dir: Path, resume_from_checkpoint: str) -> Optional[Path]:
    if resume_from_checkpoint != "latest":
        path = Path(resume_from_checkpoint)
        return path if path.is_absolute() else output_dir / path
    checkpoints = sorted(
        [path for path in output_dir.iterdir() if path.name.startswith("checkpoint-")],
        key=lambda path: int(path.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def flatten_config(config: ExperimentConfig) -> dict:
    return {
        "name": config.name,
        "dataset": config.dataset.__dict__,
        "conditioning": config.conditioning.__dict__,
        "model": config.model.__dict__,
        "generation": config.generation.__dict__,
        "training": config.training.__dict__,
    }
