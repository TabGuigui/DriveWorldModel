"""Diffusers-native training loop scaffold."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import drivewm.models  # noqa: F401 - register model adapters
from drivewm.config import ExperimentConfig
from drivewm.models.diffusers_backend import _resolve_object, _torch_dtype
from drivewm.registry import MODEL_ADAPTERS
from drivewm.training.data import DriveWorldTrainingDataset, collate_training_batch


class DiffusersTrainer:
    """Load a diffusers video pipeline and run an accelerate training loop.

    The first implementation intentionally keeps model-specific noise/loss logic
    behind `training.extra.loss_adapter`. This lets each model family grow its
    exact native training step without changing the CLI or dataset contract.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        model_cls = MODEL_ADAPTERS.get(config.model.family)
        self.model_adapter = model_cls(config.model)

    def run(self) -> dict[str, Any]:
        return self.train()

    def train(self) -> dict[str, Any]:
        try:
            import torch
            from accelerate import Accelerator
            from torch.utils.data import DataLoader
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Training requires torch and accelerate. Install with "
                "`pip install -e '.[training]'`."
            ) from exc

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision=self.config.training.mixed_precision,
        )
        if self.config.training.seed is not None:
            from accelerate.utils import set_seed

            set_seed(self.config.training.seed)

        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = self._load_pipeline()
        trainable_module = self._select_trainable_module(pipeline)
        self._freeze_non_trainable_modules(pipeline, trainable_module)
        trainable_module.train()

        optimizer = torch.optim.AdamW(
            trainable_module.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        dataset = DriveWorldTrainingDataset(self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.train_batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            collate_fn=collate_training_batch,
        )

        trainable_module, optimizer, dataloader = accelerator.prepare(
            trainable_module, optimizer, dataloader
        )

        global_step = 0
        for epoch in range(self.config.training.num_train_epochs):
            for batch in dataloader:
                with accelerator.accumulate(trainable_module):
                    loss = self._compute_loss(pipeline, trainable_module, batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_module.parameters(),
                            self.config.training.max_grad_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": loss.detach().float().item()}, step=global_step)
                if self._should_checkpoint(global_step):
                    self._save_checkpoint(accelerator, trainable_module, output_dir, global_step)
                if (
                    self.config.training.max_train_steps is not None
                    and global_step >= self.config.training.max_train_steps
                ):
                    break
            if (
                self.config.training.max_train_steps is not None
                and global_step >= self.config.training.max_train_steps
            ):
                break

        self._save_checkpoint(accelerator, trainable_module, output_dir, global_step, final=True)
        accelerator.end_training()
        return {"output_dir": str(output_dir), "global_step": global_step}

    def _load_pipeline(self):
        pipeline_cls = _resolve_object(self.model_adapter.pipeline_class_path)
        pretrained = (
            self.config.model.extra.get("pretrained_model_name_or_path")
            or self.config.model.checkpoint
            or self.config.model.variant
        )
        load_kwargs = dict(self.config.model.extra.get("load_kwargs", {}))
        torch_dtype = _torch_dtype(self.config.model.precision)
        if torch_dtype is not None and "torch_dtype" not in load_kwargs:
            load_kwargs["torch_dtype"] = torch_dtype
        pipeline = pipeline_cls.from_pretrained(pretrained, **load_kwargs)
        if self.config.model.extra.get("enable_vae_tiling", True) and hasattr(pipeline, "vae"):
            vae = getattr(pipeline, "vae")
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
        return pipeline

    def _select_trainable_module(self, pipeline):
        name = self.config.training.trainable_module
        if not hasattr(pipeline, name):
            raise AttributeError(
                f"Pipeline {type(pipeline).__name__} does not expose trainable module '{name}'."
            )
        return getattr(pipeline, name)

    def _freeze_non_trainable_modules(self, pipeline, trainable_module) -> None:
        for _, component in getattr(pipeline, "components", {}).items():
            if component is trainable_module or not hasattr(component, "requires_grad_"):
                continue
            component.requires_grad_(False)
            if hasattr(component, "eval"):
                component.eval()

    def _compute_loss(self, pipeline, trainable_module, batch):
        adapter_path = self.config.training.extra.get("loss_adapter")
        if not adapter_path:
            raise NotImplementedError(
                "Set training.extra.loss_adapter to a callable that computes the "
                "diffusers-native training loss for this model family. The callable "
                "receives (pipeline, trainable_module, batch, config)."
            )
        loss_fn = _resolve_object(adapter_path)
        return loss_fn(pipeline, trainable_module, batch, self.config)

    def _should_checkpoint(self, global_step: int) -> bool:
        steps = self.config.training.checkpointing_steps
        return steps > 0 and global_step > 0 and global_step % steps == 0

    def _save_checkpoint(self, accelerator, trainable_module, output_dir: Path, step: int, final: bool = False) -> None:
        if not accelerator.is_main_process:
            return
        name = "final" if final else f"step-{step}"
        unwrapped = accelerator.unwrap_model(trainable_module)
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(path)
        else:
            accelerator.save(unwrapped.state_dict(), path / "pytorch_model.bin")

