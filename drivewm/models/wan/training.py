"""Wan training hooks."""


def compute_loss(pipeline, trainable_module, batch, config):
    raise NotImplementedError(
        "Wan training loss is not implemented yet. Use this hook to encode target "
        "videos with the Wan VAE, sample scheduler noise/timesteps, inject "
        "history-image and trajectory conditions, and return the denoising loss."
    )
