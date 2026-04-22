"""CogVideoX training hooks."""


def compute_loss(pipeline, trainable_module, batch, config):
    raise NotImplementedError(
        "CogVideoX training loss is not implemented yet. This hook should follow "
        "the diffusers CogVideoX LoRA/SFT recipe: load target video frames, encode "
        "latents with the 3D VAE, add scheduler noise, encode prompts, call the "
        "transformer, and return the denoising objective."
    )
