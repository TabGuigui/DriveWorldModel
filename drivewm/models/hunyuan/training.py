"""HunyuanVideo training hooks."""


def compute_loss(pipeline, trainable_module, batch, config):
    raise NotImplementedError(
        "HunyuanVideo training loss is not implemented yet. Use this hook to build "
        "the model-specific latent/noise objective and condition injection."
    )
