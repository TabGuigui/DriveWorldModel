"""HunyuanVideo diffusers adapter."""

from drivewm.models.diffusers_backend import DiffusersVideoAdapter
from drivewm.registry import MODEL_ADAPTERS


@MODEL_ADAPTERS.register("hunyuan")
@MODEL_ADAPTERS.register("hunyuanvideo")
class HunyuanVideoAdapter(DiffusersVideoAdapter):
    family = "hunyuan"
    default_pipeline_class = "diffusers:HunyuanVideo15ImageToVideoPipeline"
    default_history_input_key = "image"
