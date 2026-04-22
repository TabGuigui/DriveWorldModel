"""Wan diffusers adapter."""

from drivewm.models.diffusers_backend import DiffusersVideoAdapter
from drivewm.registry import MODEL_ADAPTERS


@MODEL_ADAPTERS.register("wan")
class WanAdapter(DiffusersVideoAdapter):
    family = "wan"
    default_pipeline_class = "diffusers:WanVideoToVideoPipeline"
    default_history_input_key = "video"
