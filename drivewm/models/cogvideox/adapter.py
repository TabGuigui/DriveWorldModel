"""CogVideoX diffusers adapter."""

from drivewm.models.diffusers_backend import DiffusersVideoAdapter
from drivewm.registry import MODEL_ADAPTERS


@MODEL_ADAPTERS.register("cogvideox")
@MODEL_ADAPTERS.register("cogvideox-1.5")
class CogVideoXAdapter(DiffusersVideoAdapter):
    family = "cogvideox"
    default_pipeline_class = "diffusers:CogVideoXVideoToVideoPipeline"
    default_history_input_key = "video"
