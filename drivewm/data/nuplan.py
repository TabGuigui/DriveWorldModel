"""nuPlan dataset adapter.

This adapter reads the same normalized jsonl manifest shape as nuScenes. Keeping
the output identical is what lets model adapters stay dataset-agnostic.
"""

from drivewm.data.base import ManifestDataset
from drivewm.registry import DATASETS


@DATASETS.register("nuplan")
class NuPlanDataset(ManifestDataset):
    dataset_name = "nuplan"
