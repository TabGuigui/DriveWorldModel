"""nuScenes dataset adapter.

This first implementation consumes a normalized jsonl manifest. A later adapter can
swap in the official nuScenes SDK while preserving the SceneSample interface.
"""

from drivewm.data.base import ManifestDataset
from drivewm.registry import DATASETS


@DATASETS.register("nuscenes")
class NuScenesDataset(ManifestDataset):
    dataset_name = "nuscenes"
