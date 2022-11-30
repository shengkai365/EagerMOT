from __future__ import annotations
import os
from nuscenes import nuscenes
from nuscenes.utils import splits
from typing import Optional, List, Dict, Set, Any, Iterable, Sequence

from inputs import bbox
from configs import local_variables
from dataset_classes import mot_dataset
from dataset_classes.nuscenes import sequence
from dataset_classes.nuscenes import reporting



class MOTDatasetNuScenes(mot_dataset.MOTDataset):
    ALL_SPLITS = {"train", "val", "test", "train_detect", "train_track",
                  "mini_train", "mini_val"}

    def __init__(self, work_dir: str, det_source: str, seg_source: str, version: str):

        super().__init__(work_dir, det_source, seg_source)
        """ Initialize dataset-level NuScenes object

        :param version: version of NuScenes to use e.g. "v1.0-mini", "v1.0-trainval", "v1.0-test"
        """
        print(f"Parsing NuScenes {version} ...")
        self.nusc = nuscenes.NuScenes(version=version, dataroot=local_variables.NUSCENES_DATA_DIR, verbose=True)
        self.splits: Set[str] = set(s for s in self.ALL_SPLITS if s.split("_")[0] in version)
        self.sequences_by_name: Dict[str, Any] = {
            scene["name"]: scene for scene in self.nusc.scene
        }
        self.splits_to_scene_names: Dict[str, List[str]] = splits.create_splits_scenes()
        print("Done parsing")

        self.version = version
        self.reset()

    def reset(self) -> None:
        self.submission: Dict[str, Dict[str, Any]] = {"meta": {"use_camera": True,
                                                               "use_lidar": True,
                                                               "use_radar": False,
                                                               "use_map": False,
                                                               "use_external": False},
                                                      "results": {}}
        self.detections_3d: Dict[str, List[bbox.Bbox3d]] = {}

    def sequence_names(self, split: str) -> List[str]:  # overrides base method
        self.assert_split_exists(split)
        return self.splits_to_scene_names[split]

    def get_sequence(self, split: str, sequence_name: str) -> sequence.MOTSequenceNuScenes:  # overrides base method
        self.assert_sequence_in_split_exists(split, sequence_name)
        split_dir = os.path.join(self.work_dir, split)
        return sequence.MOTSequenceNuScenes(self.det_source, self.seg_source, split_dir, split,
                                   self.nusc, self.sequences_by_name[sequence_name],
                                   self.submission, self.detections_3d)

    def save_all_mot_results(self, folder_name: str) -> None:  # overrides base method
        reporting.save_to_json_file(self.submission, folder_name, self.version)
