from __future__ import annotations
import os
import glob
import imageio
import numpy as np
import matplotlib.image as mpimg
from typing import Optional, Dict, List, Set, Iterable, Sequence, IO, Tuple, Callable

from objects import fused_instance
from configs import local_variables
from utils import io, utils_geometry
from transform import kitti, transformation
from dataset_classes.kitti import reporting
from inputs import loading, utils, detections_2d, detection_2d, bbox
from dataset_classes import mot_dataset, mot_sequence, mot_frame








# noinspection SpellCheckingInspection
class MOTFrameKITTI(mot_frame.MOTFrame):
    def __init__(self, sequence, name):
        super().__init__(sequence, name)
        self.image_path = os.path.join(sequence.image_path, f'{name}.png')
        self.pcd_path = os.path.join(sequence.pcd_path, f'{name}.bin')
        self._ego_transform: Optional[np.ndarray] = None
        self._angle_around_vertical: Optional[float] = None

    def get_image_original(self, cam: str = "image_02"):
        '''返回此帧图像, 并记录图像的尺寸 - matplotlib.image
        '''
        image = mpimg.imread(self.image_path % cam)
        # need to remember actual image size
        self.sequence.img_shape_per_cam[cam] = image.shape[:2]
        return image

    def get_image_original_uint8(self, cam: str = "image_02"):
        '''返回此帧图像, 并记录图像的尺寸 - imageio
        '''
        image = imageio.imread(self.image_path % cam)
        # need to remember actual image size
        self.sequence.img_shape_per_cam[cam] = image.shape[:2]
        return image
    
    def load_raw_pcd(self):
        pcd = np.fromfile(self.pcd_path, dtype=np.float32)
        return pcd.reshape((-1, 4))[:, :3]

    @property
    def points_world(self):
        points = (self.ego_transform @ transformation.to_homogeneous(self.points_rect).T).T
        return points[:, :-1]

    @property
    def center_world_point(self) -> np.ndarray:
        return np.zeros((1, 3), dtype=float)  # 3D points are already centered

    ##########################################################
    # Ego motion

    @property
    def ego_transform(self):  # apply left-side to go from ego to world
        if self._ego_transform is None:
            self._ego_transform = self.sequence.ego_transform_for_frame(self.name)
        return self._ego_transform

    @property
    def angle_around_vertical(self):
        if self._angle_around_vertical is None:
            self._angle_around_vertical = utils_geometry.angles_from_rotation_matrix(self.ego_transform[:3, :3])[1]
        return self._angle_around_vertical

    @property
    def images_for_vo(self):
        images = [mpimg.imread(self.image_path % 'image_02'), mpimg.imread(self.image_path % 'image_03')]
        images = [np.mean(x, -1) for x in images]
        return images

    def transform_instances_to_world_frame(self) -> Tuple[np.ndarray, float]:
        for fused_object in self.fused_instances:
            fused_object.transform(self.ego_transform, self.angle_around_vertical)
        return self.ego_transform, self.angle_around_vertical

    @property
    def bboxes_3d_ego(self) -> List[bbox.Bbox3d]:
        return self.bboxes_3d

    @property
    def bboxes_3d_world(self):
        bboxes = self.bboxes_3d.copy()
        for bbox in bboxes:
            bbox.transform(self.ego_transform, self.angle_around_vertical)
        return bboxes

    def bbox_3d_annotations(self, world: bool = False) -> List[bbox.Bbox3d]:  # List[Box]
        assert not world  # kitti only provides annotations in the ego frame
        return self.sequence.bbox_3d_annotations[self.name]

    def bbox_2d_annotation_projections(self) -> Dict[str, List[detection_2d.Detection2D]]:
        cam = "image_02"
        return {cam: [detection_2d.Detection2D(bbox_3d.bbox_2d_in_cam(cam), cam, bbox_3d.confidence,
                                  bbox_3d.seg_class_id) for bbox_3d in self.bbox_3d_annotations(False)]}


class MOTSequenceKITTI(mot_sequence.MOTSequence):
    def __init__(self, detections_3d: str, detections_2d: str, split_dir: str, split: str,
                 name: str, frame_names: List[str]):
        super().__init__(detections_3d, detections_2d, split_dir, name, frame_names)
        self.data_dir = os.path.join(local_variables.KITTI_DATA_DIR, split)
        self.image_path = os.path.join(self.data_dir, "%s", self.name)
        self.pcd_path = os.path.join(self.data_dir, "velodyne", self.name)
        ego_motion_folder = os.path.join(split_dir, "ego_motion")
        io.makedirs_if_new(ego_motion_folder)
        self.ego_motion_filepath = os.path.join(ego_motion_folder, self.name + ".npy")

        self._transformation: Optional[kitti.TransformationKitti] = None
        self.mot.transformation = self.transformation

        self.transform_accumulated = None
        self.ego_motion_transforms = np.ones(shape=(len(self.frame_names), 4, 4), dtype=np.float)
        self.has_full_ego_motion_transforms_loaded = False

        fusion_name = 'det_%s_%s_seg_%s_%s_iou_%s_%s'
        self.instance_fusion_bbox_dir = os.path.join(
            self.work_split_input_dir, 'instance_fusion_bbox', fusion_name, self.name)

        self._bbox_3d_annotations: Dict[str, List[bbox.Bbox3d]] = {}

    @property
    def transformation(self) -> kitti.TransformationKitti:
        if self._transformation is None:
            self._transformation = kitti.TransformationKitti(self.data_dir, self.name)
        return self._transformation

    def get_frame(self, frame_name: str):
        '''创建并返回帧对象
        Args:
            frame_name: 帧名, 如 000001
        Returns:
            frame: 帧对象
        '''
        frame = MOTFrameKITTI(self, frame_name)
        
        # 通过首次加载一帧图像来确定图像尺寸
        if not self.img_shape_per_cam: 
            for cam in self.cameras:
                # 记录图像尺寸
                frame.get_image_original(cam)
            self.mot.img_shape_per_cam = self.img_shape_per_cam
        return frame

    def load_detections_3d(self) -> Dict[str, List[bbox.Bbox3d]]:
        bboxes_3d_all = loading.load_detections_3d(self.detections_3d, self.name)
        return {str(frame_i).zfill(6): bboxes_3d
                for frame_i, bboxes_3d in enumerate(bboxes_3d_all)}

    def load_detections_2d(self) -> Dict[str, Dict[str, List[detection_2d.Detection2D]]]:
        if self.detections_2d == utils.MMDETECTION_CASCADE_NUIMAGES:
            return loading.load_detections_2d_kitti_new(self.detections_2d, self.name)

        """ Load and construct 2D Detections for this sequence, sorted by score ascending """
        bboxes_all, scores_all, reids_all, classes_all, masks_all = loading.load_detections_2d_kitti(
            self.detections_2d, self.name)

        return {str(frame_i).zfill(6): self.construct_detections_2d(bboxes, scores, classes, masks, reids)
                for frame_i, (bboxes, scores, classes, masks, reids)
                in enumerate(zip(bboxes_all, scores_all, classes_all, masks_all, reids_all))}

    def construct_detections_2d(self, bboxes, scores, classes,
                                masks, reids) -> Dict[str, List[detection_2d.Detection2D]]:
        dets = [detection_2d.Detection2D(bbox.Bbox2d(*box), self.cameras[0], score, seg_class_id, mask=mask, reid=reid) for
                box, score, seg_class_id, mask, reid
                in zip(bboxes, scores, classes, masks, reids)]
        dets.sort(key=lambda x: x.score)  # sort detections by ascending score
        return {self.cameras[0]: dets}

    @property
    def bbox_3d_annotations(self) -> Dict[str, List[bbox.Bbox3d]]:
        if not self._bbox_3d_annotations:
            self._bbox_3d_annotations = loading.load_annotations_kitti(self.name)
        return self._bbox_3d_annotations

    ##########################################################
    # Ego motion

    def ego_transform_for_frame(self, frame_name: str) -> np.ndarray:
        """
        :param frame_name: for which frame to get the transformation, only works consecutively
        :return: full transform 4x4 from the current frame/3D pose, in the first frame's coordinate system
        """
        assert self.has_full_ego_motion_transforms_loaded
        frame_int = int(frame_name)
        return self.ego_motion_transforms[frame_int]

    def save_ego_motion_transforms_if_new(self) -> None:
        '''保存自我运动(ego motion)的npy文件
        '''
        if not self.has_full_ego_motion_transforms_loaded:
            with open(self.ego_motion_filepath, 'wb') as np_file:
                np.save(np_file, self.ego_motion_transforms)

    def load_ego_motion_transforms(self) -> None:
        '''加载自我运动(ego motion)文件
        更改属性: ego_motion_transforms, has_full_ego_motion_transforms_loaded
        '''
        assert os.path.isfile(self.ego_motion_filepath), "Missing ego motion files"
        with open(self.ego_motion_filepath, 'rb') as np_file:
            self.ego_motion_transforms = np.load(np_file)
        self.has_full_ego_motion_transforms_loaded = True

    def report_mot_results(self, frame_name: str, predicted_instances: Iterable[fused_instance.FusedInstance],
                           mot_3d_file: IO,
                           mot_2d_from_3d_only_file: Optional[IO]) -> None:
        '''报道(保存)多目标跟踪结果
        Args:
            frame_name:             帧序号, 如 000001
            predicted_instances:    预测实例对象
            mot_3d_file:            三维跟踪结果文件的描述符 txt
            mot_2d_from_3d_only_file:   二维跟踪结果文件的描述符 txt
        '''
        reporting.write_to_mot_file(frame_name, predicted_instances, mot_3d_file, mot_2d_from_3d_only_file)

    def save_mot_results(self, mot_3d_file: IO, mot_2d_from_3d_file: Optional[IO]) -> None:
        '''保存跟踪结果, 即关闭文件描述符
        Args:
            mot_3d_file:            三维多目标跟踪结果文件描述符
            mot_2d_from_3d_file:    二维多目标跟踪结果文件描述符
        '''
        io.close_files((mot_3d_file, mot_2d_from_3d_file))

    ##########################################################
    # Cameras

    @property
    def camera_params(self): return MOTDatasetKITTI.CAMERA_PARAMS

    @property
    def cameras(self) -> List[str]:
        return ["image_02"]  # "image_03"

    @property
    def camera_default(self) -> str:
        return "image_02"

    @property
    def classes_to_track(self) -> List[int]:
        return [detections_2d.CAR_CLASS, detections_2d.PED_CLASS]


class MOTDatasetKITTI(mot_dataset.MOTDataset):
    FOCAL = 721.537700
    CU = 609.559300
    CV = 172.854000
    BASELINE = 0.532719
    CAMERA_PARAMS = [FOCAL, CU, CV, BASELINE]

    def __init__(self, work_dir, detections_3d: str, detections_2d: str):
        super().__init__(work_dir, detections_3d, detections_2d)
        self.splits: Set[str] = {"training", "testing"}
        self.split_sequence_frame_names_map: Dict[str, Dict[str, List[str]]] = {sp: {} for sp in self.splits}

        for split in self.splits:
            seq_dir = os.path.join(local_variables.KITTI_DATA_DIR, split, 'image_02')
            if not os.path.isdir(seq_dir):
                raise NotADirectoryError(seq_dir)

            # Parse sequences
            for sequence in sorted(os.listdir(seq_dir)):
                img_dir = os.path.join(seq_dir, sequence)
                if os.path.isdir(img_dir):
                    images = glob.glob(os.path.join(img_dir, '*.png'))
                    self.split_sequence_frame_names_map[split][sequence] = [os.path.splitext(os.path.basename(image))[0]
                                                                            for image in sorted(images)]

    def sequence_names(self, split: str) -> List[str]:
        self.assert_split_exists(split)
        return list(self.split_sequence_frame_names_map[split].keys())

    def get_sequence(self, split: str, sequence_name: str) -> MOTSequenceKITTI:
        '''根据分割符和序列号创建并返回KITTI序列对象
        args:
            split: 分隔符, 有training和testing
            sequence_name: 序列号, 如 0001
        Returns:
            KITTI序列对象
        '''
        self.assert_sequence_in_split_exists(split, sequence_name)
        split_dir = os.path.join(self.work_dir, split)
        return MOTSequenceKITTI(self.detections_3d, self.detections_2d, split_dir, split, sequence_name,
                                self.split_sequence_frame_names_map[split][sequence_name])

    def save_all_mot_results(self, folder_name: str) -> None:
        """ KITTI saves results per-sequence, so this method does not apply here """
        pass
