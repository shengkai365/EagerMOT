import copy
import numpy as np
from typing import List, Optional, Dict, Any, Mapping

from inputs import bbox
from utils import utils_geometry
from tracking import utils_tracks
from objects import fused_instance
from transform import transformation


class Track(bbox.ProjectsToCam):
    '''轨迹类
    Attributes:
        instance:               融合实例
        is_angular:             是否在kf运动模型中加入角速度
        id:                     轨迹id, 从0开始. 每创建一个轨迹对象增1
        age_total:              轨迹出生开始持续的帧数
        hits:                   轨迹对象生命周期内总匹配次数(包含第一帧检测)
        time_since_update:      轨迹从上次更新到现在经历了多少帧
        time_since_3d_update:   轨迹从上次3d更新到现在经历了多少帧
        time_since_2d_update:   轨迹从上次2d更新到现在经历了多少帧
        kf_3d:                  卡尔曼滤波器对象
        obs_angle:
        confidence:
        predicted_translation:
        _predicted_bbox_2d_in_cam:
    '''
    count = 0
    def __init__(self, instance: fused_instance.FusedInstance, is_angular: bool):
        """
        Initializes a tracker using initial bounding box.
        """
        self.instance = instance
        self.is_angular = is_angular
        self.id = Track.count
        Track.count += 1

        self.age_total = 1
        self.hits = 1  # number of total hits including the first detection
        self.time_since_update = 0
        self.time_since_3d_update = 0 if instance.bbox3d is not None else 10
        self.time_since_2d_update = 0 if instance.detection_2d is not None else 10

        self.mask_score_history: List[float] = []

        self.kf_3d = None
        self.obs_angle: Optional[float] = None
        self.confidence: Optional[float] = None
        if instance.bbox3d is not None:
            self.init_motion_model(instance.bbox3d)

        self.predicted_translation = None

        self._predicted_bbox_2d_in_cam: Dict[str, Optional[bbox.Bbox2d]] = {}

    def init_motion_model(self, bbox3d):
        assert bbox3d is not None
        self.kf_3d = utils_tracks.default_kf_3d(self.is_angular)
        self.kf_3d.x[:7] = bbox3d.kf_coordinates.reshape(7, 1)
        self._set_info(bbox3d)

    def _set_info(self, bbox3d):
        self.obs_angle = bbox3d.obs_angle
        self.confidence = bbox3d.confidence
    @property
    def has_motion_model(self):
        return self.kf_3d is not None

    def predict_motion(self):
        """ Advances the state vector and returns the predicted bounding box estimate. """
        assert self.has_motion_model
        self.instance.bbox3d.clear_2d()
        old_x = self.kf_3d.x.copy()
        self.kf_3d.predict()
        # to move point cloud according to KF correction in case it will not later be updated (see below)
        self.predicted_translation = self.kf_3d.x[:3] - old_x[:3]
        return self.kf_3d.x.flatten().reshape(-1,)  # shape (10,)

    def update_with_match(self, matched_instance: fused_instance.FusedInstance):
        if matched_instance.bbox3d is not None:
            self._update_3d_info(matched_instance)
        if matched_instance.detection_2d is not None:
            self._update_2d_info(matched_instance)
        self.time_since_update = 0
        self.hits += 1

    def _update_3d_info(self, matched_instance: fused_instance.FusedInstance):
        """ Updates the state vector with observed bbox. """
        assert matched_instance.bbox3d is not None
        self.time_since_3d_update = 0

        if self.has_motion_model:
            assert self.kf_3d is not None
            # new angle needs to be corrected to be the closest match to the current angle
            new_angle = matched_instance.bbox3d.kf_coordinates[3]
            new_angle, angle_diff = utils_tracks.correct_new_angle_and_diff(self.kf_3d.x[3], new_angle)
            assert angle_diff <= np.pi / 2, f"angle_diff {angle_diff}"
            matched_instance.bbox3d.kf_coordinates[3] = new_angle
            self.kf_3d.update(matched_instance.bbox3d.kf_coordinates)
        else:
            self.init_motion_model(matched_instance.bbox3d)

        self._set_info(matched_instance.bbox3d)
        self.instance = matched_instance

    def _update_2d_info(self, instance_from_mask: fused_instance.FusedInstance):
        # set mask, bbox_2d, etc. but keep 3D fields
        self.instance.set_with_instance_from_mask(instance_from_mask)
        self.time_since_2d_update = 0

    def reset_for_new_frame(self):
        self.age_total += 1
        self.time_since_update += 1
        self.time_since_3d_update += 1
        self.time_since_2d_update += 1
        self.instance.reset_seg(keep_matching_info=True)
        self._predicted_bbox_2d_in_cam = {}

    @property
    def current_bbox_3d_coordinates(self):
        assert self.has_motion_model
        return self.kf_3d.x[:7].reshape(7,)

    def current_bbox_3d(self, ego_transform, angle_around_y) -> Optional[bbox.Bbox3d]:
        """ Returns the current bounding box estimate. """
        if not self.has_motion_model:
            return None

        bbox = bbox.Bbox3d.from_pointrcnn(self.current_bbox_3d_coordinates.copy())
        if ego_transform is not None and angle_around_y is not None:
            bbox.inverse_transform(ego_transform, angle_around_y)
        bbox.obs_angle = self.obs_angle
        bbox.confidence = self.confidence
        return bbox

    def current_instance(self, ego_transform, angle_around_y, min_hits=1) -> fused_instance.FusedInstance:
        if ego_transform is None or angle_around_y is None:
            return copy.deepcopy(self.instance)
        local_frame_instance = copy.deepcopy(self.instance)
        local_frame_instance.inverse_transform(ego_transform, angle_around_y)
        return local_frame_instance

    def bbox_2d_in_cam(self, cam: str) -> Optional[bbox.Bbox2d]:
        return self._predicted_bbox_2d_in_cam[cam]

    def predicted_bbox_2d_in_cam(self, ego_transform, angle_around_y,
                                 transformation: transformation.Transformation, img_shape_per_cam: Mapping[str, Any],
                                 cam: str, frame_data: Mapping[str, Any]) -> Optional[bbox.Bbox2d]:
        self._predicted_bbox_2d_in_cam[cam] = self.instance.bbox_2d_best(cam)

        bbox_3d = self.current_bbox_3d(ego_transform, angle_around_y)
        if bbox_3d is not None:
            bbox_2d = utils_geometry.project_bbox_3d_to_2d(bbox_3d, transformation, img_shape_per_cam, cam, frame_data)
            if bbox_2d is not None:
                self._predicted_bbox_2d_in_cam[cam] = bbox.Bbox2d(*bbox_2d)
        return self._predicted_bbox_2d_in_cam[cam]

    @property
    def class_id(self):
        return self.instance.class_id
