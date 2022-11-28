import os
import time
import abc
import collections
from typing import List, Iterable, Mapping, Dict, Any, Optional, IO

from inputs import bbox
from utils import io
from dataset_classes import mot_frame
from configs import parameters
from tracking import tracking_manager
from inputs import detection_2d
from objects import fused_instance
from transform import transformation



class MOTSequence(abc.ABC):
    """多目标跟踪序列类
    Attributes:
        detections_3d:      三维检测器名, 如 pointgnn_t3
        detections_2d:      二维检测器名, 如 rrc_trackrcnn
        split_dir:          工作目录名和分割目录名组成的路径前缀, 
                            保存跟踪结果: 如 xxx/kitti_work_dir/training
        name:               序列名, 如 0001、0020
        frame_names:        序列的帧列表, 如 [000000, 000001, ...]
        img_shape_per_cam:  每个摄像机的图像尺寸, 用于3D->2D的投影
        dets_3d_per_frame:  序列的三维检测, {frame_name: [bboxes_3d]}
        dets_2d_multicam_per_frame: 序列的二维检测, {frame_name: {cam_name: [bboxes_3d]}}
        mot:                序列的轨迹关联对象
        tracking_res_dir:   跟踪结果文件前缀名
    """

    def __init__(self, detections_3d: str, detections_2d: str, split_dir: str, name: str, frame_names: Iterable[str]):
        self.detections_3d = detections_3d
        self.detections_2d = detections_2d
        self.split_dir = split_dir
        self.name = name
        self.frame_names = frame_names

        # Image size for each camera - needed for 3D->2D projections. The dict is set in dataset-specific classes
        self.img_shape_per_cam: Dict[str, Any] = {}

        # Detections 3D {frame_name: [bboxes_3d]}
        self.dets_3d_per_frame: Dict[str, List[bbox.Bbox3d]] = {}

        # Detections 2D {frame_name: {cam_name: [bboxes_3d]}}
        self.dets_2d_multicam_per_frame: Dict[str, Dict[str, List[detection_2d.Detection2D]]] = {}

        # need to set its Transformation object and img_shape_per_cam in subclasses
        self.mot = tracking_manager.TrackManager(self.cameras, self.classes_to_track)

        det_detections_2d_folder_name = f'{self.detections_3d}_{self.detections_2d}'
        self.work_split_input_dir = os.path.join(self.split_dir, det_detections_2d_folder_name)
        self.tracking_res_dir = os.path.join(self.work_split_input_dir, 'tracking')

    ##########################################################
    # Evaluation

    def perform_tracking_for_eval(self, params: Mapping[str, Any]) -> Dict[str, Any]:


        folder_identifier = 'cleaning_0'  # 为代码相关的变化添加一个简单的标识符
        
        # 根据参数设置获得跟踪结果文件目录名
        results_folder_name_3d = self.get_results_folder_name(params, folder_identifier, "3d")
        results_folder_name_2d = self.get_results_folder_name(params, folder_identifier, "2d_projected_3d")
        
        # 根据目录名和序列号创建并获取可写文件描述符
        mot_3d_file = io.create_writable_file_if_new(results_folder_name_3d, self.name)
        mot_2d_from_3d_file = io.create_writable_file_if_new(results_folder_name_2d, self.name)

        # 返回值, 包含目标跟踪执行信息
        run_info: Dict[str, Any] = collections.defaultdict(int)

        # txt结果文件之前存在时直接返回
        if mot_3d_file is None:
            print(f'Sequence {self.name} already has results. Skipped')
            print('=====================================================================================')
            return run_info

        # run_info 加入目录名
        run_info["mot_3d_file"] = mot_3d_file.name.split(self.name)[0]
        run_info["mot_2d_from_3d_file"] = mot_2d_from_3d_file.name.split(self.name)[0]

        # 加载自我运动(ego motion)文件
        self.load_ego_motion_transforms()

        print(f"Sequence {self.name} has {len(self.frame_names)} frames. Start process...")
        for frame_name in self.frame_names:
            # 根据帧名获得帧对象
            frame = self.get_frame(frame_name)

            # 执行跟踪
            predicted_instances = frame.perform_tracking(params, run_info)

            start_reporting = time.time()
            # 
            self.report_mot_results(frame.name, predicted_instances, mot_3d_file, mot_2d_from_3d_file)
            run_info["total_time_reporting"] += time.time() - start_reporting

        self.save_mot_results(mot_3d_file, mot_2d_from_3d_file)
        self.save_ego_motion_transforms_if_new()
        return run_info

    def get_results_folder_name(self, params: Mapping[str, Any], folder_identifier: str, suffix: str):
        '''根据参数设置获得跟踪结果目录名
        Args:
            params: 参数字典
            folder_identifier: 文件临时标识
            suffix: 维度信息, 如 3d、2d_projected_3d
        Returns:
            跟踪结果目录名
        '''
        folder_suffix_full = f"{parameters.variant_name_from_params(params)}_{folder_identifier}_{suffix}"
        return f"{self.tracking_res_dir}_{folder_suffix_full}"

    ##########################################################
    # Lazy getters for frame-specific data

    def get_segmentations_for_frame(self, frame_name: str) -> Dict[str, List[detection_2d.Detection2D]]:
        """ Return a dict of detection_2d.Detection2D for each camera for the requested frame"""
        if not self.dets_2d_multicam_per_frame:
            self.dets_2d_multicam_per_frame = self.load_detections_2d()
        return self.dets_2d_multicam_per_frame.get(frame_name, collections.defaultdict(list))

    def get_bboxes_for_frame(self, frame_name: str) -> List[bbox.Bbox3d]:
        """ Return a list of bbox.Bbox3d for the requested frame"""
        if not self.dets_3d_per_frame:
            self.dets_3d_per_frame = self.load_detections_3d()
        return self.dets_3d_per_frame.get(frame_name, [])

    ##########################################################
    # Required methods and fields that need to be overridden by subclasses
    # This sadly results in some extra code, but is the best way to ensure compile-time errors

    @abc.abstractmethod
    def load_ego_motion_transforms(self) -> None: pass

    @abc.abstractmethod
    def save_ego_motion_transforms_if_new(self) -> None: pass

    @abc.abstractmethod
    def load_detections_3d(self) -> Dict[str, List[bbox.Bbox3d]]: pass

    @abc.abstractmethod
    def load_detections_2d(self) -> Dict[str, Dict[str, List[detection_2d.Detection2D]]]: pass

    @abc.abstractmethod
    def get_frame(self, frame_name: str) -> mot_frame.MOTFrame: pass

    @property
    @abc.abstractmethod
    def transformation(self) -> transformation.Transformation: pass

    @property
    @abc.abstractmethod
    def cameras(self) -> List[str]: pass

    @property
    @abc.abstractmethod
    def camera_default(self) -> str: pass

    @property
    @abc.abstractmethod
    def classes_to_track(self) -> List[int]: pass

    @abc.abstractmethod
    def report_mot_results(self, frame_name: str, predicted_instances: Iterable[fused_instance.FusedInstance],
                           mot_3d_file: IO,
                           mot_2d_from_3d_only_file: Optional[IO]) -> None:
        pass

    @abc.abstractmethod
    def save_mot_results(self, mot_3d_file: IO,
                         mot_2d_from_3d_file: Optional[IO]) -> None:
        pass
