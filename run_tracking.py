import time
import argparse
from inputs import utils
from typing import Iterable, Set, Dict, Any
from dataset_classes import mot_dataset
from dataset_classes.kitti import mot_kitti
from dataset_classes.nuscenes import dataset
from configs import parameters, local_variables


def perform_tracking_full(dataset: mot_kitti.MOTDatasetKITTI, 
                          params: Dict[str, Any], 
                          target_sequences: Set[str], 
                          sequences_to_exclude: Set[str], 
                          print_debug_info=True):
    '''
    主逻辑，指定数据集对象和参数执行跟踪
    Args:
        dataset: 数据集对象
        params: 跟踪过程中超参数
        target_sequences: 目标序列
        sequences_to_exclude: 排除序列
    Returns:
        run_info: 跟踪结果信息
        variant: 参数、方法组合的变量名
    '''

    if len(target_sequences) == 0:
        # 获取数据集序列号列表
        target_sequences = set(dataset.sequence_names(local_variables.SPLIT))
    valid_sequences = target_sequences - sequences_to_exclude
        
    total_frame_count = 0
    total_time = 0
    total_time_tracking = 0
    total_time_fusion = 0
    total_time_reporting = 0

    for sequence_name in sorted(valid_sequences):
        print(f'Starting sequence: {sequence_name}')
        start_time = time.time()
        # 根据序列号创建一个序列对象
        sequence = dataset.get_sequence(local_variables.SPLIT, sequence_name)
        # 设置序列的跟踪对象的轨迹管理参数
        sequence.mot.set_track_manager_params(params)
        # 根据参数生成变量名
        variant = parameters.variant_name_from_params(params)
        # 运行主逻辑
        run_info = sequence.perform_tracking_for_eval(params)
        if "total_time_mot" not in run_info:
            continue

        total_time = time.time() - start_time
        if print_debug_info:
            print(f'Sequence {sequence_name} took {total_time:.2f} sec, {total_time / 60.0 :.2f} min')
            print(f'Matching took {run_info["total_time_matching"]:.2f} sec, {100 * run_info["total_time_matching"] / total_time:.2f}%')
            print(f'Creating took {run_info["total_time_creating"]:.2f} sec, {100 * run_info["total_time_creating"] / total_time:.2f}%')
            print(f'Fusion   took {run_info["total_time_fusion"]:.2f} sec, {100 * run_info["total_time_fusion"] / total_time:.2f}%')
            print(f'Tracking took {run_info["total_time_mot"]:.2f} sec, {100 * run_info["total_time_mot"] / total_time:.2f}%')
            print(f'{run_info["matched_tracks_first_total"]} 1st stage and {run_info["matched_tracks_second_total"]} 2nd stage matches')
            print("\n")

        total_time += total_time
        total_time_fusion += run_info["total_time_fusion"]
        total_time_tracking += run_info["total_time_mot"]
        total_time_reporting += run_info["total_time_reporting"]
        total_frame_count += len(sequence.frame_names)

    if total_frame_count == 0:
        return variant, run_info

    dataset.save_all_mot_results(run_info["mot_3d_file"])

    if not print_debug_info:
        return variant, run_info

    # Overall variant stats
    # Timing
    print(f'Fusion    {total_time_fusion: .2f} sec, {(100 * total_time_fusion / total_time):.2f}%')
    print(f'Tracking  {total_time_tracking: .2f} sec, {(100 * total_time_tracking / total_time):.2f}%')
    print(f'Reporting {total_time_reporting: .2f} sec, {(100 * total_time_reporting / total_time):.2f}%')
    print(f'Tracking-fusion framerate: {total_frame_count / (total_time_fusion + total_time_tracking):.2f} fps')
    print(f'Tracking-only framerate: {total_frame_count / total_time_tracking:.2f} fps')
    print(f'Total framerate: {total_frame_count / total_time:.2f} fps')
    print()

    # Fused instances stats
    total_instances = run_info['instances_both'] + run_info['instances_3d'] + run_info['instances_2d']
    if total_instances > 0:
        print(f"Total instances 3D and 2D: {run_info['instances_both']} " + f"-> {100.0 * run_info['instances_both'] / total_instances:.2f}%")
        print(f"Total instances 3D only  : {run_info['instances_3d']} " + f"-> {100.0 * run_info['instances_3d'] / total_instances:.2f}%")
        print(f"Total instances 2D only  : {run_info['instances_2d']} " + f"-> {100.0 * run_info['instances_2d'] / total_instances:.2f}%")
        print()

    # Matching stats
    print(f"matched_tracks_first_total {run_info['matched_tracks_first_total']}")
    print(f"unmatched_tracks_first_total {run_info['unmatched_tracks_first_total']}")

    print(f"matched_tracks_second_total {run_info['matched_tracks_second_total']}")
    print(f"unmatched_tracks_second_total {run_info['unmatched_tracks_second_total']}")
    print(f"unmatched_dets2d_second_total {run_info['unmatched_dets2d_second_total']}")

    first_matched_percentage = (run_info['matched_tracks_first_total'] / (run_info['unmatched_tracks_first_total'] + run_info['unmatched_tracks_first_total']))
    print(f"percentage of all tracks matched in 1st stage {100.0 * first_matched_percentage:.2f}%")

    second_matched_percentage = (run_info['matched_tracks_second_total'] / run_info['unmatched_tracks_first_total'])
    print(f"percentage of leftover tracks matched in 2nd stage {100.0 * second_matched_percentage:.2f}%")

    second_matched_dets2d_second_percentage = (run_info['matched_tracks_second_total'] / (run_info['unmatched_dets2d_second_total'] + run_info['matched_tracks_second_total']))
    print(f"percentage dets 2D matched in 2nd stage {100.0 * second_matched_dets2d_second_percentage:.2f}%")

    final_unmatched_percentage = (run_info['unmatched_tracks_second_total'] / (run_info['matched_tracks_first_total'] + run_info['unmatched_tracks_first_total']))
    print(f"percentage tracks unmatched after both stages {100.0 * final_unmatched_percentage:.2f}%")

    print(f"\n3D MOT saved in {run_info['mot_3d_file']}", end="\n\n")
    return variant, run_info


def perform_tracking_with_params(dataset: mot_dataset.MOTDataset, 
                                 params: Dict[str, Any], 
                                 target_sequences: Set[str],
                                 sequences_to_exclude: Set[str]):
    '''
    指定数据集对象和参数执行跟踪
    Args:
        dataset: 数据集对象
        params: 跟踪过程中超参数
        target_sequences: 目标序列
        sequences_to_exclude: 排除序列
    Returns:
        run_info: 跟踪结果信息
    '''
    start_time = time.time()
    variant, run_info = perform_tracking_full(dataset, params,
                                              target_sequences=target_sequences,
                                              sequences_to_exclude=sequences_to_exclude)
    print(f'Variant {variant} took {(time.time() - start_time) / 60.0:.2f} mins')
    return run_info


def run_on_nuscenes():
    VERSION = "v1.0-trainval"
    # 创建nuscenes数据集对象
    nuscenes_dataset = dataset.MOTDatasetNuScenes(
                       work_dir=local_variables.NUSCENES_WORK_DIR,
                       det_source=utils.CENTER_POINT,
                       seg_source=utils.MMDETECTION_CASCADE_NUIMAGES,
                       version=VERSION)

    # if want to run on specific sequences only, add their str names here
    target_sequences: Set[str] = set()

    # if want to exclude specific sequences, add their str names here
    sequences_to_exclude: Set[str] = set()

    perform_tracking_with_params(nuscenes_dataset, 
                                 parameters.NUSCENES_BEST_PARAMS, 
                                 target_sequences, sequences_to_exclude)
    nuscenes_dataset.reset()


def run_on_kitti():
    # 创建kitti数据集对象
    kitti_dataset = mot_kitti.MOTDatasetKITTI(
                    work_dir=local_variables.KITTI_WORK_DIR,
                    detections_dir_3d=utils.POINTGNN_T3,
                    detections_dir_2d=utils.TRACKING_BEST)

    # 只在特定的序列上运行, 添加其序列名, 形如：0001、0020
    target_sequences: Set[str] = set()

    # 排除特定的序列
    sequences_to_exclude: Set[str] = set()

    # 执行跟踪
    perform_tracking_with_params(kitti_dataset, parameters.KITTI_3D_PARAMS, 
                                 target_sequences, sequences_to_exclude)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parsing command line parameters")
    parser.add_argument("--dataset", "-d", type=str, default="kitti", 
                        help="name of data set, kitti or nuscenes")
    
    args = parser.parse_args()
    if args.dataset.lower() == "kitti":
        run_on_kitti()
    elif args.dataset.lower() == "nuscenes":
        run_on_nuscenes()
    else:
        print("error, please input correct format.")