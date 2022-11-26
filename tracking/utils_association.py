import numpy as np

from utils.utils_geometry import (iou_3d_from_corners, box_2d_overlap_union,
                                  tracking_center_distance_2d, tracking_distance_2d_dims, tracking_distance_2d_full)


'''
' 3D bounding box IoU
'''
def iou_bbox_3d_matrix(detections, predictions, detections_dims, predictions_dims):
    return generic_similarity_matrix_two_args(detections, predictions,
                                              detections_dims, predictions_dims, iou_3d_from_corners)

'''
' 2d 空间下中心点的距离
'''
def distance_2d_matrix(centers_0, centers_1):
    return generic_similarity_matrix(centers_0, centers_1, tracking_center_distance_2d)


'''
' x, y, z, w, h, l 六维下的空间距离
'''
def distance_2d_dims_matrix(coords_0, coords_1):
    return generic_similarity_matrix(coords_0, coords_1, tracking_distance_2d_dims)

'''
' EgerMOT中的相似度计算, 六维距离乘以角度距离
'''
def distance_2d_full_matrix(coords_0, coords_1):
    return generic_similarity_matrix(coords_0, coords_1, tracking_distance_2d_full)

'''
' 权重设置
'''
def distance_pos_dims_weight_matrix(centers_0, centers_1, dims_0, dims_1, threshold = float('inf')):
    return generic_similarity_matrix_three_args(centers_0, centers_1, dims_0, dims_1, threshold)



'''
' 2d IOU
'''
def iou_bbox_2d_matrix(det_bboxes, seg_bboxes):
    return generic_similarity_matrix(det_bboxes, seg_bboxes, box_2d_overlap_union)


def generic_similarity_matrix(list_0, list_1, similarity_function):
    matrix = np.zeros((len(list_0), len(list_1)), dtype=np.float32)
    for i, element_0 in enumerate(list_0):
        for j, element_1 in enumerate(list_1):
            matrix[i, j] = similarity_function(element_0, element_1)
    return matrix


def generic_similarity_matrix_two_args(list_0, list_1, attrs_0, attrs_1, similarity_function):
    matrix = np.zeros((len(list_0), len(list_1)), dtype=np.float32)
    for i, element_0 in enumerate(list_0):
        for j, element_1 in enumerate(list_1):
            matrix[i, j] = similarity_function(element_0, element_1, attrs_0[i], attrs_1[j])
    return matrix

def generic_similarity_matrix_three_args(list_0, list_1, attrs_0, attrs_1, threshold):
    m, n = len(list_0), len(list_1)
    helper = np.zeros((m, n, 2), dtype=np.float32)
    matrix = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            helper[i, j, 0] = np.linalg.norm(list_0[i][np.array((0, 1, 2))] - list_1[j][np.array((0, 1, 2))])
            helper[i, j, 1] = np.linalg.norm(attrs_0[i][np.array((0, 1, 2))] - attrs_1[j][np.array((0, 1, 2))])
    
    for i in range(len(list_0)):
        for j in range(len(list_1)):
            if helper[i, j, 1] < threshold:
                matrix[i, j] = helper[i, j, 0]
            else:
                matrix[i, j] = float('inf')
    return matrix