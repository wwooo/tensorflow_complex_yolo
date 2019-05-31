"""
    Util scripts for building features, fetching ground truths, computing IoU, etc.
"""
from __future__ import division

import numpy as np
import cv2
import math

# classes
class_list = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

bc = {}
bc['minX'] = 0;
bc['maxX'] = 80;
bc['minY'] = -40;
bc['maxY'] = 40
bc['minZ'] = -2;
bc['maxZ'] = 1.25
##########################################


def read_anchors_file(file_path):
    anchors = []
    with open(file_path, 'r') as file:
        for line in file.read().splitlines():
            anchors.append(list(map(float,line.split())))

    return np.array(anchors)


def read_labels(filepath):
    classes, names, colors = [], [], []
    with open(filepath,'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            cls, name, color = line.split()
            classes.append(int(cls))
            names.append(name)
            colors.append(eval(color))
    return classes, names, colors

def cal_angle(im, re):
    if im < 0 and re < 0:
        return -np.pi + np.arctan(im / re)
    elif im > 0 and re < 0:
        return np.pi + np.arctan(im / re)
    else:
        return np.arctan(im / re)


def _draw_rotate_rec(img, cy, cx, w, h, angle, color):
    left = int(cy - w / 2)
    top = int(cx - h / 2)
    right = int(cx + h / 2)
    bottom = int(cy + h / 2)
    ro = np.sqrt(pow(left - cy, 2) + pow(top - cx, 2))
    a1 = np.arctan((w / 2) / (h / 2))
    a2 = -np.arctan((w / 2) / (h / 2))
    a3 = -np.pi + a1
    a4 = np.pi - a1
    rotated_p1_y = cy + int(ro * np.sin(angle + a1))
    rotated_p1_x = cx + int(ro * np.cos(angle + a1))
    rotated_p2_y = cy + int(ro * np.sin(angle + a2))
    rotated_p2_x = cx + int(ro * np.cos(angle + a2))
    rotated_p3_y = cy + int(ro * np.sin(angle + a3))
    rotated_p3_x = cx + int(ro * np.cos(angle + a3))
    rotated_p4_y = cy + int(ro * np.sin(angle + a4))
    rotated_p4_x = cx + int(ro * np.cos(angle + a4))
    center_p1p2y = int((rotated_p1_y + rotated_p2_y) * 0.5)
    center_p1p2x = int((rotated_p1_x + rotated_p2_x) * 0.5)
    cv2.line(img, (rotated_p1_y, rotated_p1_x), (rotated_p2_y, rotated_p2_x), color, 1)
    cv2.line(img, (rotated_p2_y, rotated_p2_x), (rotated_p3_y, rotated_p3_x), color, 1)
    cv2.line(img, (rotated_p3_y, rotated_p3_x), (rotated_p4_y, rotated_p4_x), color, 1)
    cv2.line(img, (rotated_p4_y, rotated_p4_x), (rotated_p1_y, rotated_p1_x), color, 1)
    cv2.line(img, (center_p1p2y, center_p1p2x), (cy, cx), color, 1)


def get_corner_gtbox(box):
    bx = box[0]
    by = box[1]
    bw = box[2]
    bl = box[3]
    top = int((by - bl / 2.0))
    left = int((bx - bw / 2.0))
    right = int((bx + bw / 2.0))
    bottom = int((by + bl / 2.0))
    return left, top, right, bottom

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    # import pdb; pdb.set_trace()
    return carea/uarea

def iou(r1, r2):
    intersect_w = np.maximum(np.minimum(r1[0] + r1[2], r2[0] + r2[2]) - np.maximum(r1[0], r2[0]), 0)
    intersect_h = np.maximum(np.minimum(r1[1] + r1[3], r2[1] + r2[3]) - np.maximum(r1[1], r2[1]), 0)
    area_r1 = r1[2] * r1[3]
    area_r2 = r2[2] * r2[3]
    intersect = intersect_w * intersect_h
    union = area_r1 + area_r2 - intersect

    return intersect / union

def softmax(x):
    e_x = np.exp(x)
    return e_x/np.sum(e_x)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def non_max_supression(classes, locations, prob_th, iou_th):
    classes = np.transpose(classes)
    indxs = np.argsort(-classes, axis=1)

    for i in range(classes.shape[0]):
        classes[i] = classes[i][indxs[i]]

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):
            if roi_prob < prob_th:
                classes[class_idx][roi_idx] = 0

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):

            if roi_prob == 0:
                continue
            roi = locations[indxs[class_idx][roi_idx]][0:4]


            for roi_ref_idx, roi_ref_prob in enumerate(class_vec):

                if roi_ref_prob == 0 or roi_ref_idx <= roi_idx:
                    continue

                roi_ref = locations[indxs[class_idx][roi_ref_idx]][0:4]

                if bbox_iou(roi, roi_ref, False) > iou_th:
                    classes[class_idx][roi_ref_idx] = 0

    return classes, indxs

def filter_bbox(classes, rois, indxs):
    all_bboxs = []
    for class_idx, c in enumerate(classes):
        for loc_idx, class_prob in enumerate(c):
            if class_prob > 0:
                x = int(rois[indxs[class_idx][loc_idx]][0])
                y = int(rois[indxs[class_idx][loc_idx]][1])
                w = int(rois[indxs[class_idx][loc_idx]][2])
                h = int(rois[indxs[class_idx][loc_idx]][3])
                re = rois[indxs[class_idx][loc_idx]][4]
                im = rois[indxs[class_idx][loc_idx]][5]
                all_bboxs.append([class_idx, x, y, w, h, re, im, class_prob])
    return all_bboxs





def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX'];
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'];
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'];
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
                PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]
    PointCloud[:, 2] = PointCloud[:, 2] + 2
    return PointCloud


def makeBVFeature(PointCloud_):
    # 1024 x 1024 x 3
    Height = 1024 + 1
    Width = 1024 + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / 60.0 * 768))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / 40.0 * 512) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((Height, Width, 3))
    RGB_Map[:, :, 0] = densityMap         # r_map
    RGB_Map[:, :, 1] = heightMap / 3.26   # g_map
    RGB_Map[:, :, 2] = intensityMap       # b_map

    save = np.zeros((768, 1024, 3))
    save = RGB_Map[0:768, 0:1024, :]
    return save


def get_target(label_file, Tr):
    target = np.zeros([50, 6], dtype=np.float32)

    with open(label_file, 'r') as f:
        lines = f.readlines()
    num_obj = len(lines)
    index = 0
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        # print(obj)
        if obj_class in class_list:

            t_lidar, box3d_corner, rz = box3d_cam_to_velo(obj[8:], Tr)  # get target  3D object location x,y
            location_x = t_lidar[0][0]
            location_y = t_lidar[0][1]

            if (location_x > 0) & (location_x < 60) & (location_y > -40) & (location_y < 40):
                # print(obj_class)
                target[index][2] = t_lidar[0][0] / 60.0  # make sure target inside the covering area (0,1)
                target[index][1] = (t_lidar[0][1] + 40) / 80.0  ## we should put this in [0,1] ,so divide max_size  80 m

                obj_width = obj[9].strip()
                obj_length = obj[10].strip()
                target[index][3] = float(obj_width) / 80.0
                target[index][4] = float(obj_length) / 60.0  # get target width ,length
                target[index][5] = rz

                for i in range(len(class_list)):
                    if obj_class == class_list[i]:  # get target class
                        target[index][0] = i
                index = index + 1

    return target


def box3d_cam_to_velo(box3d, Tr):
    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle

        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return t_lidar, box3d_corner.astype(np.float32), rz


def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


# anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]




