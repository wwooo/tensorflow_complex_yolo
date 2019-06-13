# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import cv2

# classes
class_list = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    'Misc'
]
class_dict = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7
}


def read_label_from_txt(label_path):
    """
    Read label from txt file.
    label: class, cx, cy, w, l, rz(rotation angle)
    """
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.readlines()
        for label in labels:
            if not label:
                continue
            label = label.strip().split(' ')
            label[0] = class_dict[label[0]]
            bounding_box.append(label)
    if bounding_box:
        return np.array(bounding_box, dtype=np.float32)
    else:
        return None


def read_anchors_from_file(file_path):
    """
    Read  anchors from the configuration file
    """
    anchors = []
    with open(file_path, 'r') as file:
        for line in file.read().splitlines():
            anchors.append(list(map(float, line.split())))
    return np.array(anchors)


def read_class_flag(file_path):
    """
    Read class flags for visualization
    """
    classes, names, colors = [], [], []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            cls, name, color = line.split()
            classes.append(int(cls))
            names.append(name)
            colors.append(eval(color))
    return classes, names, colors


def calculate_angle(im, re):
    """
    param: im(float): imaginary parts of the plural
    param: re(float): real parts of the plural
    return: The angle at which the objects rotate
    around the Z axis in the velodyne coordinate system
    """
    if re > 0:
        return np.arctan(im / re)
    elif im < 0:
        return -np.pi + np.arctan(im / re)
    else:
        return np.pi + np.arctan(im / re)


def draw_rotated_box(img, cy, cx, w, h, angle, color):
    """
    param: img(array): RGB image
    param: cy(int, float):  Here cy is cx in the image coordinate system
    param: cx(int, float):  Here cx is cy in the image coordinate system
    param: w(int, float):   box's width
    param: h(int, float):   box's height
    param: angle(float): rz
    param: color(tuple, list): the color of box, (R, G, B)
    """
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
    cv2.line(img, (rotated_p1_y, rotated_p1_x), (rotated_p2_y, rotated_p2_x),
             color, 1)
    cv2.line(img, (rotated_p2_y, rotated_p2_x), (rotated_p3_y, rotated_p3_x),
             color, 1)
    cv2.line(img, (rotated_p3_y, rotated_p3_x), (rotated_p4_y, rotated_p4_x),
             color, 1)
    cv2.line(img, (rotated_p4_y, rotated_p4_x), (rotated_p1_y, rotated_p1_x),
             color, 1)
    cv2.line(img, (center_p1p2y, center_p1p2x), (cy, cx), color, 1)


def get_corner_gtbox(box):
    """
    param: box(tuple, list): cx, cy, w, l
    return: (tuple): x_min, y_min, x_max, y_max
    """
    bx = box[0]
    by = box[1]
    bw = box[2]
    bl = box[3]
    top = int((by - bl / 2.0))
    left = int((bx - bw / 2.0))
    right = int((bx + bw / 2.0))
    bottom = int((by + bl / 2.0))
    return left, top, right, bottom


def remove_points(point_cloud, boundary_condition):
    """
    param point_cloud(array): Original point cloud data
    param boundary_condition(dict): The boundary of the area of interest
    return (array): Point cloud data within the area of interest
    """
    # Boundary condition
    min_x = boundary_condition['minX']
    max_x = boundary_condition['maxX']
    min_y = boundary_condition['minY']
    max_y = boundary_condition['maxY']
    min_z = boundary_condition['minZ']
    max_z = boundary_condition['maxZ']
    # Remove the point out of range x,y,z
    mask = np.where((point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x)
                    & (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y)
                    & (point_cloud[:, 2] >= min_z) & (point_cloud[:, 2] <= max_z))
    point_cloud = point_cloud[mask]
    point_cloud[:, 2] = point_cloud[:, 2] + 2
    return point_cloud


def make_bv_feature(point_cloud_):
    """
    param point_cloud_ (array): Point cloud data within the area of interest
    return (array): RGB map
    """
    # 1024 x 1024 x 3
    Height = 1024 + 1
    Width = 1024 + 1
    # Discretize Feature Map
    point_cloud = np.copy(point_cloud_)
    point_cloud[:, 0] = np.int_(np.floor(point_cloud[:, 0] / 60.0 * 768))
    point_cloud[:, 1] = np.int_(
        np.floor(point_cloud[:, 1] / 40.0 * 512) + Width / 2)
    # sort-3times
    indices = np.lexsort(
        (-point_cloud[:, 2], point_cloud[:, 1], point_cloud[:, 0]))
    point_cloud = point_cloud[indices]
    # Height Map
    height_map = np.zeros((Height, Width))
    _, indices = np.unique(point_cloud[:, 0:2], axis=0, return_index=True)
    point_cloud_frac = point_cloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    height_map[np.int_(point_cloud_frac[:, 0]),
               np.int_(point_cloud_frac[:, 1])] = point_cloud_frac[:, 2]
    # Intensity Map & DensityMap
    intensity_map = np.zeros((Height, Width))
    density_map = np.zeros((Height, Width))
    _, indices, counts = np.unique(point_cloud[:, 0:2],
                                   axis=0,
                                   return_index=True,
                                   return_counts=True)
    point_cloud_top = point_cloud[indices]
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensity_map[np.int_(point_cloud_top[:, 0]),
                  np.int_(point_cloud_top[:, 1])] = point_cloud_top[:, 3]
    density_map[np.int_(point_cloud_top[:, 0]),
                np.int_(point_cloud_top[:, 1])] = normalized_counts

    rgb_map = np.zeros((Height, Width, 3))
    rgb_map[:, :, 0] = density_map  # r_map
    rgb_map[:, :, 1] = height_map / 3.26  # g_map
    rgb_map[:, :, 2] = intensity_map  # b_map

    save = np.zeros((768, 1024, 3))
    save = rgb_map[0:768, 0:1024, :]
    return save


def get_target(label_file, transform):
    """

    param label_file  (str): The kitti label path
    param transform (array): Coordinate transformation matrix
    return (array): label
    """
    target = np.zeros([50, 6], dtype=np.float32)
    with open(label_file, 'r') as f:
        lines = f.readlines()
    num_obj = len(lines)
    index = 0
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        if obj_class in class_list:
            t_lidar, box3d_corner, rz = box3d_cam_to_velo(
                obj[8:], transform)  # get target  3D object location x,y
            location_x = t_lidar[0][0]
            location_y = t_lidar[0][1]
            if (location_x > 0) & (location_x < 60) & (location_y > -40) & (
                    location_y < 40):
                target[index][2] = t_lidar[0][0] / 60.0  # make sure target inside the covering area (0,1)
                target[index][1] = (t_lidar[0][1] + 40) / 80.0  # we should put this in [0,1], so divide max_size  80 m
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


def box3d_cam_to_velo(box3d, transform):
    def project_cam2velo(cam, transform):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = transform
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        """
        param ry (float): yaw angle in cam coordinate system
        return: (flaot): yaw angle in velodyne coordinate system
        """
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
    t_lidar = project_cam2velo(cam, transform)
    box = np.array(
        [[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
         [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
         [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)
    rot_mat = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                       [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]])
    velo_box = np.dot(rot_mat, box)
    corner_pos_in_velo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = corner_pos_in_velo.transpose()
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

    return {
        'P2': P2.reshape(3, 4),
        'R0': R0.reshape(3, 3),
        'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)
    }


def angle_rz_to_ry(rz):
    """
    param rz (float): yaw angle in velodyne coordinate system
    return (float): yaw angle in cam coordinate system
    """
    angle = -rz - np.pi / 2
    if angle < -np.pi:
        angle = 2 * np.pi + angle
    return angle


def coord_image_to_velo(hy, wx):
    """
    Convert image coordinates to velodyne coordinates
    """
    velo_x = hy * 60 / 768.0
    velo_y = (wx - 512) * 40 / 512.0
    return velo_x, velo_y


def coord_velo_to_cam(velo_x, velo_y, transform):
    """
    Convert velodyne coordinates to image coordinates
    """
    T = np.zeros([4, 4], dtype=np.float32)
    T[:3, :] = transform
    T[3, 3] = 1
    velo_coord = np.array([[velo_x], [velo_y], [1.5], [1]])
    cam_coord = np.dot(T, velo_coord)
    return cam_coord[0][0], cam_coord[2][0]


# anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
