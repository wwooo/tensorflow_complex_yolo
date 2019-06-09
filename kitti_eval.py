from __future__ import division
import numpy as np
import os
import tensorflow as tf
from kitti import KittiDataset
import cv2
from utils import softmax, sigmoid, draw_rotate_rec,\
    cal_angle, get_corner_gtbox, read_anchors_file, \
    read_class_flag, non_max_supression, filter_bbox
from utils import load_kitti_calib
gt_color = (255, 255, 255)
prob_th = 0.3
iou_th = 0.4
n_anchors = 5
n_classes = 8
net_scale = 32
img_h, img_w = 768, 1024
grid_w, grid_h = 32, 24
test_image_path = "./images/"
class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram', 'Misc' ]
# dataset
#dataset = KittiDataset(root='../../kitti/KITTI',set='test')
train_list = './train_image_list.txt'
test_list = './test_image_list.txt'
calib_dir = './kitti/training/calib/'
kitti_static_cylist = 'cyclist_detection_ground.txt'
kitti_static_car = 'car_detection_ground.txt'
kitti_static_pedestrian = 'pedestrian_detection_ground.txt'


def angle_rz_to_ry(rz):
    angle = -rz - np.pi / 2
    if angle < -np.pi:
        angle = 2 * np.pi + angle

    return angle


def coord_image_to_velo(hy, wx):
    velo_x = hy * 60 / 768.0
    velo_y = (wx - 512) * 40 / 512.0
    return velo_x, velo_y


def coord_velo_to_cam(velo_x, velo_y, tr):
    T = np.zeros([4, 4], dtype=np.float32)
    T[:3, :] = tr
    T[3, 3] = 1
    velo_coord = np.array([[velo_x], [velo_y], [1.5], [1]])
    cam_coord = np.dot(T, velo_coord)
    return cam_coord[0][0],  cam_coord[2][0]


def preprocess_data(data, anchors, important_classes):
    locations = []
    classes = []
    for i in range(grid_h):
        for j in range(grid_w):
            for k in range(n_anchors):
                class_vec = softmax(data[0, i, j, k, 7:])
                objectness = sigmoid(data[0, i, j, k, 6])
                class_prob = objectness*class_vec
                w = np.exp(data[0, i, j, k, 2]) * anchors[k][0] / 80 * grid_w * net_scale
                h = np.exp(data[0, i, j, k, 3]) * anchors[k][1] / 60 * grid_h * net_scale
                dx = sigmoid(data[0, i, j, k, 0])
                dy = sigmoid(data[0, i, j, k, 1])
                re = 2 * sigmoid(data[0, i, j, k, 4]) - 1
                im = 2 * sigmoid(data[0, i, j, k, 5]) - 1
                y = (i+dy) * net_scale
                x = (j+dx) * net_scale
                classes.append(class_prob[important_classes])
                locations.append([x, y, w, h, re, im])
    classes = np.array(classes)
    locations = np.array(locations)
    return classes, locations


def kitti_eval():
    important_classes, names, colors = read_class_flag('./yolo.labels')
    anchors = read_anchors_file('./kitti_anchors.txt')
    sess = tf.Session()
    saver = tf.train.import_meta_graph(
        './model_serve/yolo_tloss_0.03682877495884895_vloss_3.0288465037403336-171000.meta')
    saver.restore(sess, './model_serve/yolo_tloss_0.03682877495884895_vloss_3.0288465037403336-171000')
    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image_placeholder:0")
    train_flag = graph.get_tensor_by_name("flag_placeholder:0")
    y = graph.get_tensor_by_name("net/y:0")
    # with open(test_list, 'r') as test_file:
    #     test_file_index_list = test_file.readlines()
    for test_file_index in range(1000):
        # test_file_index = test_file_index.strip()
        calib_file = calib_dir + str(test_file_index).zfill(6) + '.txt'
        calib = load_kitti_calib(calib_file)
        result_file = "./eval_results/" + str(test_file_index).zfill(6) + ".txt"
        img_path = test_image_path + str(test_file_index).zfill(6) + '.png'
        rgb_map = cv2.imread(img_path)[:, :, ::-1]
        img_for_net = rgb_map / 255.0
        data = sess.run(y, feed_dict={image: [img_for_net], train_flag: False})
        classes, rois = preprocess_data(data, anchors, important_classes)
        classes, indxs = non_max_supression(classes, rois, prob_th, iou_th)
        all_bboxs = filter_bbox(classes, rois, indxs)
        with open(result_file, "w") as f:
            for box in all_bboxs:
                pred_img_y = box[2]
                pred_img_x = box[1]
                velo_x, velo_y = coord_image_to_velo(pred_img_y, pred_img_x)
                cam_x, cam_z = coord_velo_to_cam(velo_x, velo_y, calib['Tr_velo2cam'])
                pred_width = box[3] * 80 / 1024.0
                pred_height = box[4] * 60 / 768.0
                pred_cls = class_list[box[0]]
                pred_conf = box[7]
                angle_rz = cal_angle(box[6], box[5])
                angle_ry = angle_rz_to_ry(angle_rz)
                pre_line = pred_cls + " " + "-1" + " " + "-1" + " " + "-10" + " " + \
                           "{:.2f} {:.2f} {:.2f} {:.2f}".format(-1, -1, -1, -1) + " " + \
                           "-1" + " " + "{:.2f}".format(pred_width) + " " + "{:.2f}".format(pred_height) + \
                           " " + "{:.2f} {:.2f} {:.2f}".format(cam_x, -1000, cam_z) + " " + "{:.2f}".format(angle_ry) + \
                           " " + "{:.2f}".format(pred_conf)
                f.write(pre_line)
                f.write("\n")


def cal_ap(kitti_statics_results):
    with open(kitti_statics_results, 'r') as f:
        lines = f.readlines()
        all_lines = []
        for line in lines:
            pr = list(map(float, line.strip().split(' ')))
            all_lines.append(pr)
        all_lines = np.array(all_lines)
        ap = np.zeros([all_lines.shape[0], 3])
        ap[1:, 0] = 0.025 * all_lines[1:, 1]
        ap[1:, 1] = 0.025 * all_lines[1:, 2]
        ap[1:, 2] = 0.025 * all_lines[1:, 3]
        result = np.sum(ap, 0)
        return result


if __name__ == '__main__':
    kitti_eval()
    # print("car ap: {}".format(cal_ap(kitti_static_car)))
    # print("cyclist ap: {}".format(cal_ap(kitti_static_cylist)))
    # print("pedestrian ap: {}".format(cal_ap(kitti_static_pedestrian)))

