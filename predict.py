# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import tensorflow as tf
from dataset.dataset import PointCloudDataset
import cv2
from utils.model_utils import softmax, sigmoid, non_max_supression, filter_bbox
from utils.kitti_utils import draw_rotated_box, cal_angle, get_corner_gtbox, \
    read_anchors_from_file, read_class_flag
gt_box_color = (255, 255, 255)
prob_th = 0.3
nms_iou_th = 0.4
n_anchors = 5
n_classes = 8
net_scale = 32
img_h, img_w = 768, 1024
grid_w, grid_h = 32, 24
class_list = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    'Misc'
]
# dataset
dataset = PointCloudDataset(root='./kitti/', data_set='test')


def preprocess_data(data, anchors, important_classes):
    locations = []
    classes = []
    for i in range(grid_h):
        for j in range(grid_w):
            for k in range(n_anchors):
                class_vec = softmax(data[0, i, j, k, 7:])
                object_conf = sigmoid(data[0, i, j, k, 6])
                class_prob = object_conf * class_vec
                w = np.exp(data[0, i, j, k, 2]
                           ) * anchors[k][0] / 80 * grid_w * net_scale
                h = np.exp(data[0, i, j, k, 3]
                           ) * anchors[k][1] / 60 * grid_h * net_scale
                dx = sigmoid(data[0, i, j, k, 0])
                dy = sigmoid(data[0, i, j, k, 1])
                re = 2 * sigmoid(data[0, i, j, k, 4]) - 1
                im = 2 * sigmoid(data[0, i, j, k, 5]) - 1
                y = (i + dy) * net_scale
                x = (j + dx) * net_scale
                classes.append(class_prob[important_classes])
                locations.append([x, y, w, h, re, im])
    classes = np.array(classes)
    locations = np.array(locations)
    return classes, locations


def predict(draw_gt_box=False):

    important_classes, names, color = read_class_flag('config/class_flag.txt')
    anchors = read_anchors_from_file('config/kitti_anchors.txt')
    sess = tf.Session()
    saver = tf.train.import_meta_graph(
        './weights/yolo_tloss_0.17186672985553741_vloss_3.1235308539436524-96000.meta'
    )
    saver.restore(
        sess,
        './weights/yolo_tloss_0.17186672985553741_vloss_3.1235308539436524-96000'
    )
    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image_placeholder:0")
    train_flag = graph.get_tensor_by_name("flag_placeholder:0")
    y = graph.get_tensor_by_name("net/y:0")
    for img_idx, rgb_map, target in dataset.getitem():
        img = np.array(rgb_map * 255, np.uint8)
        target = np.array(target)
        # draw gt bbox
        if draw_gt_box:
            for i in range(target.shape[0]):
                if target[i].sum() == 0:
                    break
                cx = int(target[i][1] * img_w)
                cy = int(target[i][2] * img_h)
                w = int(target[i][3] * img_w)
                h = int(target[i][4] * img_h)
                rz = target[i][5]
                draw_rotated_box(img, cx, cy, w, h, rz, gt_box_color)
                label = class_list[int(target[i][0])]
                box = get_corner_gtbox([cx, cy, w, h])
                cv2.putText(img, label, (box[0], box[1]),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, gt_box_color, 1)
        data = sess.run(y, feed_dict={image: [rgb_map], train_flag: False})
        classes, rois = preprocess_data(data, anchors, important_classes)
        classes, index = non_max_supression(classes, rois, prob_th, nms_iou_th)
        all_boxes = filter_bbox(classes, rois, index)
        for box in all_boxes:
            class_idx = box[0]
            corner_box = get_corner_gtbox(box[1:5])
            angle = cal_angle(box[6], box[5])
            class_prob = box[7]
            draw_rotated_box(img, box[1], box[2], box[3], box[4], 
                             angle, color[class_idx])
            cv2.putText(img,
                        class_list[class_idx] + ' : {:.2f}'.format(class_prob),
                        (corner_box[0], corner_box[1]), cv2.FONT_HERSHEY_PLAIN,
                        0.7, color[class_idx], 1, cv2.LINE_AA)
        cv2.imwrite('./predict_result/{}.png'.format(img_idx), img)


if __name__ == '__main__':
    predict(draw_gt_box=True)
