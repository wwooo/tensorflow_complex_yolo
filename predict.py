# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import argparse
import numpy as np
import tensorflow as tf
from dataset.dataset import PointCloudDataset
from utils.model_utils import preprocess_data, non_max_supression, filter_bbox, make_dir
from utils.kitti_utils import draw_rotated_box, calculate_angle, get_corner_gtbox, \
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
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

parser = argparse.ArgumentParser()
parser.add_argument("--draw_gt_box", type=str,  default='True', help="Whether to draw_gtbox, True or False")
parser.add_argument("--weights_path", type=str, default='./weights/yolo_tloss_1.185166835784912_vloss_2.9397876932621-220800',
                    help="set the weights_path")
args = parser.parse_args()
weights_path = args.weights_path

# dataset
dataset = PointCloudDataset(root='./kitti/', data_set='test')
make_dir('./predict_result')


def predict(draw_gt_box='False'):

    important_classes, names, color = read_class_flag('config/class_flag.txt')
    anchors = read_anchors_from_file('config/kitti_anchors.txt')
    sess = tf.Session()
    saver = tf.train.import_meta_graph(weights_path + '.meta')
    saver.restore(sess, weights_path)
    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image_placeholder:0")
    train_flag = graph.get_tensor_by_name("flag_placeholder:0")
    y = graph.get_tensor_by_name("net/y:0")
    for img_idx, rgb_map, target in dataset.getitem():
        print("process data: {}, saved in ./predict_result/".format(img_idx))
        img = np.array(rgb_map * 255, np.uint8)
        target = np.array(target)
        # draw gt bbox
        if draw_gt_box == 'True':
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
        classes, rois = preprocess_data(data, anchors, important_classes,
                                        grid_w, grid_h, net_scale)
        classes, index = non_max_supression(classes, rois, prob_th, nms_iou_th)
        all_boxes = filter_bbox(classes, rois, index)
        for box in all_boxes:
            class_idx = box[0]
            corner_box = get_corner_gtbox(box[1:5])
            angle = calculate_angle(box[6], box[5])
            class_prob = box[7]
            draw_rotated_box(img, box[1], box[2], box[3], box[4],
                             angle, color[class_idx])
            cv2.putText(img,
                        class_list[class_idx] + ' : {:.2f}'.format(class_prob),
                        (corner_box[0], corner_box[1]), cv2.FONT_HERSHEY_PLAIN,
                        0.7, color[class_idx], 1, cv2.LINE_AA)
        cv2.imwrite('./predict_result/{}.png'.format(img_idx), img)


if __name__ == '__main__':
    predict(draw_gt_box=args.draw_gt_box)
