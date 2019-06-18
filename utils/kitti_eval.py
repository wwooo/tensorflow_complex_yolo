# -*- coding: utf-8 -*-
"""
This script is mainly used to generate the prediction result
of each RGB-map of the test set, convert the bounding box coordinates
in the image to the lidar coordinate system, and then convert the
coordinates to the camera coordinate system using the coordinate
transformation matrix provided by the kitti data set.
The coordinate transformation requires (x, y, z), since the complex-yolo
does not predict the height of the objects, the height is set to a fixed value
of 1.5, which does not affect the bird's eye view benchmark evaluation.

"""
from __future__ import division
import numpy as np
import argparse
import tensorflow as tf
import cv2
from model_utils import preprocess_data, non_max_supression, filter_bbox, make_dir
from kitti_utils import load_kitti_calib, calculate_angle,\
     read_anchors_from_file, read_class_flag
from kitti_utils import angle_rz_to_ry, coord_image_to_velo, coord_velo_to_cam
prob_th = 0.3
iou_th = 0.4
n_anchors = 5
n_classes = 8
net_scale = 32
img_h, img_w = 768, 1024
grid_w, grid_h = 32, 24
test_image_path = "kitti/image_dataset/images/"
class_list = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]
train_list = 'config/train_image_list.txt'
test_list = 'config/test_image_list.txt'
calib_dir = 'kitti/training/calib/'

# kitti_static_cylist = 'cyclist_detection_ground.txt'
# kitti_static_car = 'car_detection_ground.txt'
# kitti_static_pedestrian = 'pedestrian_detection_ground.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, default='./weights/yolo_tloss_1.185166835784912_vloss_2.9397876932621-220800', help="set the weights_path")
args = parser.parse_args()
weights_path = args.weights_path

make_dir("./eval_results")


def kitti_eval():
    important_classes, names, colors = read_class_flag('config/class_flag.txt')
    anchors = read_anchors_from_file('config/kitti_anchors.txt')
    sess = tf.Session()
    saver = tf.train.import_meta_graph(weights_path + '.meta')
    saver.restore(sess, weights_path)
    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image_placeholder:0")
    train_flag = graph.get_tensor_by_name("flag_placeholder:0")
    y = graph.get_tensor_by_name("net/y:0")
    for test_file_index in range(1000):
        print('process data: {}, saved in ./eval_results'.format(test_file_index))
        calib_file = calib_dir + str(test_file_index).zfill(6) + '.txt'
        calib = load_kitti_calib(calib_file)
        result_file = "./eval_results/" + str(test_file_index).zfill(6) + ".txt"
        img_path = test_image_path + str(test_file_index).zfill(6) + '.png'
        rgb_map = cv2.imread(img_path)[:, :, ::-1]
        img_for_net = rgb_map / 255.0
        data = sess.run(y,
                        feed_dict={image: [img_for_net],
                                   train_flag: False})
        classes, rois = preprocess_data(data,
                                        anchors,
                                        important_classes,
                                        grid_w,
                                        grid_h,
                                        net_scale)
        classes, index = non_max_supression(classes,
                                            rois,
                                            prob_th,
                                            iou_th)
        all_boxes = filter_bbox(classes, rois, index)
        with open(result_file, "w") as f:
            for box in all_boxes:
                pred_img_y = box[2]
                pred_img_x = box[1]
                velo_x, velo_y = coord_image_to_velo(pred_img_y, pred_img_x)
                cam_x, cam_z = coord_velo_to_cam(velo_x, velo_y, calib['Tr_velo2cam'])
                pred_width = box[3] * 80.0 / img_w
                pred_height = box[4] * 60.0 / img_h
                pred_cls = class_list[box[0]]
                pred_conf = box[7]
                angle_rz = calculate_angle(box[6], box[5])
                angle_ry = angle_rz_to_ry(angle_rz)
                pred_line = pred_cls + " -1 -1 -10 -1 -1 -1 -1 -1" + \
                    " {:.2f} {:.2f}".format(pred_width, pred_height) + \
                    " {:.2f} {:.2f} {:.2f}".format(cam_x, -1000, cam_z) + \
                    " {:.2f} {:.2f}".format(angle_ry, pred_conf)
                f.write(pred_line)
                f.write("\n")


def cal_ap(kitti_statics_results):
    """
    Calculate the ap  approximately.
    param kitti_statics_results(str): Kitti evaluation script output statistics result file.
    return:
    """
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
