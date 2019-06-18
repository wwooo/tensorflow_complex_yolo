# -*- coding: utf-8 -*-
import cv2
import os
import sys
sys.path.append('.')
from dataset.dataset import ImageDataSet
from utils.kitti_utils import draw_rotated_box, get_corner_gtbox
img_h, img_w = 768, 1024
gt_box_color = (255, 255, 255)
class_list = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    'Misc'
]
dataset = ImageDataSet(data_set='test',
                       mode='visualize',
                       flip=True,
                       random_scale=True,
                       load_to_memory=False)


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


make_dir('./tmp')
for img_idx, img, target in dataset.data_generator():
    # draw gt bbox
    print("process data: {}, saved in ./tmp".format(img_idx))
    for i in range(target.shape[0]):
        cx = int(target[i][1])
        cy = int(target[i][2])
        w = int(target[i][3])
        h = int(target[i][4])
        rz = target[i][5]
        draw_rotated_box(img, cx, cy, w, h, rz, gt_box_color)
        label = class_list[int(target[i][0])]
        box = get_corner_gtbox([cx, cy, w, h])
        cv2.putText(img, label, (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    gt_box_color, 1)
    cv2.imwrite('./tmp/{}.png'.format(img_idx), img[:, :, ::-1])
