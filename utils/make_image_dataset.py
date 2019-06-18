# -*- coding: utf-8 -*-
from __future__ import division
import os
import cv2
import numpy as np
import sys
sys.path.append('.')
from dataset.dataset import PointCloudDataset
from utils.model_utils import make_dir
image_dataset_dir = 'kitti/image_dataset/'
class_list = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]
img_h, img_w = 768, 1024
# dataset
train_dataset = PointCloudDataset(root='kitti/', data_set='train')
test_dataset = PointCloudDataset(root='kitti/', data_set='test')


def delete_file_folder(src):
    if os.path.isfile(src):
        os.remove(src)
    elif os.path.isdir(src):
        for item in os.listdir(src):
            item_src = os.path.join(src, item)
            delete_file_folder(item_src)


def preprocess_dataset(dataset):
    """
    Convert point cloud data to image  while
    filtering out image without objects.
    param: dataset: (PointCloudDataset)
    return: None
    """
    for img_idx, rgb_map, target in dataset.getitem():
        rgb_map = np.array(rgb_map * 255, np.uint8)
        target = np.array(target)
        print('process image： {}'.format(img_idx))
        for i in range(target.shape[0]):
            if target[i].sum() == 0:
                break
            with open("kitti/image_dataset/labels/{}.txt".format(img_idx), 'a+') as f:
                label = class_list[int(target[i][0])]
                cx = target[i][1] * img_w
                cy = target[i][2] * img_h
                w = target[i][3] * img_w
                h = target[i][4] * img_h
                rz = target[i][5]
                line = label + ' ' + '{} {} {} {} {}\n'.format(cx, cy, w, h, rz)
                f.write(line)
        cv2.imwrite('kitti/image_dataset/images/{}.png'.format(img_idx), rgb_map[:, :, ::-1])
    print('make image dataset done！')


def make_train_test_list():
    name_list = os.listdir(image_dataset_dir + 'labels')
    name_list.sort()
    with open('config/test_image_list.txt', 'w') as f:
        for name in name_list[0:1000]:
            f.write(name.split('.')[0])
            f.write('\n')
    with open('config/train_image_list.txt', 'w') as f:
        for name in name_list[1000:]:
            f.write(name.split('.')[0])
            f.write('\n')


if __name__ == "__main__":
    make_dir(image_dataset_dir + 'images')
    make_dir(image_dataset_dir + 'labels')
    delete_file_folder(image_dataset_dir + 'labels')
    preprocess_dataset(train_dataset)
    preprocess_dataset(test_dataset)
    make_train_test_list()

