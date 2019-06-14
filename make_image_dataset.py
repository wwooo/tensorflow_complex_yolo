# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import cv2
import os
from dataset.dataset import PointCloudDataset
class_list = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    'Misc'
]
img_h, img_w = 768, 1024
# dataset
train_dataset = PointCloudDataset(root='./kitti/', data_set='train')
test_dataset = PointCloudDataset(root='./kitti/', data_set='test')


def delete_file_folder(src):
    if os.path.isfile(src):
        os.remove(src)
    elif os.path.isdir(src):
        for item in os.listdir(src):
            item_src = os.path.join(src, item)
            delete_file_folder(item_src)


def preprocess_dataset(data_type, dataset):
    """
    Convert point cloud data to image  while
    filtering out image without objects.
    param: data_type (str) : 'train' or 'test',
    param: dataset: (PointCloudDataset)
    return: None
    """
    for img_idx, rgb_map, target in dataset.getitem():
        rgb_map = np.array(rgb_map * 255, np.uint8)
        target = np.array(target)
        print('process {} data： {}'.format(data_type, img_idx))
        for i in range(target.shape[0]):
            if target[i].sum() == 0:
                break
            with open("./{}/labels/{}.txt".format(data_type, img_idx), 'a+') as f:
                label = class_list[int(target[i][0])]
                cx = target[i][1] * img_w
                cy = target[i][2] * img_h
                w = target[i][3] * img_w
                h = target[i][4] * img_h
                rz = target[i][5]
                line = label + ' ' + '{} {} {} {} {}\n'.format(cx, cy, w, h, rz)
                f.write(line)
        cv2.imwrite('./{}/images/{}.png'.format(data_type, img_idx), rgb_map[:, :, ::-1])
    print('make {} dataset done！'.format(data_type))


if __name__ == "__main__":
    delete_file_folder('train/labels')
    delete_file_folder('test/labels')
    preprocess_dataset('train', train_dataset)
    preprocess_dataset('test', test_dataset)
