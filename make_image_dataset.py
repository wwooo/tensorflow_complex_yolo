from __future__ import division
import numpy as np
import cv2
import os
from dataset.dataset import PointCloudDataset
class_list = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
img_h, img_w = 768, 1024
# dataset
train_dataset = PointCloudDataset(root='./kitti/', data_set='train')
test_dataset = PointCloudDataset(root='./kitti/', data_set='test')


def delete_file_folder(src):
    if os.path.isfile(src):
        os.remove(src)
    elif os.path.isdir(src):
        for item in os.listdir(src):
            itemsrc = os.path.join(src, item)
            delete_file_folder(itemsrc)


delete_file_folder('train/labels')
delete_file_folder('test/labels')
for img_idx, rgb_map, target in train_dataset.getitem():
    rgb_map = np.array(rgb_map * 255,  np.uint8)
    target = np.array(target)
    print('process train data： {}'.format(img_idx))
    for i in range(target.shape[0]):
        if target[i].sum() == 0:
            break
        with open("./train/labels/{}.txt".format(img_idx), 'a+') as f:
            label = class_list[int(target[i][0])]
            cx = target[i][1] * img_w
            cy = target[i][2] * img_h
            w = target[i][3] * img_w
            h = target[i][4] * img_h
            rz = target[i][5]
            line = label + ' ' + '{} {} {} {} {}\n'.format(cx, cy, w, h, rz)
            f.write(line)
    cv2.imwrite('./train/images/{}.png'.format(img_idx), rgb_map[:, :, ::-1])

for img_idx, rgb_map, target in test_dataset.getitem():
    rgb_map = np.array(rgb_map * 255,  np.uint8)
    target = np.array(target)
    print('process test data： {}'.format(img_idx))
    for i in range(target.shape[0]):
        if target[i].sum() == 0:
            break
        with open("./test/labels/{}.txt".format(img_idx), 'a+') as f:
            label = class_list[int(target[i][0])]
            cx = target[i][1] * img_w
            cy = target[i][2] * img_h
            w = target[i][3] * img_w
            h = target[i][4] * img_h
            rz = target[i][5]
            line = label + ' ' + '{} {} {} {} {}\n'.format(cx, cy, w, h, rz)
            f.write(line)
    cv2.imwrite('./test/images/{}.png'.format(img_idx), rgb_map[:, :, ::-1])
print('make image dataset done！')
