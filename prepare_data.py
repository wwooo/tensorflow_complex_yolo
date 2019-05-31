from __future__ import division
import numpy as np
from kitti import KittiDataset
import cv2

from utils import *
batch_size = 1
class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram', 'Misc' ]
img_h, img_w = 768, 1024
# dataset
dataset=KittiDataset(root='./kitti/KITTI', set='train')

############################################################################

for img_idx, rgb_map, target in dataset.getitem():

    rgb_map =np.array(rgb_map * 255,  np.uint8)
    target = np.array(target)

  ###############################################

    for i in range(target.shape[0]):
        if target[i][1] == 0 and target[i][2] == 0:
            break
        with open("./train/image_label/{}.txt".format(img_idx), 'a+') as f:
            label = class_list[int(target[i][0])]
            cx = target[i][1] * img_w
            cy = target[i][2] * img_h
            w =  target[i][3] * img_w
            h =  target[i][4] * img_h
            rz = target[i][5]
            line = label  + ' ' + '{} {} {} {} {}\n'.format(cx, cy, w, h, rz)
            f.write(line)
        # cx = int(cx)
        # cy = int(cy)
        # w = int(w)
        # h = int(h)
        # angle = cal_angle(np.sin(target[i][5]), np.cos(target[i][5]) )
        # _draw_rotate_rec(rgb_map, cx, cy, w, h, angle)
        # label = class_list[int(target[i][0])]
        # box = get_corner_gtbox([cx, cy, w, h])
        # cv2.putText(rgb_map, label, (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1.0, [0, 0, 255], 1)
        # cv2.rectangle(rgb_map, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
    ########################################################

    cv2.imwrite('./train/image/{}.png'.format(img_idx), rgb_map[:,:,::-1])
