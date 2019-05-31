from __future__ import division
import os
import os.path
import numpy as np
import cv2
import math

from utils import *


class KittiDataset(object):

    def __init__(self, root='./kitti/KITTI',set='train',type='velodyne_train'):

        self.type = type
        self.root = root
        self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "velodyne/")
        self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()


    def getitem(self):
        for file in self.file_list:
            lidar_file = self.lidar_path + '/' + file + '.bin'
            calib_file = self.calib_path + '/' + file + '.txt'
            label_file = self.label_path + '/' + file + '.txt'
            #image_file = self.image_path + '/' + file + '.png'

            if self.type == 'velodyne_train':
                calib = load_kitti_calib(calib_file)
                target = get_target(label_file,calib['Tr_velo2cam'])
            # load point cloud data
                a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
                b = removePoints(a,bc)
                data = makeBVFeature(b)   # (768, 1024, 3)

            yield file, data , target

