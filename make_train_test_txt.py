# -*- coding: utf-8 -*-
import os
train_label_dir = './train/labels'
test_label_dir = './test/labels'

train_list = sorted(os.listdir(train_label_dir))
test_list = sorted(os.listdir(test_label_dir))
print("the num of train images: {}".format(len(train_list)))
print("the num of test images: {}".format(len(test_list)))


def make_image_list(dataset_type, name_list):
    with open("config/{}_image_list.txt".format(dataset_type), 'w') as f:
        for name in name_list:
            image_idx = name.split('.')[0]
            f.write(str(image_idx))
            f.write('\n')


if __name__ == "__main__":
    make_image_list('test', test_list)
    make_image_list('train', train_list)