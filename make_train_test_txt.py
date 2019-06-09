import os
train_label_dir = './train/labels'
test_label_dir = './test/labels'

train_list = sorted(os.listdir(train_label_dir))
test_list = sorted(os.listdir(test_label_dir))
print("the num of train: {}".format(len(train_list)))
print("the num of test: {}".format(len(test_list)))
with open("config/test_image_list.txt", 'w') as f:
    for name in test_list:
        image_idx = name.split('.')[0]
        f.write(str(image_idx))
        f.write('\n')

with open("config/train_image_list.txt", 'w') as f:
    for name in train_list:
        image_idx = name.split('.')[0]
        f.write(str(image_idx))
        f.write('\n')
