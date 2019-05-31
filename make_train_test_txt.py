import os
train_label_dir = './train/image_label'
test_label_dir = './test/image_label'

train_list = sorted(os.listdir(train_label_dir))
test_list = sorted(os.listdir(test_label_dir))
print("the num of train: {}".format(len(train_list)))
print("the num of test: {}".format(len(test_list)))
with open("test_image_list.txt", 'w') as f:
    for name in test_list:
        image_idx = name.split('.')[0]
        f.write(str(image_idx))
        f.write('\n')

with open("train_image_list.txt", 'w') as f:
    for name in train_list:
        image_idx = name.split('.')[0]
        f.write(str(image_idx))
        f.write('\n')
