import numpy as np
import cv2
from model import encode_label
img_h, img_w = 768, 1024
grrid_h, grid_w = 24, 32
iou_th = 0.5
anchors_path = './kitti_anchors.txt'
labels_dir = 'labels/'
images_dir = 'images/'
train_name_list = './train_image_list.txt'
test_name_list = './test_image_list.txt'
class_dict = {'Car':0,'Van':1,'Truck':2,'Pedestrian':3,'Person_sitting':4,'Cyclist':5,'Tram':6,'Misc':7}

def read_label_from_txt(label_path):  # label relative image coord-> class x y w h rz, rz: rotation angle
	"""Read label from txt file."""
	bounding_box = []
	with open(label_path, "r") as f:
		labels = f.readlines()

		for label in labels:
			if not label:
				continue
			label = label.strip().split(' ')
			bounding_box.append(label)
	if bounding_box:
		return bounding_box
	else:
		return None


def read_anchors_file(file_path):
	anchors = []
	with open(file_path, 'r') as file:
		for line in file.read().splitlines():
			anchors.append(list(map(float, line.split())))
	return np.array(anchors)

anchors = read_anchors_file(anchors_path)


def data_generator(data_name_list):
    with open(data_name_list, 'r') as f:
        lines = f.readlines()
        while True:
            np.random.shuffle(lines)
            for line in lines:
                img_index = line.strip()
                label_file = labels_dir + img_index + '.txt'
                label_txt = read_label_from_txt(label_file)
                image_path = images_dir + img_index + '.png'
                img = cv2.imread(image_path)[:, :, ::-1] / 255.0
                label_encoded = encode_label(label_txt, anchors, img_w, img_h, grid_w, grrid_h, iou_th)
                yield img, label_encoded

def get_batch(batch_size, data_name_list = train_name_list):
    img_batch = []
    label_batch = []
    i = 0
    for img, label_encoded in data_generator(data_name_list):
        i += 1
        img_batch.append(img)
        label_batch.append(label_encoded)
        if i % batch_size == 0:
            yield np.array(img_batch) , np.array(label_batch)
            i = 0
            img_batch = []
            label_batch = []






