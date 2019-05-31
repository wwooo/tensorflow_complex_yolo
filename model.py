import tensorflow as tf
import numpy as np
SCALE = 32
GRID_W, GRID_H = 32, 24
N_CLASSES = 8
N_ANCHORS = 5
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H * SCALE, GRID_W * SCALE, 3
class_dict = {'Car':0,'Van':1,'Truck':2,'Pedestrian':3,'Person_sitting':4,'Cyclist':5,'Tram':6,'Misc':7}
def lrelu(x, leak):
    return tf.maximum(x, leak * x, name='relu')


def maxpool_layer(x, size, stride, name):
    with tf.name_scope(name):
        x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')
    return x


def conv_layer(x, kernel, depth, train_logical, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             bias_initializer=tf.zeros_initializer())
        x = tf.layers.batch_normalization(x, training=train_logical, momentum=0.9, epsilon=0.001, center=True,
                                          scale=True)
        x = lrelu(x, 0.2)
    # x = tf.nn.relu(x)
    return x


def passthrough_layer(a, b, kernel, depth, size, train_logical, name):
    b = conv_layer(b, kernel, depth, train_logical, name)
    b = tf.space_to_depth(b, size)
    y = tf.concat([a, b], axis=3)
    return y

def slice_tensor(x, start, end=None):
    if end < 0:
        y = x[..., start:]
    else:
        if end is None:
            end = start
        y = x[..., start:end + 1]
    return y


def iou_wh(r1, r2):
    min_w = min(r1[0], r2[0])
    min_h = min(r1[1], r2[1])
    area_r1 = r1[0] * r1[1]
    area_r2 = r2[0] * r2[1]

    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect

    return intersect / union
def get_grid_cell(roi, img_w, img_h, grid_w, grid_h):#roi[x, y, w, h, rz]
	x_center = roi[0]
	y_center = roi[1]
	grid_x = int(grid_w * x_center / img_w)
	grid_y = int(grid_h * y_center / img_h)
	return grid_x, grid_y

def get_active_anchors(roi, anchors, iou_th):
	indxs = []
	iou_max, index_max = 0, 0
	for i, a in enumerate(anchors):
		iou = iou_wh(roi[2:4], a)
		if iou>iou_th:
			indxs.append(i)
		if iou > iou_max:
			iou_max, index_max = iou, i
	if len(indxs) == 0:
		indxs.append(index_max)
	return indxs


def roi2label(roi, anchor, img_w, img_h, grid_w, grid_h):
    x_center = roi[0]
    y_center = roi[1]
    w = grid_w * roi[2] / img_w
    h = grid_h * roi[3] / img_h
    anchor_w = grid_w * anchor[0] / img_w
    anchor_h = grid_h * anchor[1] / img_h

    grid_x = grid_w * x_center / img_w
    grid_y = grid_h * y_center / img_h

    grid_x_offset = grid_x - int(grid_x)
    grid_y_offset = grid_y - int(grid_y)

    roi_w_scale = np.log(w / anchor_w + 1e-16)
    roi_h_scale = np.log(h / anchor_h + 1e-16)

    re = np.cos(roi[4])
    im = np.sin(roi[4])

    label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale, re, im]

    return label
def encode_label(label_txt, anchors, img_w, img_h, grid_w, grid_h, iou_th):
    #anchors = read_anchors_file(anchors_path)
    anchors_on_image = np.array([img_w, img_h]) * anchors / np.array([80, 60])
    n_anchors = np.shape(anchors_on_image)[0]

    label = np.zeros([grid_h, grid_w, n_anchors, (6 + 1 + 1)], dtype=np.float32)
    for label_list in label_txt:
        rois = label_list[1:]
        classes = class_dict[label_list[0]]
        rois = np.array(rois, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)
        active_indxs = get_active_anchors(rois, anchors_on_image, iou_th)
        grid_x, grid_y = get_grid_cell(rois, img_w, img_h, grid_w, grid_h)

        for active_indx in active_indxs:
            anchor_label = roi2label(rois, anchors_on_image[active_indx], img_w, img_h, grid_w, grid_h)

            label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [classes], [1.0]))

    return label



def yolo_net(x, train_logical):
    """darknet"""
    x = conv_layer(x, (3, 3), 24, train_logical, 'conv1')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool1')
    x = conv_layer(x, (3, 3), 48, train_logical, 'conv2')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool2')

    x = conv_layer(x, (3, 3), 64, train_logical, 'conv3')
    x = conv_layer(x, (1, 1), 32, train_logical, 'conv4')
    x = conv_layer(x, (3, 3), 64, train_logical, 'conv5')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool5')

    x = conv_layer(x, (3, 3), 128, train_logical, 'conv6')
    x = conv_layer(x, (1, 1), 64, train_logical, 'conv7')
    x = conv_layer(x, (3, 3), 128, train_logical, 'conv8')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool8')

    # x = conv_layer(x, (3, 3), 512, train_logical, 'conv9')
    # x = conv_layer(x, (1, 1), 256, train_logical, 'conv10')
    x = conv_layer(x, (3, 3), 512, train_logical, 'conv11')
    x = conv_layer(x, (1, 1), 256, train_logical, 'conv12')
    passthrough = conv_layer(x, (3, 3), 512, train_logical, 'conv13')
    x = maxpool_layer(passthrough, (2, 2), (2, 2), 'maxpool13')

    # x = conv_layer(x, (3, 3), 1024, train_logical, 'conv14')
    # x = conv_layer(x, (1, 1), 512, train_logical, 'conv15')
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv16')
    x = conv_layer(x, (1, 1), 512, train_logical, 'conv17')
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv18')

    x = passthrough_layer(x, passthrough, (3, 3), 64, 2, train_logical, 'conv21')
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv19')
    x = conv_layer(x, (1, 1), N_ANCHORS * (7 + N_CLASSES), train_logical, 'conv20')  # x,y,w,l,re,im,conf + 8 class

    y = tf.reshape(x, shape=(-1, GRID_H, GRID_W, N_ANCHORS, 7 + N_CLASSES), name='y')
    # print y
    return y

def yolo_loss(pred, label, batch_size):

    mask = slice_tensor(label, 7, 7)
    # print (mask.shape)
    label = slice_tensor(label, 0, 6)
    mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)), tf.bool)
    with tf.name_scope('mask'):
        masked_label = tf.boolean_mask(label, mask)
        masked_pred = tf.boolean_mask(pred, mask)
        neg_masked_pred = tf.boolean_mask(pred, tf.logical_not(mask))
    with tf.name_scope('pred'):
        masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
        masked_pred_wh = slice_tensor(masked_pred, 2, 3)
        masked_pred_re = 2 * tf.sigmoid(slice_tensor(masked_pred, 4, 4)) - 1
        masked_pred_im = 2 * tf.sigmoid(slice_tensor(masked_pred, 5, 5)) - 1
        masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 6, 6))

        masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 6, 6))
        #masked_pred_c = tf.nn.sigmoid(slice_tensor(masked_pred, 7, -1))
        masked_pred_c = tf.nn.softmax(slice_tensor(masked_pred, 7, -1))
    # masked_pred_no_c = tf.nn.sigmoid(slice_tensor(neg_masked_pred, 7, -1))
    # print (masked_pred_c, masked_pred_o, masked_pred_no_o)

    with tf.name_scope('lab'):
        masked_label_xy = slice_tensor(masked_label, 0, 1)
        masked_label_wh = slice_tensor(masked_label, 2, 3)
        masked_label_re = slice_tensor(masked_label, 4, 4)
        masked_label_im = slice_tensor(masked_label, 5, 5)
        masked_label_class = slice_tensor(masked_label, 6, 6)
        masked_label_class_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_class, tf.int32), depth=N_CLASSES),
                                            shape=(-1, N_CLASSES))
    with tf.name_scope('merge'):
        with tf.name_scope('loss_xy'):
            loss_xy = tf.reduce_sum(tf.square(masked_pred_xy - masked_label_xy)) / batch_size
        with tf.name_scope('loss_wh'):
            loss_wh = tf.reduce_sum(tf.square(masked_pred_wh - masked_label_wh)) / batch_size
        with tf.name_scope('loss_re'):
            loss_re = tf.reduce_sum(tf.square(masked_pred_re - masked_label_re)) / batch_size
        with tf.name_scope('loss_im'):
            loss_im = tf.reduce_sum(tf.square(masked_pred_im - masked_label_im)) / batch_size
        with tf.name_scope('loss_obj'):
            loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1)) / batch_size
        # loss_obj =  tf.reduce_sum(-tf.log(masked_pred_o+0.000001))*10
        with tf.name_scope('loss_no_obj'):
            loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o)) * 0.5 / batch_size
        # loss_no_obj =  tf.reduce_sum(-tf.log(1-masked_pred_no_o+0.000001))
        with tf.name_scope('loss_class'):
            # loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))
            loss_c = (tf.reduce_sum(-tf.log(masked_pred_c + 0.000001) * masked_label_class_vec) \
                     + tf.reduce_sum(-tf.log(1 - masked_pred_c + 0.000001) * (1 - masked_label_class_vec))) / batch_size
                # + tf.reduce_sum(-tf.log(1 - masked_pred_no_c+0.000001)) * 0.1
    # loss = (loss_xy + loss_wh+ loss_re + loss_im+ lambda_coord*loss_obj) + lambda_no_obj*loss_no_obj + loss_c
    loss = (loss_xy + loss_wh + loss_re + loss_im) * 5 + loss_obj + loss_no_obj + loss_c
    return loss, loss_xy, loss_wh, loss_re, loss_im, loss_obj, loss_no_obj, loss_c

