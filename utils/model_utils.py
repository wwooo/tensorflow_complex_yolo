import numpy as np
def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea


def iou(r1, r2):
    intersect_w = np.maximum(np.minimum(r1[0] + r1[2], r2[0] + r2[2]) - np.maximum(r1[0], r2[0]), 0)
    intersect_h = np.maximum(np.minimum(r1[1] + r1[3], r2[1] + r2[3]) - np.maximum(r1[1], r2[1]), 0)
    area_r1 = r1[2] * r1[3]
    area_r2 = r2[2] * r2[3]
    intersect = intersect_w * intersect_h
    union = area_r1 + area_r2 - intersect
    return intersect / union


def softmax(x):
    e_x = np.exp(x)
    return e_x/np.sum(e_x)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def non_max_supression(classes, locations, prob_th, iou_th):
    classes = np.transpose(classes)
    indxs = np.argsort(-classes, axis=1)

    for i in range(classes.shape[0]):
        classes[i] = classes[i][indxs[i]]

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):
            if roi_prob < prob_th:
                classes[class_idx][roi_idx] = 0

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):
            if roi_prob == 0:
                continue
            roi = locations[indxs[class_idx][roi_idx]][0:4]
            for roi_ref_idx, roi_ref_prob in enumerate(class_vec):
                if roi_ref_prob == 0 or roi_ref_idx <= roi_idx:
                    continue
                roi_ref = locations[indxs[class_idx][roi_ref_idx]][0:4]
                if bbox_iou(roi, roi_ref, False) > iou_th:
                    classes[class_idx][roi_ref_idx] = 0
    return classes, indxs


def filter_bbox(classes, rois, indxs):
    all_bboxs = []
    for class_idx, c in enumerate(classes):
        for loc_idx, class_prob in enumerate(c):
            if class_prob > 0:
                x = int(rois[indxs[class_idx][loc_idx]][0])
                y = int(rois[indxs[class_idx][loc_idx]][1])
                w = int(rois[indxs[class_idx][loc_idx]][2])
                h = int(rois[indxs[class_idx][loc_idx]][3])
                re = rois[indxs[class_idx][loc_idx]][4]
                im = rois[indxs[class_idx][loc_idx]][5]
                all_bboxs.append([class_idx, x, y, w, h, re, im, class_prob])
    return all_bboxs
