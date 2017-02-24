import numpy as np
from skimage.io import imread
from skimage.transform import rescale

import chainer

from faster_rcnn import FasterRCNN
from nms_cpu import nms_cpu as nms

from chainer_cv.datasets.pascal_voc.voc_utils import pascal_voc_labels
from chainer_cv.visualizations import vis_img_bbox


def predict_to_bboxes(cls_prob, pred_bboxes, nms_thresh, confidence):
    final_bboxes = []
    for cls_id in range(1, 21):
        _cls = cls_prob[:, cls_id][:, None]  # (300, 1)
        _bbx = pred_bboxes[:, cls_id * 4: (cls_id + 1) * 4]  # (300, 4)
        dets = np.hstack((_bbx, _cls))  # (300, 5)
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= confidence)[0]
        if len(inds) > 0:
            selected = dets[inds]
            final_bboxes.append(
                np.concatenate(
                    (selected[:, :4], np.ones((len(selected), 1)) * cls_id),
                    axis=1)
            )
    final_bboxes = np.concatenate(final_bboxes, axis=0)
    return final_bboxes


if __name__ == '__main__':
    nms_thresh = 0.3
    confidence = 0.8

    model = FasterRCNN()
    chainer.serializers.load_npz('VGG16_faster_rcnn_final.model', model)

    bgr_mean = np.array([103.939, 116.779, 123.68])
    original = imread('004545.jpg')

    img = original.astype(np.float32)
    scale = 600 / float(img.shape[0])
    max_value = np.max(np.abs(img))
    img = rescale(img / max_value, scale) * max_value
    img = img[:, :, ::-1]
    img -= bgr_mean
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)

    model.train = False
    cls_prob, pred_bboxes = model(img[None])  # (300, 21) and (300, 84)
    cls_prob = cls_prob[0]
    pred_bboxes = pred_bboxes[0] / scale

    final_bboxes = predict_to_bboxes(
        cls_prob, pred_bboxes, nms_thresh, confidence)

    vis_img_bbox(original, final_bboxes, pascal_voc_labels)
    import matplotlib.pyplot as plt
    plt.show()
