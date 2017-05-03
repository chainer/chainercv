import matplotlib
matplotlib.use('agg')
import chainer

import chainercv
from chainercv.datasets import VOCDetectionDataset
from chainercv.links import FasterRCNNVGG
from chainercv import transforms
from chainercv.datasets import TransformDataset

import numpy as np
import chainer
from chainer.dataset.convert import concat_examples

from chainercv.datasets.pascal_voc.voc_utils import pascal_voc_labels

from chainercv.evaluations import eval_detection

import fire

mean_pixel = np.array([102.9801, 115.9465, 122.7717])[:, None, None]


def record_bbox(model, dataset, device, n_class):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    n_img = len(dataset)
    bboxes = []
    confs = []
    labels = []
    gt_bboxes = []
    gt_labels = []
    gt_difficults = []
    for i in range(len(dataset)):
        batch = dataset[i:i+1]

        in_arrays = concat_examples(batch, device)
        in_vars = tuple(chainer.Variable(x) for x in in_arrays)
        # note that bbox is in original scale
        img, bbox, label, scale, difficult = in_vars
        bbox = chainer.cuda.to_cpu(bbox.data)[0]  # (M, 4)
        label = chainer.cuda.to_cpu(label.data)[0]  # (M,)
        difficult = chainer.cuda.to_cpu(difficult.data)[0]  # (M,)

        pred_bbox, pred_label, pred_confidence = model.predict(img, scale=scale)
        pred_bbox = chainer.cuda.to_cpu(pred_bbox)[0]  # (N, 4)
        pred_label = chainer.cuda.to_cpu(pred_label)[0]  # (N,)
        pred_confidence = chainer.cuda.to_cpu(pred_confidence)[0]  # (N,)

        bboxes.append(pred_bbox)
        labels.append(pred_label)
        confs.append(pred_confidence)

        if i % 100 == 0:
            print('currently processing {} / {}'.format(i, len(dataset)))
            img = chainer.cuda.to_cpu(img.data)
            img = img[0] + mean_pixel
            img = transforms.scale(
                img,
                int(min(img.shape[1:]) / chainer.cuda.to_cpu(scale[0].data)))
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            chainercv.tasks.vis_bbox(img, bbox, ax=ax1)
            ax2 = fig.add_subplot(1, 2, 2)
            chainercv.tasks.vis_bbox(img, pred_bbox, ax=ax2)
            # plt.show()
            plt.savefig('result/{}'.format(i))
            fig.clear()

        gt_bboxes.append(bbox)
        gt_labels.append(label)
        gt_difficults.append(difficult)
    return bboxes, labels, confs, gt_bboxes, gt_labels, gt_difficults


def transform(in_data):
    img, bbox, label, difficult = in_data
    img -= mean_pixel 
    # Resize bounding box to a shape
    # with the smaller edge at least at length 600
    _, H, W = img.shape
    img = transforms.scale(img, 600)
    _, o_H, o_W = img.shape
    # Prevent the biggest axis from being more than MAX_SIZE
    if max(o_H, o_W) > 1000:
        rate = 1000 / float(max(o_H, o_W))
        img = transforms.resize(img, (int(o_W * rate), int(o_H * rate)))
        _, o_H, o_W = img.shape
    # bbox = transforms.resize_bbox(bbox, (W, H), (o_W, o_H))

    # horizontally flip
    # img, params = transforms.random_flip(img, x_random=True, return_param=True)
    # bbox = transforms.flip_bbox(bbox, (o_W, o_H), params['x_flip'])
    return img, bbox, label, float(o_W) / float(W), difficult


def main(device=0, weight='', not_targets_precomputed=False):
    dataset = VOCDetectionDataset(mode='test', year='2007',
                                  use_difficult=True, return_difficult=True)
    labels = pascal_voc_labels
    dataset = TransformDataset(dataset, transform)
    model = FasterRCNNVGG(
        conf_threh=0.05, target_precomputed=not not_targets_precomputed)
    chainer.serializers.load_npz(weight, model)
    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    bboxes, labels, confs, gt_bboxes, gt_labels, gt_difficults = record_bbox(
        model, dataset, device, len(pascal_voc_labels))

    metric = eval_detection(
        bboxes, labels, confs, gt_bboxes,
        gt_labels, n_class=len(pascal_voc_labels),
        gt_difficults=gt_difficults,
        minoverlap=0.5, use_07_metric=True)
    print metric


if __name__ == '__main__':
    fire.Fire(main)
