import argparse
import matplotlib.pyplot as plt

import chainer

import chainercv
from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv import utils

from chainercv.links import MaskRCNNFPNResNet101
from chainercv.links import MaskRCNNFPNResNet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', choices=('resnet50', 'resnet101'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pretrained-model')
    group.add_argument('--snapshot')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'resnet50':
        model = MaskRCNNFPNResNet50(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'resnet101':
        model = MaskRCNNFPNResNet101(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image)
    # bboxes, masks, labels, scores = model.predict([img])
    masks, labels, scores = model.predict([img])
    # bbox = bboxes[0]
    mask = masks[0]
    label = labels[0]
    score = scores[0]

    # chainercv.visualizations.vis_bbox(
    #     img, bbox, label, score, label_names=coco_bbox_label_names)

    import numpy as np
    # flag = np.array([bb[3] - bb[1] < 300 for bb in bbox], dtype=np.bool)
    flag = np.ones(len(mask), dtype=np.bool)
    chainercv.visualizations.vis_instance_segmentation(
        img, mask[flag], label[flag], score[flag],
        label_names=coco_instance_segmentation_label_names)
    plt.show()


if __name__ == '__main__':
    main()
