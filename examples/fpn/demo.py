import argparse
import matplotlib.pyplot as plt

import chainer

import chainercv
from chainercv.datasets import coco_bbox_label_names
from chainercv import utils

from fpn import FasterRCNNFPNResNet101
from fpn import FasterRCNNFPNResNet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', choices=('resnet50', 'resnet101'))
    parser.add_argument(
        '--mean', choices=('chainercv', 'detectron'), default='chainercv')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pretrained-model')
    group.add_argument('--snapshot')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'resnet50':
        model = FasterRCNNFPNResNet50(n_fg_class=len(coco_bbox_label_names),
                                      mean=args.mean)
    elif args.model == 'resnet101':
        model = FasterRCNNFPNResNet101(n_fg_class=len(coco_bbox_label_names),
                                       mean=args.mean)

    if args.pretrained_model:
        chainer.serializers.load_npz(args.pretrained_model, model)
    elif args.snapshot:
        chainer.serializers.load_npz(
            args.snapshot, model, path='updater/model:main/model/')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image)
    bboxes, labels, scores = model.predict([img])
    bbox = bboxes[0]
    label = labels[0]
    score = scores[0]

    chainercv.visualizations.vis_bbox(
        img, bbox, label, score, label_names=coco_bbox_label_names)
    plt.show()


if __name__ == '__main__':
    main()
