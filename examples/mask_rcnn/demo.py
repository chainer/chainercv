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
    parser.add_argument(
        '--model',
        choices=('mask_rcnn_fpn_resnet50', 'mask_rcnn_fpn_resnet101'),
        default='mask_rcnn_fpn_resnet50'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pretrained-model')
    group.add_argument('--snapshot')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'mask_rcnn_fpn_resnet50':
        model = MaskRCNNFPNResNet50(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'mask_rcnn_fpn_resnet101':
        model = MaskRCNNFPNResNet101(
            n_fg_class=len(coco_instance_segmentation_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image)
    masks, labels, scores = model.predict([img])
    mask = masks[0]
    label = labels[0]
    score = scores[0]
    chainercv.visualizations.vis_instance_segmentation(
        img, mask, label, score,
        label_names=coco_instance_segmentation_label_names)
    plt.show()


if __name__ == '__main__':
    main()
