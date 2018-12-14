import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import coco_bbox_label_names
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=('faster_rcnn_fpn_resnet50', 'faster_rcnn_fpn_resnet101'),
        default='faster_rcnn_fpn_resnet50')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='coco')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'faster_rcnn_fpn_resnet50':
        model = FasterRCNNFPNResNet50(
            n_fg_class=len(coco_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'faster_rcnn_fpn_resnet101':
        model = FasterRCNNFPNResNet101(
            n_fg_class=len(coco_bbox_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image)
    bboxes, labels, scores = model.predict([img])
    bbox = bboxes[0]
    label = labels[0]
    score = scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=coco_bbox_label_names)
    plt.show()


if __name__ == '__main__':
    main()
