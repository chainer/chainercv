import argparse
import matplotlib.pyplot as plt

import chainer

import chainercv
from chainercv.datasets import coco_bbox_label_names
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument(
        '--model', choices=('resnet50', 'resnet101'), default='resnet50')
    parser.add_argument('--pretrained-model')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'resnet50':
        model = FasterRCNNFPNResNet50(
            n_fg_class=len(coco_bbox_label_names))
    elif args.model == 'resnet101':
        model = FasterRCNNFPNResNet101(
            n_fg_class=len(coco_bbox_label_names))

    if args.pretrained_model:
        chainer.serializers.load_npz(args.pretrained_model, model)

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
