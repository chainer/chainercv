import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    parser.add_argument(
        '--dataset', choices=('voc'), default='voc')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.dataset == 'voc':
        if args.pretrained_model is None:
            args.pretrained_model = 'voc07'
        label_names = voc_bbox_label_names

    model = FasterRCNNVGG16(
        pretrained_model=args.pretrained_model,
        **FasterRCNNVGG16.preset_params[args.dataset])

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=label_names)
    plt.show()


if __name__ == '__main__':
    main()
