import argparse
import matplotlib.pyplot as plot

import chainer

from chainercv.datasets import voc_detection_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc0712')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_detection_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_detection_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=voc_detection_label_names)
    plot.show()


if __name__ == '__main__':
    main()
