import argparse
import matplotlib.pyplot as plot
import numpy as np

import chainer

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu')
    parser.add_argument('image')
    args = parser.parse_args()

    model = FasterRCNNVGG16(n_class=21, pretrained_model='voc07')
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict(img[np.newaxis])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=voc_utils.pascal_voc_labels)
    plot.show()


if __name__ == '__main__':
    main()
