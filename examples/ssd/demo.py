import argparse
import matplotlib.pyplot as plot
import numpy as np

from chainer import cuda

from chainercv.datasets import voc_detection_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(pretrained_model='voc0712')
    elif args.model == 'ssd512':
        model = SSD512(pretrained_model='voc0712')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict(img[np.newaxis])
    bbox = cuda.to_cpu(bboxes[0])
    label = cuda.to_cpu(labels[0])
    score = cuda.to_cpu(scores[0])

    vis_bbox(
        img, bbox, label, score, label_names=voc_detection_label_names)
    plot.show()


if __name__ == '__main__':
    main()
