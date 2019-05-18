import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import camvid_label_colors
from chainercv.datasets import camvid_label_names
from chainercv.links import SegNetBasic
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    parser.add_argument('--dataset', choices=('camvid'), default='camvid')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.dataset == 'camvid':
        if args.pretrained_model is None:
            args.pretrained_model = 'camvid'
        label_names = camvid_label_names
        colors = camvid_label_colors

    model = SegNetBasic(
        n_class=len(label_names),
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    labels = model.predict([img])
    label = labels[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    vis_image(img, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    # Do not overlay the label image on the color image
    vis_semantic_segmentation(None, label, label_names, colors, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()
