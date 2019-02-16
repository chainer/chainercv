import argparse

import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import DeepLabV3plusXception65
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='cityscapes')
    parser.add_argument('--min-input-size', type=int, default=None)
    parser.add_argument('image')
    args = parser.parse_args()

    model = DeepLabV3plusXception65(
        pretrained_model=args.pretrained_model,
        min_input_size=args.min_input_size)

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
    vis_semantic_segmentation(
        None, label, voc_semantic_segmentation_label_names,
        voc_semantic_segmentation_label_colors, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()
