import argparse

import matplotlib.pyplot as plt

import chainer
from chainercv.datasets import cityscapes_semantic_segmentation_label_colors
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.experimental.links import PSPNetResNet101
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    parser.add_argument('--input-size', type=int, default=713)
    parser.add_argument('image')
    args = parser.parse_args()

    label_names = cityscapes_semantic_segmentation_label_names
    colors = cityscapes_semantic_segmentation_label_colors
    n_class = len(label_names)

    input_size = (args.input_size, args.input_size)
    model = PSPNetResNet101(n_class, args.pretrained_model, input_size)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    img = read_image(args.image)
    labels = model.predict([img])
    label = labels[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    vis_image(img, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2, legend_handles = vis_semantic_segmentation(
        img, label, label_names, colors, ax=ax2)
    ax2.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)

    plt.show()


if __name__ == '__main__':
    main()
