import argparse
import copy
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import ade20k_semantic_segmentation_label_colors
from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import cityscapes_semantic_segmentation_label_colors
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import DeepLabV3plusXception65
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model')
    parser.add_argument('--min-input-size', type=int, default=None)
    parser.add_argument(
        '--dataset', choices=('cityscapes', 'ade20k', 'voc'),
        default='cityscapes')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.dataset == 'cityscapes':
        if args.pretrained_model is None:
            args.pretrained_model = 'cityscapes'
        label_names = cityscapes_semantic_segmentation_label_names
        colors = cityscapes_semantic_segmentation_label_colors
    elif args.dataset == 'ade20k':
        if args.pretrained_model is None:
            args.pretrained_model = 'ade20k'
        label_names = ade20k_semantic_segmentation_label_names
        colors = ade20k_semantic_segmentation_label_colors
    elif args.dataset == 'voc':
        if args.pretrained_model is None:
            args.pretrained_model = 'voc'
        label_names = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors

    params = copy.deepcopy(DeepLabV3plusXception65.preset_params[args.dataset])
    params['min_input_size'] = args.min_input_size
    model = DeepLabV3plusXception65(
        pretrained_model=args.pretrained_model, **params)

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
        None, label, label_names, colors, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()
