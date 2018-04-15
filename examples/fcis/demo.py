import argparse
import chainer

import matplotlib.pyplot as plt

from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.experimental.links import FCISResNet101
from chainercv import utils
from chainercv.visualizations.colormap import voc_colormap
from chainercv.visualizations import vis_instance_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='sbd')
    parser.add_argument('image')
    args = parser.parse_args()

    model = FCISResNet101(pretrained_model=args.pretrained_model)
    model.to_gpu()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)

    masks, labels, scores = model.predict([img])
    mask, label, score = masks[0], labels[0], scores[0]
    colors = voc_colormap(list(range(1, len(mask) + 1)))
    vis_instance_segmentation(
        img, mask, label, score,
        label_names=sbd_instance_segmentation_label_names,
        instance_colors=colors, alpha=0.7)
    plt.show()


if __name__ == '__main__':
    main()
