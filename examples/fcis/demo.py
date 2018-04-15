import argparse
import chainer

import matplotlib.pyplot as plt

from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.experimental.links import FCISResNet101
from chainercv.utils import mask_to_bbox
from chainercv.utils import read_image
from chainercv.visualizations.colormap import voc_colormap
from chainercv.visualizations import vis_bbox
from chainercv.visualizations import vis_instance_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='sbd')
    parser.add_argument('image')
    args = parser.parse_args()

    model = FCISResNet101(
        n_fg_class=20, pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = read_image(args.image, color=True)

    masks, labels, scores = model.predict([img])
    mask, label, score = masks[0], labels[0], scores[0]
    bbox = mask_to_bbox(mask)
    colors = voc_colormap(list(range(1, len(mask) + 1)))
    ax = vis_bbox(
        img, bbox, instance_colors=colors, alpha=0.5, linewidth=1.5)
    vis_instance_segmentation(
        None, mask, label, score,
        label_names=sbd_instance_segmentation_label_names,
        instance_colors=colors, alpha=0.7, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
