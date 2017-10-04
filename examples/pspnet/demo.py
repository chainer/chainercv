import argparse
import os

import chainer
from chainercv.datasets import ade20k_semantic_segmentation_label_colors
from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import cityscapes_semantic_segmentation_label_colors
from chainercv.datasets import cityscapes_semantic_segmentation_label_names
from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import PSPNet
from chainercv.utils import read_image
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_label
import matplotlib.pyplot as plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_fn', '-f', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--scales', '-s', type=float, nargs='*', default=None)
    parser.add_argument('--out_dir', '-o', type=str, default='.')
    parser.add_argument('--model', '-m', type=str,
                        choices=['voc2012', 'cityscapes', 'ade20k'])
    args = parser.parse_args()

    chainer.config.train = False

    if args.model == 'voc2012':
        model = PSPNet(pretrained_model='voc2012')
        labels = voc_semantic_segmentation_label_names
        colors = voc_semantic_segmentation_label_colors
    elif args.model == 'cityscapes':
        model = PSPNet(pretrained_model='cityscapes')
        labels = cityscapes_semantic_segmentation_label_names
        colors = cityscapes_semantic_segmentation_label_colors
    elif args.model == 'ade20k':
        model = PSPNet(pretrained_model='ade20k')
        labels = ade20k_semantic_segmentation_label_names
        colors = ade20k_semantic_segmentation_label_colors

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    img = read_image(args.img_fn)
    pred = model.predict([img])[0]

    # Save the result image
    ax = vis_image(img)
    _, legend_handles = vis_label(pred, labels, colors, alpha=1.0, ax=ax)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.)
    base = os.path.splitext(os.path.basename(args.img_fn))[0]
    out_fn = os.path.join(args.out_dir, 'predict_{}.png'.format(base))
    plot.savefig(out_fn, bbox_inches='tight', dpi=400)
