import argparse
import matplotlib.pyplot as plot
import numpy as np

import chainer

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.links import SSD300
from chainercv import tasks
from chainercv import transforms
from chainercv import utils


def main():
    parser = argparse.ArgumentParser()
    # GPU mode is not testedx
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('image')
    args = parser.parse_args()

    # load pre-trained model
    model = SSD300()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # convert image to CHW, BGR and float32
    img = utils.read_image_as_array(args.image) \
               .transpose(2, 0, 1)[::-1] \
               .astype(np.float32)
    bbox, label, _ = model.predict(img, min_conf=0.6)

    vis_img = transforms.chw_to_pil_image(img)
    tasks.vis_bbox(vis_img, bbox, label, voc_utils.pascal_voc_labels)
    plot.show()


if __name__ == '__main__':
    main()
