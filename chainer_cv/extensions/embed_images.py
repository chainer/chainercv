import collections
import numpy as np
import os.path as osp
from skimage.color import label2rgb
import warnings

import chainer
from chainer.utils import type_check
from chainer.dataset import convert

from chainer_cv.extensions.utils import forward, check_type


class EmbedImages(chainer.training.extension.Extension):
    """

    The tuple returned by iterator should have an image as its first index
    element. The embedded feature of the image will be saved.
    """

    invoke_before_training = False

    def __init__(self, iterator, target,
                 embed_func=None, filename='embed.npy'):
        self.iterator = iterator
        self.target = target
        if embed_func is None:
            embed_func = target
        self.embed_func = embed_func
        self.filename = filename

    def __call__(self, trainer):
        x = []
        embedded_feats = []
        for v in self.iterator:
            arrays = convert.concat_examples(
                v, device=chainer.cuda.get_device(self.target))
            h = forward(
                self.target, arrays[0], forward_func=self.embed_func)[0]
            embedded_feats.append(h)
        embedded_feats = np.concatenate(embedded_feats, axis=0)

        np.save(osp.join(trainer.out, self.filename),
                embedded_feats)
