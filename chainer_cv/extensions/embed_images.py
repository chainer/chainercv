import copy
import numpy as np
import os.path as osp

import chainer
from chainer.dataset import convert
from chainer.utils import type_check

from chainer_cv.extensions.utils import check_type
from chainer_cv.extensions.utils import forward


class EmbedImages(chainer.training.extension.Extension):
    """An extension that embeds images collected through an iterator.

    Through iterating an iterator, images will be embedded into some low
    dimensional space.

    The tuple returned by the iterator should have an image as its first index
    element. The embedded feature of the image will be saved.

    This extension will be executed in higher priority than any other default
    extensions.

    Args:
        iterator (chainer.iterators.Iterator): Dataset iterator for the
            images to be embedded. The element in the first index of the
            returned tuple will be image.
        target (chainer.Chain): model that embeds images to a feature space.
        embed_func (callable): This function will be called to embed batch
            of images.
        filename (string): NumPy array of embedded features will be saved as a
            file with this name.

    """

    invoke_before_training = False
    priority = 400  # higher than any other default priorities

    def __init__(self, iterator, target, embed_func=None,
                 filename='embed.npy'):
        self.iterator = iterator
        self.target = target
        if embed_func is None:
            embed_func = target
        self.embed_func = embed_func
        self.filename = filename

    @check_type
    def _check_type_dataset(self, in_types):
        img_type = in_types[0]
        type_check.expect(
            img_type.dtype.kind == 'f',
        )

    def __call__(self, trainer):
        iterator = copy.copy(self.iterator)
        embedded_feats = []
        for v in iterator:
            self._check_type_dataset(v[0])
            arrays = convert.concat_examples(
                v, device=chainer.cuda.get_device(self.target))
            h = forward(
                self.target, arrays[0:1], forward_func=self.embed_func)[0]
            embedded_feats.append(h)
        embedded_feats = np.concatenate(embedded_feats, axis=0)

        np.save(osp.join(trainer.out, self.filename),
                embedded_feats)
