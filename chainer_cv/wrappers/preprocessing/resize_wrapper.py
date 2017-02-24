import collections
import numpy as np
import skimage.transform

from chainer.utils import type_check

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class ResizeWrapper(DatasetWrapper):

    """Resize image to match a certain size.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        preprocess_idx (int or list of ints): this wrapper will preprocess k-th
            output of wrapped dataset's get_example if k is in
            `preprocess_idx`.
        output_shape (tuple or callable): When this is a tuple, itis the size
            of output image after padding. This needs to be in HWC format.
            In the case when this is a callable, this function should take
            input image's shape as an argument and returns the output shape.
            The image shape is organized as (height, width, channels).

    """

    def __init__(self, dataset, preprocess_idx, output_shape=None):
        super(ResizeWrapper, self).__init__(dataset)
        if not callable(output_shape) and len(output_shape) != 3:
            raise ValueError('output_shape needs to be of length 3')
        self.output_shape = output_shape
        if not isinstance(preprocess_idx, collections.Iterable):
            preprocess_idx = (preprocess_idx,)
        self.preprocess_idx = preprocess_idx

    def check_type_get_example(self, in_types):
        for idx in self.preprocess_idx:
            in_type = in_types[idx]
            type_check.expect(
                in_type.dtype.kind == 'f',
                in_type.ndim == 3
            )

    def _get_example(self, in_data):
        """Returns the i-th example.

        All returned images are in CHW format.

        Returns:
            i-th example.

        """
        out_data = list(in_data)
        for idx in self.preprocess_idx:
            img = in_data[idx]
            img = np.transpose(img, (1, 2, 0))
            if callable(self.output_shape):
                output_shape = self.output_shape(img.shape)
            else:
                output_shape = self.output_shape

            scale = np.max(np.abs(img))
            out_img = skimage.transform.resize(
                img / scale, output_shape).astype(img.dtype)
            out_data[idx] = out_img.transpose(2, 0, 1) * scale
        return tuple(out_data)


def output_shape_hard_max_soft_min(soft_min, hard_max):
    def output_shape(img_shape):
        lengths = np.array(img_shape[:2]).astype(np.float)
        min_length = np.min(lengths)
        scale = float(soft_min) / min_length
        lengths *= scale

        max_length = np.max(lengths)
        if max_length > hard_max:
            lengths *= float(hard_max) / max_length
        out_shape = (np.asscalar(lengths[0]),
                     np.asscalar(lengths[1]),
                     img_shape[2])
        return out_shape

    return output_shape
