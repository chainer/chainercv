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
        hook (dict {int: callable}): The key corresponds to the index of the
            output of `get_example`. For the keys included in `hooks`, the
            callables takes `idx`, `out_data`,
            `input_shape` and `output_shape` as arguments.
            `input_shape` is the shape of the image before resizing.
            `output_shape` is the shape of the image after resizing.
            If this is `None`, no hook functions will be called.

    """

    def __init__(self, dataset, preprocess_idx, output_shape=None, hook=None):
        super(ResizeWrapper, self).__init__(dataset)
        if not callable(output_shape) and len(output_shape) != 3:
            raise ValueError('output_shape needs to be of length 3')
        self.output_shape = output_shape
        if not isinstance(preprocess_idx, collections.Iterable):
            preprocess_idx = (preprocess_idx,)
        self.preprocess_idx = preprocess_idx

        self.hook = hook

    def check_type_get_example(self, in_types):
        for idx in self.preprocess_idx:
            in_type = in_types[idx]
            type_check.expect(
                in_type.ndim == 3
            )

    def _get_example(self, in_data):
        """Returns the i-th example.

        All returned images are in CHW format.

        Returns:
            i-th example.

        """
        out_data = list(in_data)
        input_shape, output_shape = None, None
        for idx in self.preprocess_idx:
            img = in_data[idx]
            img = np.transpose(img, (1, 2, 0))
            if input_shape is None and output_shape is None:
                input_shape = img.shape
                if callable(self.output_shape):
                    output_shape = self.output_shape(input_shape)
                else:
                    output_shape = self.output_shape
            if input_shape != img.shape:
                raise ValueError('shape of images in arguments can not vary')

            scale = np.max(np.abs(img))
            out_img = skimage.transform.resize(
                img / scale, output_shape).astype(img.dtype)
            out_data[idx] = out_img.transpose(2, 0, 1) * scale

        if self.hook is not None:
            out_data = self.hook(out_data, input_shape, output_shape)
        return tuple(out_data)


def output_shape_soft_min_hard_max(soft_min, hard_max):
    """Callable output shape

    This returns an output shape whose maximum axis is always smaller than
    `hard_max`. This tries to return an output shape whose minimum axis is
    equal to `soft_min`.

    """

    def output_shape(img_shape):
        lengths = np.array(img_shape[:2]).astype(np.float)
        min_length = np.min(lengths)
        scale = float(soft_min) / min_length
        lengths *= scale

        max_length = np.max(lengths)
        if max_length > hard_max:
            lengths *= float(hard_max) / max_length
        out_shape = (int(np.asscalar(lengths[0])),
                     int(np.asscalar(lengths[1])),
                     img_shape[2])
        return out_shape

    return output_shape


def bbox_resize_hook(idx=1):
    def _bbox_resize_hook(out_data, input_shape, output_shape):
        """A hook that resizes bounding boxes according to resizing

        Args:
            out_data (list)
            input_shape: shape of an array in HWC format.
            output_shape: shape of an array in HWC format.

        """
        bboxes = out_data[idx]
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 5

        scale = float(output_shape[0]) / input_shape[0]
        if abs(scale - (float(output_shape[1]) / input_shape[1])) > 0.05:
            raise ValueError('resizing has to preserve the aspect ratio.')

        bboxes[:, :4] = (scale * bboxes[:, :4]).astype(bboxes.dtype)
        out_data[idx] = bboxes
        return out_data
    return _bbox_resize_hook
