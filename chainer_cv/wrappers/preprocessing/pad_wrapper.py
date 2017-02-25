import collections
import numpy as np

from chainer.utils import type_check

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class PadWrapper(DatasetWrapper):

    """Pads image to fill given image size.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        max_size (tuple of two ints): the size of output image after
            padding (max_H, max_W).
        preprocess_idx (int or list of ints): this wrapper will preprocess k-th
            output of wrapped dataset's get_example if k is in
            `preprocess_idx`.
        bg_values (dict {int: int}): The key corresponds to the index of the
            output of `get_example`. The value determines the background
            values for those outputs.

    """

    def __init__(self, dataset, max_size, preprocess_idx, bg_values=0):
        super(PadWrapper, self).__init__(dataset)
        self.max_size = max_size
        if not isinstance(preprocess_idx, collections.Iterable):
            preprocess_idx = (preprocess_idx,)
        self.preprocess_idx = preprocess_idx
        if not isinstance(bg_values, dict):
            bg_values = {key: bg_values for key in preprocess_idx}
        self.bg_values = bg_values

        # error check
        if any([key not in preprocess_idx for key in bg_values.keys()]):
            raise ValueError(
                'bg_values contains keys that are not in preprocess_idx')

    def check_type_get_example(self, in_types):
        for idx in self.preprocess_idx:
            in_type = in_types[idx]
            type_check.expect(
                in_type.ndim == 3
            )

    def _get_example(self, in_data):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example.

        """
        out_data = list(in_data)
        x_slices, y_slices = None, None
        for idx in self.preprocess_idx:
            img = in_data[idx]
            if x_slices is None and y_slices is None:
                x_slices, y_slices = self._get_pad_slices(
                    img, max_size=self.max_size)
            ones = np.ones((img.shape[0],) + self.max_size, dtype=img.dtype)
            out_data[idx] = self.bg_values[idx] * ones
            out_data[idx][:, y_slices, x_slices] = img

        return tuple(out_data)

    def _get_pad_slices(self, img, max_size):
        """Get slices needed for padding.

        Args:
            img (numpy.ndarray): this image is in format CHW.
            max_size (tuple of two ints): (max_H, max_W).
        """
        _, H, W = img.shape

        if H < max_size[0]:
            diff_y = max_size[0] - H
            margin_y = diff_y / 2
            if diff_y % 2 == 0:
                y_slices = slice(margin_y, max_size[0] - margin_y)
            else:
                y_slices = slice(margin_y, max_size[0] - margin_y - 1)
        else:
            y_slices = slice(0, max_size[0])

        if W < max_size[1]:
            diff_x = max_size[1] - W
            margin_x = diff_x / 2
            if diff_x % 2 == 0:
                x_slices = slice(margin_x, max_size[1] - margin_x)
            else:
                x_slices = slice(margin_x, max_size[1] - margin_x - 1)
        else:
            x_slices = slice(0, max_size[1])
        return x_slices, y_slices
