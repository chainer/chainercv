import collections
import random

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class CropWrapper(DatasetWrapper):
    """Crop array into `cropped_shape`.

    If `start_idx` is None, this wrapper randomly crops images into one with
    shape `cropped_shape`.
    If `start_idx` is not None, returned images will be something like
    `image[start_idx[0]:start_idx[0] + cropped_shape[0], ...]`.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        augment_idx (int or list of ints): this wrapper will augment k-th
            output of wrapped dataset's get_example if k is in `augment_idx`.
        cropped_shape (tuple of ints): shape of data after cropping.
        start_idx (tuple of ints): If this is None, this wrapper randomly
            crops images.

    """

    def __init__(self, dataset, augment_idx, cropped_shape, start_idx=None):
        super(CropWrapper, self).__init__(dataset)

        self.cropped_shape = cropped_shape

        if not isinstance(augment_idx, collections.Iterable):
            augment_idx = (augment_idx,)
        self.augment_idx = augment_idx

        self.start_idx = start_idx

    def _get_example(self, in_data):
        """Returns the i-th example.

        All returned images are in CHW format.

        Args:
            in_data (tuple): The i-th example from the wrapped dataset.

        Returns:
            i-th example.

        """
        out_data = list(in_data)
        for idx in self.augment_idx:
            img = in_data[idx]
            shape = img.shape
            slices = []
            for i, (width, cropped_width) in enumerate(
                    zip(shape, self.cropped_shape)):
                if self.start_idx is None:
                    if width > cropped_width:
                        start_idx = random.choice(range(width - cropped_width))
                    elif width == cropped_width:
                        start_idx = 0
                    else:
                        raise ValueError('width of an input has to be larger '
                                         'than values in cropped_shape')
                else:
                    start_idx = self.start_idx[i]
                slices.append(slice(start_idx, start_idx + cropped_width))
            out_data[idx] = img[tuple(slices)]
        return tuple(out_data)
