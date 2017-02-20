import collections
import random

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class RandomCropWrapper(DatasetWrapper):
    """Crop array by `crop_width`  along each dimension.

    Args:
        dataset: a chainer.dataset.DatasetMixin to be wrapped.
        augment_idx (int or list of ints): this wrapper will augment k-th
            output of wrapped dataset's get_example if k is in `augment_idx`.
        cropped_shape : (tuple of ints)
            shape of data after cropping.
    """

    def __init__(self, dataset, augment_idx, cropped_shape):
        super(RandomCropWrapper, self).__init__(dataset)

        self.cropped_shape = cropped_shape

        if not isinstance(augment_idx, collections.Iterable):
            augment_idx = (augment_idx,)
        self.augment_idx = augment_idx

    def get_example(self, i):
        """Returns the i-th example.

        All returned images are in CHW format.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example.

        """
        in_data = self.dataset[i]
        out_data = list(in_data)
        for idx in self.augment_idx:
            img = in_data[idx]
            shape = img.shape
            slices = []
            for width, cropped_width in zip(shape, self.cropped_shape):
                if width > cropped_width:
                    start_idx = random.choice(range(width - cropped_width))
                elif width == cropped_width:
                    start_idx = 0
                else:
                    raise ValueError('width of an input has to be larger than '
                                     'values in cropped_shape')
                slices.append(slice(start_idx, start_idx + cropped_width))
            out_data[idx] = img[tuple(slices)]
        return out_data
