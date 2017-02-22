import collections
import random

from chainer.utils import type_check

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class RandomMirrorWrapper(DatasetWrapper):
    """Crop array by `crop_width`  along each dimension.

    Args:
        dataset: a chainer.dataset.DatasetMixin to be wrapped.
        augment_idx (int or list of ints): this wrapper will augment k-th
            output of wrapped dataset's get_example if k is in `augment_idx`.
        orientation ({'h', 'v', 'both'}): chooses whether to mirror
            horizontally or vertically.
    """

    def __init__(self, dataset, augment_idx, orientation='h'):
        super(RandomMirrorWrapper, self).__init__(dataset)

        if orientation not in ['h', 'v', 'both']:
            raise ValueError('orientation has to be either \'h\', \'v\' or '
                             '\'both\'')
        if orientation == 'both':
            orientation = ['h', 'v']
        else:
            orientation = list(orientation)
        self.orientation = orientation

        if not isinstance(augment_idx, collections.Iterable):
            augment_idx = (augment_idx,)
        self.augment_idx = augment_idx

    def check_type_get_example(self, in_types):
        for idx in self.augment_idx:
            in_type = in_types[idx]
            type_check.expect(
                in_type.ndim == 3
            )

    def _get_example(self, in_data):
        """Returns the i-th example.

        All returned images are in CHW format.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example.

        """
        out_data = list(in_data)
        h_mirror = random.choice([True, False])
        v_mirror = random.choice([True, False])
        for idx in self.augment_idx:
            img = in_data[idx]
            if 'h' in self.orientation:
                if h_mirror:
                    img = img[:, :, ::-1]
            if 'v' in self.orientation:
                if v_mirror:
                    img = img[:, ::-1, :]
            out_data[idx] = img
        return tuple(out_data)
