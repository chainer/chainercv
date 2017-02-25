import collections
import random

from chainer.utils import type_check

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class RandomMirrorWrapper(DatasetWrapper):
    """Crop array by `crop_width`  along each dimension.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        augment_idx (int or list of ints): this wrapper will augment k-th
            output of wrapped dataset's get_example if k is in `augment_idx`.
        orientation ({'h', 'v', 'both'}): chooses whether to mirror
            horizontally or vertically.
        hook (callable or `None`): The callable takes `out_data`,
            `h_mirror` and `v_mirror` as arguments. `h_mirror` is a bool
            that indicates whether a horizontal mirroring is carried out or
            not. `v_mirror` is a bool that indicates whether a vertical
            mirroring is carried out or not.
            If this is `None`, hook function is not called.

    """

    def __init__(self, dataset, augment_idx, orientation='h', hook=None):
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

        self.hook = hook

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
        h_mirror, v_mirror = False, False
        if 'h' in self.orientation:
            h_mirror = random.choice([True, False])
        if 'v' in self.orientation:
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

        if self.hook is not None:
            out_data = self.hook(out_data, h_mirror, v_mirror)
        return tuple(out_data)


def bbox_mirror_hook(img_idx=0, bboxes_idx=1):
    def _bbox_mirror_hook(out_data, h_mirror, v_mirror):
        img = out_data[img_idx]
        bboxes = out_data[bboxes_idx]

        _, H, W = img.shape
        if h_mirror:
            x_max = W - 1 - bboxes[:, 0]
            x_min = W - 1 - bboxes[:, 2]
            bboxes[:, 0] = x_min
            bboxes[:, 2] = x_max
        if v_mirror:
            y_max = H - 1 - bboxes[:, 1]
            y_min = H - 1 - bboxes[:, 3]
            bboxes[:, 1] = y_min
            bboxes[:, 3] = y_max

        out_data[bboxes_idx] = bboxes
        return out_data
    return _bbox_mirror_hook
