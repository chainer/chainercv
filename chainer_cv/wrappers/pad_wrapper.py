import numpy as np

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class PadWrapper(DatasetWrapper):

    """Pads image to fill given image size

    Args:
        dataset: a chainer.dataset.DatasetMixin to be wrapped
        max_size: the size of output image after padding

    """

    def __init__(self, dataset, max_size=(512, 512)):
        super(PadWrapper, self).__init__(dataset)
        self.max_size = max_size

    def _get_example(self, in_data):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example.

        """
        img, label = in_data

        x_slices, y_slices = self._get_pad_slices(img, max_size=self.max_size)
        out = np.zeros((3,) + self.max_size, dtype=np.float32)
        # unknown is -1
        out_label = np.ones((1,) + self.max_size, dtype=np.int32) * -1
        out[:, y_slices, x_slices] = img
        out_label[:, y_slices, x_slices] = label
        return out, out_label

    def _get_pad_slices(self, img, max_size=(500, 500)):
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


if __name__ == '__main__':
    from chainer_cv.datasets.pascal_voc_dataset import PascalVOCDataset
    import os.path as osp
    base_dir = osp.expanduser('~/datasets/VOC2012')
    dataset = PadWrapper(PascalVOCDataset(base_dir))

    img, label = dataset.get_example(0)
