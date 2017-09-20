import numpy as np
import os

from chainercv.datasets.cub.cub_utils import CUBDatasetBase
from chainercv import utils


class CUBLabelDataset(CUBDatasetBase):

    """`Caltech-UCSD Birds-200-2011`_ dataset  with annotated class labels.

    .. _`Caltech-UCSD Birds-200-2011`:
        http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    When queried by an index, this dataset returns a corresponding
    :obj:`img, label`, a tuple of an image and class id.
    The image is in RGB and CHW format.
    The class id are between 0 and 199.

    A bounding box is a one-dimensional array of shape :math:`(4,)`.
    The elements of the bounding box corresponds to
    :obj:`(y_min, x_min, y_max, x_max)`, where the four attributes are
    coordinates of the top left and the bottom right vertices.
    This information can optionally be retrieved from the dataset
    by setting :obj:`return_bb = True`.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        return_bb (bool): If :obj:`True`, this returns a bounding box
            around a bird. Default value is :obj:`False`.

    """

    def __init__(self, data_dir='auto', return_bb=False):
        super(CUBLabelDataset, self).__init__(
            data_dir=data_dir, return_bb=return_bb)

        image_class_labels_file = os.path.join(
            self.data_dir, 'image_class_labels.txt')
        labels = [int(d_label.split()[1]) - 1 for
                  d_label in open(image_class_labels_file)]
        self._labels = np.array(labels, dtype=np.int32)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and its label.

        """
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.paths[i]),
            color=True)
        label = self._labels[i]

        if self.return_bb:
            return img, label, self.bbs[i]
        return img, label
