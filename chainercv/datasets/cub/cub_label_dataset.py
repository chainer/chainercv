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

    There are 200 labels of birds in total.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        crop_bbox (bool): If true, this class returns an image cropped
            by the bounding box of the bird inside it.

    """

    def __init__(self, data_dir='auto', crop_bbox=True):
        super(CUBLabelDataset, self).__init__(
            data_dir=data_dir, crop_bbox=crop_bbox)

        image_class_labels_file = os.path.join(
            self.data_dir, 'image_class_labels.txt')
        self._data_labels = [int(d_label.split()[1]) - 1 for
                             d_label in open(image_class_labels_file)]

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and its label.

        """
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.fns[i]), color=True)

        if self.crop_bbox:
            # (y_min, x_min, y_max, x_max)
            bbox = self.bboxes[i].astype(np.int32)
            img = img[:, bbox[0]: bbox[2], bbox[1]: bbox[3]]
        label = self._data_labels[i]
        return img, label
