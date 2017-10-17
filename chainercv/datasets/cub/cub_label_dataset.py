import numpy as np
import os

from chainercv.datasets.cub.cub_utils import CUBDatasetBase


class CUBLabelDataset(CUBDatasetBase):

    """`Caltech-UCSD Birds-200-2011`_ dataset  with annotated class labels.

    .. _`Caltech-UCSD Birds-200-2011`:
        http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    When queried by an index, this dataset returns a corresponding
    :obj:`img, label`, a tuple of an image and class id.
    The image is in RGB and CHW format.
    The class id are between 0 and 199.
    If :obj:`return_bb = True`, a bounding box :obj:`bb` is appended to the
    tuple.
    If :obj:`return_prob_map = True`, a probability map :obj:`prob_map` is
    appended.

    A bounding box is a one-dimensional array of shape :math:`(4,)`.
    The elements of the bounding box corresponds to
    :obj:`(y_min, x_min, y_max, x_max)`, where the four attributes are
    coordinates of the top left and the bottom right vertices.
    This information can optionally be retrieved from the dataset
    by setting :obj:`return_bb = True`.

    The probability map of a bird shows how likely the bird is located at each
    pixel. If the value is close to 1, it is likely that the bird
    locates at that pixel. The shape of this array is :math:`(H, W)`,
    where :math:`H` and :math:`W` are height and width of the image
    respectively.
    This information can optionally be retrieved from the dataset
    by setting :obj:`return_prob_map = True`.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        return_bb (bool): If :obj:`True`, this returns a bounding box
            around a bird. The default value is :obj:`False`.
        prob_map_dir (string): Path to the root of the probability maps.
            If this is :obj:`auto`, this class will automatically download data
            for you under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        return_prob_map (bool): Decide whether to include a probability map of
            the bird in a tuple served for a query. The default value is
            :obj:`False`.

    """

    def __init__(self, data_dir='auto', return_bb=False,
                 prob_map_dir='auto', return_prob_map=False):
        super(CUBLabelDataset, self).__init__(data_dir, prob_map_dir)
        self.data_names = ('img', 'label')
        if return_bb:
            self.data_names += ('bb',)
        if return_prob_map:
            self.data_names += ('prob_map',)
        self.add_getter('label', self.get_label)

        image_class_labels_file = os.path.join(
            self.data_dir, 'image_class_labels.txt')
        labels = [int(d_label.split()[1]) - 1 for
                  d_label in open(image_class_labels_file)]
        self._labels = np.array(labels, dtype=np.int32)

    def get_label(self, i):
        """Returns the label of the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            A label.

        """
        return self._labels[i]
