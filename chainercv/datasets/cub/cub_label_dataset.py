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
    The class id is between 0 and 199.
    If :obj:`return_bb = True`, a bounding box :obj:`bb` is appended to the
    tuple.
    If :obj:`return_prob_map = True`, a probability map :obj:`prob_map` is
    appended.

    A bounding box is a one-dimensional array of shape :math:`(4,)`.
    The elements of the bounding box corresponds to
    :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the four attributes are
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
        super(CUBLabelDataset, self).__init__(
            data_dir=data_dir, return_bb=return_bb,
            prob_map_dir=prob_map_dir, return_prob_map=return_prob_map)

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
            The image is in CHW format and its color channel is ordered in
            RGB.
            If :obj:`return_bb = True`,
            a bounding box is appended to the returned value.
            If :obj:`return_mask = True`,
            a probability map is appended to the returned value.

        """
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.paths[i]),
            color=True)
        label = self._labels[i]

        if not self.return_prob_map:
            if self.return_bb:
                return img, label, self.bbs[i]
            else:
                return img, label

        prob_map = utils.read_image(self.prob_map_paths[i],
                                    dtype=np.uint8, color=False)
        prob_map = prob_map.astype(np.float32) / 255  # [0, 255] -> [0, 1]
        prob_map = prob_map[0]  # (1, H, W) --> (H, W)
        if self.return_bb:
            return img, label, self.bbs[i], prob_map
        else:
            return img, label, prob_map
