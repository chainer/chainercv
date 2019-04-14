import numpy as np
import os

from chainercv.datasets.cub.cub_utils import CUBDatasetBase
from chainercv import utils


class CUBLabelDataset(CUBDatasetBase):

    """`Caltech-UCSD Birds-200-2011`_ dataset  with annotated class labels.

    .. _`Caltech-UCSD Birds-200-2011`:
        http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        return_bbox (bool): If :obj:`True`, this returns a bounding box
            around a bird. The default value is :obj:`False`.
        prob_map_dir (string): Path to the root of the probability maps.
            If this is :obj:`auto`, this class will automatically download data
            for you under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        return_prob_map (bool): Decide whether to include a probability map of
            the bird in a tuple served for a query. The default value is
            :obj:`False`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, scalar, :obj:`int32`, ":math:`[0, \#class - 1]`"
        :obj:`bbox` [#cub_label_1]_, ":math:`(1, 4)`", :obj:`float32`, \
            ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`prob_map` [#cub_label_2]_, ":math:`(H, W)`", :obj:`float32`, \
            ":math:`[0, 1]`"

    .. [#cub_label_1] :obj:`bb` indicates the location of a bird. \
        It is available if :obj:`return_bbox = True`.
    .. [#cub_label_2] :obj:`prob_map` indicates how likey a bird is located \
        at each the pixel. \
        It is available if :obj:`return_prob_map = True`.
    """

    def __init__(self, data_dir='auto', return_bbox=False,
                 prob_map_dir='auto', return_prob_map=False):
        super(CUBLabelDataset, self).__init__(data_dir, prob_map_dir)

        image_class_labels_file = os.path.join(
            self.data_dir, 'image_class_labels.txt')
        labels = [int(d_label.split()[1]) - 1 for
                  d_label in open(image_class_labels_file)]
        self._labels = np.array(labels, dtype=np.int32)

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

        keys = ('img', 'label')
        if return_bbox:
            keys += ('bbox',)
        if return_prob_map:
            keys += ('prob_map',)
        self.keys = keys

    def _get_image(self, i):
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.paths[i]),
            color=True)
        return img

    def _get_label(self, i):
        return self._labels[i]
