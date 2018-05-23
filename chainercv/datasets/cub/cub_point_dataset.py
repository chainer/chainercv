import collections
import numpy as np
import os

from chainercv.datasets.cub.cub_utils import CUBDatasetBase


class CUBPointDataset(CUBDatasetBase):

    """`Caltech-UCSD Birds-200-2011`_ dataset  with annotated points.

    .. _`Caltech-UCSD Birds-200-2011`:
        http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

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

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`point`, ":math:`(P, 2)`", :obj:`float32`, ":math:`(y, x)`"
        :obj:`mask`, ":math:`(P,)`", :obj:`bool`, --
        :obj:`bb` [#cub_point_1]_, ":math:`(4,)`", :obj:`float32`, \
            ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`prob_map` [#cub_point_2]_, ":math:`(H, W)`", :obj:`float32`, \
            ":math:`[0, 1]`"

    .. [#cub_point_1] :obj:`bb` indicates the location of a bird. \
        It is available if :obj:`return_bb = True`.
    .. [#cub_point_2] :obj:`prob_map` indicates how likey a bird is located \
        at each the pixel. \
        It is available if :obj:`return_prob_map = True`.

    """

    def __init__(self, data_dir='auto', return_bb=False,
                 prob_map_dir='auto', return_prob_map=False):
        super(CUBPointDataset, self).__init__(data_dir, prob_map_dir)

        # load point
        parts_loc_file = os.path.join(self.data_dir, 'parts', 'part_locs.txt')
        self._point_dict = collections.defaultdict(list)
        self._mask_dict = collections.defaultdict(list)
        for loc in open(parts_loc_file):
            values = loc.split()
            id_ = int(values[0]) - 1

            # (y, x) order
            point = [float(v) for v in values[3:1:-1]]
            mask = bool(int(values[4]))

            self._point_dict[id_].append(point)
            self._mask_dict[id_].append(mask)

        self.add_getter(('point', 'mask'), self._get_annotations)

        keys = ('img', 'point', 'mask')
        if return_bb:
            keys += ('bb',)
        if return_prob_map:
            keys += ('prob_map',)
        self.keys = keys

    def _get_annotations(self, i):
        point = np.array(self._point_dict[i], dtype=np.float32)
        mask = np.array(self._mask_dict[i], dtype=np.bool)
        return point, mask
