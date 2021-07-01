import collections
import numpy as np
import os

from chainercv.datasets.cub.cub_utils import CUBDatasetBase
from chainercv import utils


class CUBKeypointDataset(CUBDatasetBase):

    """`Caltech-UCSD Birds-200-2011`_ dataset  with annotated points.

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
        :obj:`point`, ":math:`(1, 15, 2)`", :obj:`float32`, ":math:`(y, x)`"
        :obj:`visible`, ":math:`(1, 15)`", :obj:`bool`, --
        :obj:`bbox` [#cub_point_1]_, ":math:`(1, 4)`", :obj:`float32`, \
            ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`prob_map` [#cub_point_2]_, ":math:`(H, W)`", :obj:`float32`, \
            ":math:`[0, 1]`"

    .. [#cub_point_1] :obj:`bb` indicates the location of a bird. \
        It is available if :obj:`return_bbox = True`.
    .. [#cub_point_2] :obj:`prob_map` indicates how likey a bird is located \
        at each the pixel. \
        It is available if :obj:`return_prob_map = True`.

    """

    def __init__(self, data_dir='auto', return_bbox=False,
                 prob_map_dir='auto', return_prob_map=False):
        super(CUBKeypointDataset, self).__init__(data_dir, prob_map_dir)

        # load point
        parts_loc_file = os.path.join(self.data_dir, 'parts', 'part_locs.txt')
        self._point_dict = collections.defaultdict(list)
        self._visible_dict = collections.defaultdict(list)
        for loc in open(parts_loc_file):
            values = loc.split()
            id_ = int(values[0]) - 1

            # (y, x) order
            point = [float(v) for v in values[3:1:-1]]
            mask = bool(int(values[4]))

            self._point_dict[id_].append(point)
            self._visible_dict[id_].append(mask)

        self.add_getter(('img', 'point', 'visible'),
                        self._get_img_and_annotations)

        keys = ('img', 'point', 'visible')
        if return_bbox:
            keys += ('bbox',)
        if return_prob_map:
            keys += ('prob_map',)
        self.keys = keys

    def _get_img_and_annotations(self, i):
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.paths[i]),
            color=True)

        pnt = np.array(self._point_dict[i], dtype=np.float32)
        vsble = np.array(self._visible_dict[i], dtype=np.bool)

        _, H, W = img.shape
        invisible = np.logical_or(
            np.logical_or(pnt[:, 0] > H, pnt[:, 1] > W),
            np.any(pnt < 0, axis=1))
        vsble[invisible] = False
        return img, pnt[None], vsble[None]
