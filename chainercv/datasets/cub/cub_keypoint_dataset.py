import collections
import numpy as np
import os

from chainercv.datasets.cub.cub_utils import CUBDatasetBase
from chainercv import utils


class CUBKeypointDataset(CUBDatasetBase):

    """`Caltech-UCSD Birds-200-2011`_ dataset  with annotated keypoints.

    .. _`Caltech-UCSD Birds-200-2011`:
        http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    An index corresponds to each image.

    When queried by an index, this dataset returns the corresponding
    :obj:`img, keypoint, kp_mask`, a tuple of an image, keypoints
    and a keypoint mask that indicates visible keypoints in the image.
    The data type of the three elements are :obj:`float32, float32, bool`.
    If :obj:`return_bb = True`, a bounding box :obj:`bb` is appended to the
    tuple.
    If :obj:`return_prob_map = True`, a probability map :obj:`prob_map` is
    appended.

    keypoints are packed into a two dimensional array of shape
    :math:`(K, 2)`, where :math:`K` is the number of keypoints.
    Note that :math:`K=15` in CUB dataset. Also note that not all fifteen
    keypoints are visible in an image. When a keypoint is not visible,
    the values stored for that keypoint are undefined. The second axis
    corresponds to the :math:`y` and :math:`x` coordinates of the
    keypoints in the image.

    A keypoint mask array indicates whether a keypoint is visible in the
    image or not. This is a boolean array of shape :math:`(K,)`.

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
        super(CUBKeypointDataset, self).__init__(
            data_dir=data_dir, return_bb=return_bb,
            prob_map_dir=prob_map_dir, return_prob_map=return_prob_map)

        # load keypoint
        parts_loc_file = os.path.join(self.data_dir, 'parts', 'part_locs.txt')
        self.kp_dict = collections.OrderedDict()
        self.kp_mask_dict = collections.OrderedDict()
        for loc in open(parts_loc_file):
            values = loc.split()
            id_ = int(values[0]) - 1

            if id_ not in self.kp_dict:
                self.kp_dict[id_] = list()
            if id_ not in self.kp_mask_dict:
                self.kp_mask_dict[id_] = list()

            # (y, x) order
            keypoint = [float(v) for v in values[3:1:-1]]
            kp_mask = bool(int(values[4]))

            self.kp_dict[id_].append(keypoint)
            self.kp_mask_dict[id_].append(kp_mask)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image, keypoints and a keypoint mask.
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
        keypoint = np.array(self.kp_dict[i], dtype=np.float32)
        kp_mask = np.array(self.kp_mask_dict[i], dtype=np.bool)

        if not self.return_prob_map:
            if self.return_bb:
                return img, keypoint, kp_mask, self.bbs[i]
            else:
                return img, keypoint, kp_mask

        prob_map = utils.read_image(self.prob_map_paths[i],
                                    dtype=np.uint8, color=False)
        prob_map = prob_map.astype(np.float32) / 255  # [0, 255] -> [0, 1]
        prob_map = prob_map[0]  # (1, H, W) --> (H, W)
        if self.return_bb:
            return img, keypoint, kp_mask, self.bbs[i], prob_map
        else:
            return img, keypoint, kp_mask, prob_map
