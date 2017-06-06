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
    If :obj:`return_mask = True`, :obj:`mask` will be returned as well,
    making the returned tuple to be of length four. :obj:`mask` is a
    :obj:`uint8` image which indicates the region of the image
    where a bird locates.

    keypoints are packed into a two dimensional array of shape
    :math:`(K, 2)`, where :math:`K` is the number of keypoints.
    Note that :math:`K=15` in CUB dataset. Also note that not all fifteen
    keypoints are visible in an image. When a keypoint is not visible,
    the values stored for that keypoint are undefined. The second axis
    corresponds to the :math:`x` and :math:`y` coordinates of the
    keypoints in the image.

    A keypoint mask array indicates whether a keypoint is visible in the
    image or not. This is a boolean array of shape :math:`(K,)`.

    A mask image of the bird shows how likely the bird is located at a
    given pixel. If the value is close to 255, more likely that a bird
    locates at that pixel. The shape of this array is :math:`(1, H, W)`,
    where :math:`H` and :math:`W` are height and width of the image
    respectively.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        crop_bbox (bool): If true, this class returns an image cropped
            by the bounding box of the bird inside it.
        mask_dir (string): Path to the root of the mask data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        return_mask (bool): Decide whether to include mask image of the bird
            in a tuple served for a query.

    """

    def __init__(self, data_dir='auto', crop_bbox=True,
                 mask_dir='auto', return_mask=False):
        super(CUBKeypointDataset, self).__init__(
            data_dir=data_dir, crop_bbox=crop_bbox)
        self.return_mask = return_mask

        # load keypoint
        parts_loc_file = os.path.join(self.data_dir, 'parts/part_locs.txt')
        self.kp_dict = collections.OrderedDict()
        self.kp_mask_dict = collections.OrderedDict()
        for loc in open(parts_loc_file):
            values = loc.split()
            id_ = int(values[0]) - 1

            if id_ not in self.kp_dict:
                self.kp_dict[id_] = []
            if id_ not in self.kp_mask_dict:
                self.kp_mask_dict[id_] = []

            keypoint = [float(v) for v in values[2:4]]
            kp_mask = bool(int(values[4]))

            self.kp_dict[id_].append(keypoint)
            self.kp_mask_dict[id_].append(kp_mask)

    def __len__(self):
        return len(self.fns)

    def get_example(self, i):
        # this i is transformed to id for the entire dataset
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.fns[i]),
            color=True)
        keypoint = np.array(self.kp_dict[i], dtype=np.float32)
        kp_mask = np.array(self.kp_mask_dict[i], dtype=np.bool)

        if self.crop_bbox:
            bbox = self.bboxes[i]  # (x, y, width, height)
            img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
            keypoint[:, :2] = keypoint[:, :2] - np.array([bbox[0], bbox[1]])

        if not self.return_mask:
            return img, keypoint, kp_mask

        mask = utils.read_image(
            os.path.join(self.mask_dir, self.fns[i][:-4] + '.png'),
            dtype=np.uint8,
            color=False)
        if self.crop_bbox:
            mask = mask[:,
                        bbox[1]: bbox[1] + bbox[3],
                        bbox[0]: bbox[0] + bbox[2]]

        return img, keypoint, kp_mask, mask
