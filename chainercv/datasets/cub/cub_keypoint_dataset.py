import collections
import numpy as np
import os.path as osp

from chainercv.datasets.cub.cub_utils import CUBDatasetBase
from chainercv import utils


class CUBKeypointDataset(CUBDatasetBase):

    """Dataset class for `CUB-200-2011`_ with keypoint as supervision data.

    .. _`CUB-200-2011`:
        http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    This dataset has annotations of 15 keypoint of birds for each image.
    Note that not all keypoint are visible in an image. In that case,
    :obj:`valid` value is :math:`0`.
    The shape of keypoint is :math:`(15, 3)`. The last dimension is
    composed of :obj:`(x, y, valid)` in this order. :obj:`x` and
    :obj:`y` are coordinates of a keypoint. :obj:`valid` is whether the
    keypoint is visible in the image or not.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/cub`.
        mode ({`train`, `test`}): Select train or test split used in
            [Kanazawa]_.
        crop_bbox (bool): If true, this class returns an image cropped
            by the bounding box of the bird inside it.

    .. [Kanazawa] Angjoo Kanazawa, David W. Jacobs, \
       Manmohan Chandraker. WarpNet: Weakly Supervised Matching for \
       Single-view Reconstruction. https://arxiv.org/abs/1604.05592.

    """

    def __init__(self, data_dir='auto', mode='train',
                 crop_bbox=True):
        super(CUBKeypointDataset, self).__init__(
            data_dir=data_dir, crop_bbox=crop_bbox)

        # set mode
        test_images = np.load(
            osp.join(osp.split(osp.split(osp.abspath(__file__))[0])[0],
                     'data/cub_keypoint_dataset_test_image_ids.npy'))
        # the original one has ids starting from 1
        test_images = test_images - 1
        train_images = np.setdiff1d(np.arange(len(self.fns)), test_images)
        if mode == 'train':
            self.selected_ids = train_images
        elif mode == 'test':
            self.selected_ids = test_images
        else:
            raise ValueError('invalid mode')

        # load keypoint
        parts_loc_file = osp.join(self.data_dir, 'parts/part_locs.txt')
        keypoint_dict = collections.OrderedDict()
        for loc in open(parts_loc_file):
            values = loc.split()
            id_ = int(values[0]) - 1
            if id_ not in keypoint_dict:
                keypoint_dict[id_] = []
            keypoint = [float(v) for v in values[2:]]
            keypoint_dict[id_].append(keypoint)
        self.keypoint_dict = keypoint_dict

    def __len__(self):
        return len(self.selected_ids)

    def get_example(self, i):
        # this i is transformed to id for the entire dataset
        original_idx = self.selected_ids[i]
        img = utils.read_image_as_array(osp.join(
            self.data_dir, 'images', self.fns[original_idx]))  # RGB
        keypoint = self.keypoint_dict[original_idx]
        keypoint = np.array(keypoint, dtype=np.float32)

        if self.crop_bbox:
            bbox = self.bboxes[original_idx]  # (x, y, width, height)
            img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
            keypoint[:, :2] = keypoint[:, :2] - np.array([bbox[0], bbox[1]])

        if img.ndim == 2:
            img = utils.gray2rgb(img)

        img = img[:, :, ::-1]  # RGB to BGR
        img = img.transpose(2, 0, 1).astype(np.float32)
        return img, keypoint


if __name__ == '__main__':
    dataset = CUBKeypointDataset()

    from chainercv.tasks import vis_verts_pairs
    import matplotlib.pyplot as plt

    for i in range(200, 220):
        src_img, src_keys = dataset[2 * i]
        dst_img, dst_keys = dataset[2 * i + 1]
        keys = np.stack([src_keys, dst_keys], axis=1)
        vis_verts_pairs(src_img.transpose(1, 2, 0),
                        dst_img.transpose(1, 2, 0), keys)
        plt.show()
