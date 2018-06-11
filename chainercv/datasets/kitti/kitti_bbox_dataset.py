import os
import warnings

import numpy as np
try:
    import pykitti
    _available = True
except ImportError:
    _available = False

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.kitti.kitti_utils import get_kitti_label
from chainercv.datasets.kitti.kitti_utils import get_kitti_nosync_data
from chainercv.datasets.kitti.kitti_utils import get_kitti_sync_data
from chainercv.datasets.kitti.kitti_utils import get_kitti_tracklets


def _check_available():
    if not _available:
        warnings.warn(
            'pykitti is not installed in your environment,'
            'so the dataset cannot be loaded.'
            'Please install pykitti to load dataset.\n\n'
            '$ pip install pykitti==0.2.4')


class KITTIBboxDataset(GetterDataset):
    r"""Image dataset for test split of `KITTI dataset`_.

    .. _`KITTI dataset`: http://www.cvlibs.net/datasets/kitti/raw_data.php

    .. note::

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain the :obj:`---` directory. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/kitti`.
        date ({'2011_09_26', '2011_09_28', '2011_09_29',
                                           '2011_09_30', '2011_10_03'}):
            reference Calibration datas.
        drive_num ({'0xxx'}): get datas drive No.
        color (bool): use glay/color image.
        sync (bool): get timer sync/nosync data.
        is_left (bool): left/right camera image use 2type.
        tracklet (bool): 3d bblox data. date only 2011_09_26.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox` [#kitti_bbox_1]_, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label`, scalar, :obj:`int32`, ":math:`[0, \#class - 1]`"

    .. [#kitti_bbox_1] If :obj:`use_pykitty = False`, \
        :obj:`bbox` and :obj:`label` not contain instances.
    """

    def __init__(self, data_dir='auto', date='', drive_num='',
                 color=True, sync=True, is_left=True, tracklet=False):
        super(KITTIBboxDataset, self).__init__()

        _check_available()

        self.color = color
        self.sync = sync
        self.is_left = is_left
        if date == '2011_09_26':
            self.tracklet = tracklet
        else:
            self.tracklet = False

        if data_dir == 'auto':
            if sync is True:
                # download sync data
                data_dir = get_kitti_sync_data(
                    os.path.join('pfnet', 'chainercv', 'KITTI'),
                    date, drive_num, self.tracklet)
            else:
                # download nosync data
                data_dir = get_kitti_nosync_data(
                    os.path.join('pfnet', 'chainercv', 'KITTI'),
                    date, drive_num, self.tracklet)

        # use pykitti
        # read All images
        # imformat='None'
        # self.dataset = pykitti.raw(data_dir, date, drive_num, frames=None)
        self.dataset = pykitti.raw(
            data_dir, date, drive_num, frames=None, imformat='cv2')

        # current camera calibration R/P settings.
        if self.color is True:
            if self.is_left is True:
                # img02
                self.cur_rotation_matrix = self.dataset.calib.R_rect_20
                self.cur_position_matrix = self.dataset.calib.P_rect_20
                self.imgs = np.array(list(self.dataset.cam2))
            else:
                # img03
                self.cur_rotation_matrix = self.dataset.calib.R_rect_30
                self.cur_position_matrix = self.dataset.calib.P_rect_30
                self.imgs = np.array(list(self.dataset.cam3))
        else:
            if self.is_left is True:
                # img00
                self.cur_rotation_matrix = self.dataset.calib.R_rect_00
                self.cur_position_matrix = self.dataset.calib.P_rect_00
                self.imgs = np.array(list(self.dataset.cam0))
            else:
                # img01
                self.cur_rotation_matrix = self.dataset.calib.R_rect_10
                self.cur_position_matrix = self.dataset.calib.P_rect_10
                self.imgs = np.array(list(self.dataset.cam1))

        # get object info(type/area/bbox/...)
        if self.tracklet is True:
            self.tracklets = get_kitti_tracklets(data_dir, date, drive_num)
        else:
            self.tracklets = None

        self.bboxes, self.labels = get_kitti_label(
            self.tracklets, self.dataset.calib,
            self.cur_rotation_matrix, self.cur_position_matrix,
            self.__len__())

        self.add_getter('img', self._get_image)
        # self.add_getter('label', self._get_label)
        # self.add_getter('bbox', self._get_bbox)
        self.add_getter(['bbox', 'label'], self._get_annotations)
        keys = ('img', 'bbox', 'label')
        self.keys = keys

    def __len__(self):
        return len(self.imgs)

    def _get_image(self, i):
        img = self.imgs[i]
        # convert data is utils.read_image function return values
        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            return img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))

    def _get_annotations(self, i):
        # List[{'segmentation', 'area', 'iscrowd',
        #       'image_id', 'bbox', 'category_id', 'id'}]
        bbox = self.bboxes
        label = self.labels

        # convert list to ndarray
        if len(bbox[i]) == 0:
            # bbox[i] = np.zeros((0, 4), dtype=np.float32)
            bbox[i] = [[0.0, 0.0, 0.01, 0.01]]

        np_bbox = np.array(bbox[i], dtype=np.float32)

        if len(label[i]) == 0:
            # label[i] = np.zeros((0, 1), dtype=np.int32)
            label[i] = [0]

        np_label = np.array(label[i], dtype=np.int32)

        print(np_bbox)
        print(np_label)
        return np_bbox, np_label
