import os
import warnings

from pkg_resources import get_distribution
from pkg_resources import parse_version

import numpy as np
try:
    import pykitti
    pykitti_version = get_distribution('pykitti').version
    if parse_version(pykitti_version) >= parse_version('0.3.0'):
        # pykitti>=0.3.0
        _available = True
    else:
        # pykitti<0.3.0
        warnings.warn('not support pykitti version : ' + pykitti_version)
        _available = False
except ImportError:
    _available = False

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.kitti.kitti_utils import get_kitti_label
from chainercv.datasets.kitti.kitti_utils import get_kitti_nosync_data
from chainercv.datasets.kitti.kitti_utils import get_kitti_sync_data
from chainercv.datasets.kitti.kitti_utils import get_kitti_tracklets
from chainercv.datasets.kitti.kitti_utils import kitti_date_lists
from chainercv.datasets.kitti.kitti_utils import kitti_date_num_dicts


def _check_available():
    if not _available:
        raise ValueError(
            'pykitti is not installed in your environment,'
            'so the dataset cannot be loaded.'
            'Please install pykitti to load dataset.\n\n'
            '$ pip install pykitti>=0.3.0')


class KITTIBboxDataset(GetterDataset):

    """Image dataset for test split of `KITTI dataset`_.

    .. _`KITTI dataset`: http://www.cvlibs.net/datasets/kitti/raw_data.php

    When queried by an index, if :obj:`return_crowded == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, mask, label, crowded, area`, a tuple of an image, bounding
    boxes, masks, labels, crowdness indicators and areas of masks.
    The parameters :obj:`return_crowded` and :obj:`return_area` decide
    whether to return :obj:`crowded` and :obj:`area`.
    :obj:`crowded` is a boolean array
    that indicates whether bounding boxes are for crowd labeling.
    When there are more than ten objects from the same category,
    bounding boxes correspond to crowd of instances instead of individual
    instances. Please see more detail in the Fig. ... (?) of the summary
    paper [#]_.

    .. [#] Andreas Geiger and Philip Lenz \
         and Christoph Stiller and Raquel Urtasun. \
        `Vision meets Robotics: The KITTI Dataset \
        <http://www.cvlibs.net/publications/Geiger2013IJRR.pdf>`_. \
        Geiger2013IJRR.


    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain the :obj:`---` directory. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/kitti`.
        date ({'2011_09_26', '2011_09_28', '2011_09_29',
                                           '2011_09_30', '2011_10_03'}):
            reference Calibration datas.
        drive_num ({'0xxx'}): get datas drive No.
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

    .. [#kitti_bbox_1] If :obj:`tracklet = True`, \
        :obj:`bbox` and :obj:`label` contain crowded instances.
    """

    def __init__(self, data_dir='auto', date='', drive_num='',
                 sync=True, is_left=True, tracklet=False):
        super(KITTIBboxDataset, self).__init__()

        _check_available()

        self.sync = sync
        self.is_left = is_left

        if date not in kitti_date_lists:
            raise ValueError('\'date\' argment must be one of the ' +
                             str(kitti_date_lists) + 'values.')

        # date(key map)
        # if drive_num not in ['0001', '0002', ...]:
        if drive_num not in kitti_date_num_dicts[date]:
            raise ValueError('\'drive_num\' argment must be one of the ' +
                             str(kitti_date_num_dicts[date]) + 'values.')

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

        if not os.path.exists(data_dir) or not os.path.exists(data_dir):
            raise ValueError(
                'kitti dataset does not exist at the expected location.'
                'Please download it from http://www.cvlibs.net/datasets/kitti/'
                'Then place directory at {}.'
                .format(os.path.join(data_dir, date + '_drive_' + drive_num)))

        # use pykitti
        self.dataset = pykitti.raw(
            data_dir, date, drive_num, frames=None, imformat='cv2')

        # current camera calibration R/P settings.
        if self.is_left is True:
            # img02
            self.cur_rotation_matrix = self.dataset.calib.R_rect_20
            self.cur_position_matrix = self.dataset.calib.P_rect_20
            # pykitti>=0.3.0
            # get PIL Image
            # convert from PIL.Image to numpy
            dataArray = []
            for cam2 in self.dataset.cam2:
                data = np.asarray(cam2)
                # Convert RGB to BGR
                if len(data.shape) > 2:
                    data = data[:, :, ::-1]
                dataArray.append(data)

            self.imgs = dataArray
            pass
        else:
            # img03
            self.cur_rotation_matrix = self.dataset.calib.R_rect_30
            self.cur_position_matrix = self.dataset.calib.P_rect_30
            # pykitti>=0.3.0
            # get PIL Image
            # convert from PIL.Image to numpy
            dataArray = []
            for cam2 in self.dataset.cam2:
                data = np.asarray(cam2)
                # Convert RGB to BGR
                if len(data.shape) > 2:
                    data = data[:, :, ::-1]
                dataArray.append(data)

            self.imgs = dataArray
            pass

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
            # pykitti img data
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1))

    def _get_annotations(self, i):
        bbox = self.bboxes
        label = self.labels

        # convert list to ndarray
        if len(bbox[i]) == 0:
            # NG
            # bbox[i] = [[0.0, 0.0, 0.0, 0.0]]
            # Data Padding(Pass Bbox Test)
            # bbox[i] = [[0.0, 0.0, 0.01, 0.01]]
            np_bbox = np.zeros((0, 4), dtype=np.float32)
        else:
            np_bbox = np.array(bbox[i], dtype=np.float32)

        if len(label[i]) == 0:
            # Data Padding(Pass Bbox Test)
            # label[i] = [0]
            np_label = np.zeros(0, dtype=np.int32)
        else:
            np_label = np.array(label[i], dtype=np.int32)

        return np_bbox, np_label
