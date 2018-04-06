import numpy as np
import os
import warnings

import chainer

from chainercv.datasets.sbd import sbd_utils
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image

try:
    import scipy
    _available = True
except ImportError:
    _available = False


def _check_available():
    if not _available:
        warnings.warn(
            'SciPy is not installed in your environment,',
            'so the dataset cannot be loaded.'
            'Please install SciPy to load dataset.\n\n'
            '$ pip install scipy')


class SBDInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    """Instance segmentation dataset for Semantic Boundaries Dataset `SBD`_.

    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.sbd_instance_segmentation_label_names`.

    .. _`SBD`: http://home.bharathh.info/pubs/codes/SBD/download.html

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/sbd`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        _check_available()

        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = sbd_utils.get_sbd()

        id_list_file = os.path.join(
            data_dir, '{}_voc2012.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image, bounding boxes, masks and labels. The color
        image is in CHW format.

        Args:
            i (int): The index of the example.

        Returns:
            A tuple of color image, masks and labels whose
            shapes are :math:`(3, H, W), (R, H, W), (R, )`
            respectively.
            :math:`H` and :math:`W` are height and width of the images,
            and :math:`R` is the number of objects in the image.
            The dtype of the color image is
            :obj:`numpy.float32`, that of the masks is :obj: `numpy.bool`,
            and that of the labels is :obj:`numpy.int32`.

        """
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'img', data_id + '.jpg')
        img = read_image(img_file, color=True)
        label_img, inst_img = self._load_label_inst(data_id)
        mask, label = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        return img, mask, label

    def _load_label_inst(self, data_id):
        label_file = os.path.join(
            self.data_dir, 'cls', data_id + '.mat')
        inst_file = os.path.join(
            self.data_dir, 'inst', data_id + '.mat')
        label_anno = scipy.io.loadmat(label_file)
        label_img = label_anno['GTcls']['Segmentation'][0][0].astype(np.int32)
        inst_anno = scipy.io.loadmat(inst_file)
        inst_img = inst_anno['GTinst']['Segmentation'][0][0].astype(np.int32)
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        return label_img, inst_img
