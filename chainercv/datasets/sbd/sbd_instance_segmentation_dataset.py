import numpy as np
import os
import warnings


from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
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
            'SciPy is not installed in your environment,'
            'so the dataset cannot be loaded.'
            'Please install SciPy to load dataset.\n\n'
            '$ pip install scipy')


class SBDInstanceSegmentationDataset(GetterDataset):

    """Instance segmentation dataset for Semantic Boundaries Dataset `SBD`_.

    .. _`SBD`: http://home.bharathh.info/pubs/codes/SBD/download.html

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/sbd`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.


    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`mask`, ":math:`(R, H, W)`", :obj:`bool`, --
        :obj:`label`, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
    """

    def __init__(self, data_dir='auto', split='train'):
        super(SBDInstanceSegmentationDataset, self).__init__()

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

        self.add_getter('img', self._get_image)
        self.add_getter(('mask', 'label'), self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'img', data_id + '.jpg')
        return read_image(img_file, color=True)

    def _get_annotations(self, i):
        data_id = self.ids[i]
        label_img, inst_img = self._load_label_inst(data_id)
        mask, label = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        return mask, label

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
