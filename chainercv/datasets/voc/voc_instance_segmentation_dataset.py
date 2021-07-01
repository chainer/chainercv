import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label


class VOCInstanceSegmentationDataset(GetterDataset):

    """Instance segmentation dataset for PASCAL `VOC2012`_.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
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
        super(VOCInstanceSegmentationDataset, self).__init__()

        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2012', split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

        self.add_getter('img', self._get_image)
        self.add_getter(('mask', 'label'), self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        data_id = self.ids[i]
        img_file = os.path.join(
            self.data_dir, 'JPEGImages', data_id + '.jpg')
        return read_image(img_file, color=True)

    def _get_annotations(self, i):
        data_id = self.ids[i]
        label_img, inst_img = self._load_label_inst(data_id)
        mask, label = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        return mask, label

    def _load_label_inst(self, data_id):
        label_file = os.path.join(
            self.data_dir, 'SegmentationClass', data_id + '.png')
        inst_file = os.path.join(
            self.data_dir, 'SegmentationObject', data_id + '.png')
        label_img = read_label(label_file, dtype=np.int32)
        label_img[label_img == 255] = -1
        inst_img = read_label(inst_file, dtype=np.int32)
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        return label_img, inst_img
