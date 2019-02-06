import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from chainercv.utils import read_label


class VOCSemanticSegmentationDataset(GetterDataset):

    """Semantic segmentation dataset for PASCAL `VOC2012`_.

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
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    """

    def __init__(self, data_dir='auto', split='train'):
        super(VOCSemanticSegmentationDataset, self).__init__()

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
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(
            self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def _get_label(self, i):
        label_path = os.path.join(
            self.data_dir, 'SegmentationClass', self.ids[i] + '.png')
        label = read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label
