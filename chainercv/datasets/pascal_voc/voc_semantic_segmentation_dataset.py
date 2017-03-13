import numpy as np
import os.path as osp
from PIL import Image

import chainer

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.utils import read_image_as_array


class VOCSemanticSegmentationDataset(chainer.dataset.DatasetMixin):

    """Dataset class for the semantic segmantion task of Pascal `VOC2012`_.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/pascal_voc`.
        mode ({'train', 'val', 'trainval'}): select from dataset splits used
            in VOC.
        year ({'2007', '2012'}): use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If true, use images that are labeled as
            difficult in the original annotation.

    """

    labels = voc_utils.pascal_voc_labels

    def __init__(self, data_dir='auto', mode='train'):
        if mode not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick mode from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_pascal_voc('2012', mode)

        id_list_file = osp.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(mode))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. Both of them are in CHW
        format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of color image and label whose shapes are (3, H, W) and
            (1, H, W) respectively. H and W are height and width of the
            images. The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        img_file = osp.join(self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image_as_array(img_file)
        label = self._load_label(self.data_dir, self.ids[i])

        img = img[:, :, ::-1]  # RGB to BGR
        img = img.transpose(2, 0, 1).astype(np.float32)
        label = label[None]
        return img, label

    def _load_label(self, data_dir, id_):
        label_rgb_file = osp.join(
            data_dir, 'SegmentationClass', id_ + '.png')
        im = Image.open(label_rgb_file)
        label = np.array(im, dtype=np.uint8).astype(np.int32)
        label[label == 255] = -1
        return label
