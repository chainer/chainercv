import numpy as np
import os

import chainer

from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class VOCInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    """Instance segmentation dataset for PASCAL `VOC2012`_.

    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_instance_segmentation_label_names`.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

    """

    def __init__(self, data_dir='auto', split='train'):
        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_voc('2012', split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
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
            self.data_dir, 'JPEGImages', data_id + '.jpg')
        img = read_image(img_file, color=True)
        label_img, inst_img = self._load_label_inst(data_id)
        mask, label = voc_utils.image_wise_to_instance_wise(
            label_img, inst_img)
        return img, mask, label

    def _load_label_inst(self, data_id):
        label_file = os.path.join(
            self.data_dir, 'SegmentationClass', data_id + '.png')
        inst_file = os.path.join(
            self.data_dir, 'SegmentationObject', data_id + '.png')
        label_img = read_image(label_file, dtype=np.int32, color=False)
        label_img = label_img[0]
        label_img[label_img == 255] = -1
        inst_img = read_image(inst_file, dtype=np.int32, color=False)
        inst_img = inst_img[0]
        inst_img[inst_img == 0] = -1
        inst_img[inst_img == 255] = -1
        return label_img, inst_img
