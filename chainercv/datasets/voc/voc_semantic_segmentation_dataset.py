import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class VOCSemanticSegmentationDataset(GetterDataset):

    """Semantic segmentation dataset for PASCAL `VOC2012`_.

    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_semantic_segmentation_label_names`.

    .. _`VOC2012`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval'}): Select a split of the dataset.

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

        self.add_getter('img', self.get_image)
        self.add_getter('label', self.get_label)

    def __len__(self):
        return len(self.ids)

    def get_image(self, i):
        """Returns the i-th image.

        Returns a color image. The color image is in CHW format.

        Args:
            i (int): The index of the example.

        Returns:
            A color image whose shape is (3, H, W). H and W are height
            and width of the images. The dtype of the color image is
            :obj:`numpy.float32`.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        img_path = os.path.join(
            self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def get_label(self, i):
        """Returns the label of the i-th example.

        Returns a label image. The label image is in HW format.

        Args:
            i (int): The index of the example.

        Returns:
            A label whose shape is  (H, W). H and W are height and width of the
            image. The dtype of the label image is :obj:`numpy.int32`.

        """
        label_path = os.path.join(
            self.data_dir, 'SegmentationClass', self.ids[i] + '.png')
        label = read_image(label_path, dtype=np.int32, color=False)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label[0]
