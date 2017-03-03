import numpy as np
import os.path as osp
from PIL import Image
from skimage.io import imread

import chainer

import voc_utils


class VOCSemanticSegmentationDataset(chainer.dataset.DatasetMixin):

    """Dataset class for the semantic segmantion task of Pascal VOC2012.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            'auto', this class will automatically download data for you
            under ``$CHAINER_DATASET_ROOT/yuyu2172/chainer-cv/pascal_voc``.
        bgr (bool): If true, method `get_example` will return an image in BGR
            format.
    """

    labels = voc_utils.pascal_voc_labels

    def __init__(self, data_dir='auto', mode='train', bgr=True):
        if mode not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick mode from \'train\', \'trainval\', \'val\'')

        if data_dir == 'auto':
            data_dir = voc_utils.get_pascal_voc('2012')

        id_list_file = osp.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(mode))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.bgr = bgr

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. Both of them are in CHW
        format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of color image and label whose shapes are (3, H, W) and \
                (1, H, W) respectively. H and W are height and width of the \
                images. The dtype of the color image is ``numpy.float32`` and \
                the dtype of the label image is ``numpy.int32``. The color \
                channel of the color image is determined by the paramter \
                ``self.bgr``.

        """
        if i >= len(self):
            raise IndexError('index is too large')
        img, label = self.get_raw_data(i)
        if self.bgr:
            img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1).astype(np.float32)
        label = label[None]
        return img, label

    def get_raw_data(self, i):
        """Returns the i-th example's images in HWC format.

        The color image that is returned is in RGB.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example (image, label image)

        """
        img_file = osp.join(self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = imread(img_file, mode='RGB')
        label = self._load_label(self.data_dir, self.ids[i])
        return img, label

    def _load_label(self, data_dir, id_):
        label_rgb_file = osp.join(
            data_dir, 'SegmentationClass', id_ + '.png')
        im = Image.open(label_rgb_file)
        label = np.array(im, dtype=np.uint8).astype(np.int32)
        label[label == 255] = -1
        return label


if __name__ == '__main__':
    dataset = VOCSemanticSegmentationDataset()
    for i in range(100):
        dataset.get_example(i)
