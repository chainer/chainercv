import glob
import os

from chainer import dataset
from chainer.dataset import download
from chainercv.utils import read_image


class CityscapesTestImageDataset(dataset.DatasetMixin):

    """Image dataset for test split of `Cityscapes dataset`_.

    .. _`Cityscapes dataset`: https://www.cityscapes-dataset.com

    .. note::

        Please manually download the data because it is not allowed to
        re-distribute Cityscapes dataset.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain the :obj:`leftImg8bit` directory. If :obj:`auto` is given,
            it uses :obj:`$CHAINER_DATSET_ROOT/pfnet/chainercv/cityscapes` by
            default.

    """

    def __init__(self, data_dir='auto'):
        if data_dir == 'auto':
            data_dir = download.get_dataset_directory(
                'pfnet/chainercv/cityscapes')

        img_dir = os.path.join(data_dir, os.path.join('leftImg8bit', 'test'))
        if not os.path.exists(img_dir):
            raise ValueError(
                'Cityscapes dataset does not exist at the expected location.'
                'Please download it from https://www.cityscapes-dataset.com/.'
                'Then place directory leftImg8bit at {}.'.format(
                    os.path.join(data_dir, 'leftImg8bit')))

        self.img_paths = list()
        for city_dname in sorted(glob.glob(os.path.join(img_dir, '*'))):
            for img_path in sorted(glob.glob(
                    os.path.join(city_dname, '*_leftImg8bit.png'))):
                self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        """Returns the i-th test image.

        Returns a color image. The color image is in CHW format.

        Args:
            i (int): The index of the example.

        Returns:
            A color image whose shape is (3, H, W). H and W are height and
            width of the image.
            The dtype of the color image is :obj:`numpy.float32`.

        """
        return read_image(self.img_paths[i])
