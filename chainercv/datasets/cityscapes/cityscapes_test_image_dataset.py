import glob
import os

from chainer.dataset import download

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.utils import read_image


class CityscapesTestImageDataset(GetterDataset):

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

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
    """

    def __init__(self, data_dir='auto'):
        super(CityscapesTestImageDataset, self).__init__()

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

        self.img_paths = []
        for city_dname in sorted(glob.glob(os.path.join(img_dir, '*'))):
            for img_path in sorted(glob.glob(
                    os.path.join(city_dname, '*_leftImg8bit.png'))):
                self.img_paths.append(img_path)

        self.add_getter('img', self._get_image)
        self.keys = 'img'  # do not return tuple

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        return read_image(self.img_paths[i])
