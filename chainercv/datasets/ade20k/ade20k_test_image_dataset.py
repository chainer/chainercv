import glob
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.ade20k.ade20k_utils import get_ade20k
from chainercv.utils import read_image

root = 'pfnet/chainercv/ade20k'
url = 'http://data.csail.mit.edu/places/ADEchallenge/release_test.zip'


class ADE20KTestImageDataset(GetterDataset):

    """Image dataset for test split of `ADE20K`_.

    This is an image dataset of test split in ADE20K dataset distributed at
    MIT Scene Parsing Benchmark website. It has 3,352 test images.

    .. _`MIT Scene Parsing Benchmark`: http://sceneparsing.csail.mit.edu/

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain the :obj:`release_test` dir. If :obj:`auto` is given, the
            dataset is automatically downloaded into
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/ade20k`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
    """

    def __init__(self, data_dir='auto'):
        super(ADE20KTestImageDataset, self).__init__()

        if data_dir is 'auto':
            data_dir = get_ade20k(root, url)
        img_dir = os.path.join(data_dir, 'release_test', 'testing')
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

        self.add_getter('img', self._get_image)
        self.keys = 'img'  # do not return tuple

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        return read_image(self.img_paths[i])
