import filelock
import numpy as np
import os

from chainer.dataset import download

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv import utils


root = 'pfnet/chainercv/online_products'
url = 'http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'

online_products_super_label_names = (
    'bicycle',
    'cabinet',
    'chair',
    'coffee_maker',
    'fan',
    'kettle',
    'lamp',
    'mug',
    'sofa',
    'stapler',
    'table',
    'toaster'
)


def _get_online_products():
    # To support ChainerMN, the target directory should be locked.
    with filelock.FileLock(os.path.join(download.get_dataset_directory(
            'pfnet/chainercv/.lock'), 'online_products.lock')):
        data_root = download.get_dataset_directory(root)
        base_path = os.path.join(data_root, 'Stanford_Online_Products')
        if os.path.exists(base_path):
            # skip downloading
            return base_path

        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, data_root, ext)
    return base_path


class OnlineProductsDataset(GetterDataset):

    """Dataset class for `Stanford Online Products Dataset`_.

    .. _`Stanford Online Products Dataset`:
        http://cvgl.stanford.edu/projects/lifted_struct

    The :obj:`split` selects train and test split of the dataset as done in
    [#]_. The train split contains the first 11318 classes and the test
    split contains the remaining 11316 classes.

    .. [#] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, Silvio Savarese.
        `Deep Metric Learning via Lifted Structured Feature Embedding\
        <https://arxiv.org/abs/1511.06452>`_. arXiv 2015.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/online_products`.
        split ({'train', 'test'}): Select a split of the dataset.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, scalar, :obj:`int32`, ":math:`[0, \#class - 1]`"
        :obj:`super_label`, scalar, :obj:`int32`, \
        ":math:`[0, \#super\_class - 1]`"
    """

    def __init__(self, data_dir='auto', split='train'):
        super(OnlineProductsDataset, self).__init__()

        if data_dir == 'auto':
            data_dir = _get_online_products()
        self.data_dir = data_dir

        self.paths = []
        # for split in ['train', 'test']:
        id_list_file = os.path.join(data_dir, 'Ebay_{}.txt'.format(split))
        ids_tmp = [id_.strip().split() for id_ in open(id_list_file)][1:]
        # ids start from 0
        self.class_ids = np.array(
            [int(id_[1]) - 1 for id_ in ids_tmp], dtype=np.int32)
        self.super_class_ids = np.array(
            [int(id_[2]) - 1 for id_ in ids_tmp], dtype=np.int32)
        self.paths += [os.path.join(data_dir, id_[3]) for id_ in ids_tmp]

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)
        self.add_getter('super_label', self._get_super_label)

    def __len__(self):
        return len(self.paths)

    def _get_image(self, i):
        return utils.read_image(self.paths[i], color=True)

    def _get_label(self, i):
        return self.class_ids[i]

    def _get_super_label(self, i):
        return self.super_class_ids[i]
