import numpy as np
import os

import chainer
from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/online_products'
url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'


def _get_online_products():
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'Stanford_Online_Products')
    if os.path.exists(base_path):
        # skip downloading
        return base_path

    download_file_path = utils.cached_download(url)
    ext = os.path.splitext(url)[1]
    utils.extractall(download_file_path, data_root, ext)
    return base_path


class OnlineProductsDataset(chainer.dataset.DatasetMixin):

    """Dataset class for `Stanford Online Products Dataset`_.

    .. _`Stanford Online Products Dataset`:
        http://cvgl.stanford.edu/projects/lifted_struct

    When queried by an index, this dataset returns a corresponding
    :obj:`img, class_id, super_class_id`, a tuple of an image, a class id and
    a coarse level class id.
    Images are in BGR and CHW format.
    Class ids start from 0.

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

    """

    def __init__(self, data_dir='auto', split='train'):
        if data_dir == 'auto':
            data_dir = _get_online_products()
        self.data_dir = data_dir

        self.class_ids = []
        self.super_class_ids = []
        self.paths = []
        # for split in ['train', 'test']:
        id_list_file = os.path.join(data_dir, 'Ebay_{}.txt'.format(split))
        ids_tmp = [id_.strip().split() for id_ in open(id_list_file)][1:]
        # ids start from 0
        self.class_ids += [int(id_[1]) - 1 for id_ in ids_tmp]
        self.super_class_ids += [int(id_[2]) - 1 for id_ in ids_tmp]
        self.paths += [os.path.join(data_dir, id_[3]) for id_ in ids_tmp]

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image, class_id and super_class_id. The image is in CHW
        format.
        The returned image is BGR.

        Args:
            i (int): The index of the example.
        Returns:
            i-th example
        """
        class_id = np.array(self.class_ids[i], np.int32)
        super_class_id = np.array(self.super_class_ids[i], np.int32)

        img = utils.read_image(self.paths[i], color=True)
        return img, class_id, super_class_id
