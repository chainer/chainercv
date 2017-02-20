import numpy as np
import os.path as osp
from skimage.io import imread
import zipfile

import chainer
from chainer.dataset import download

from chainer_cv import utils


root = 'pfnet/chainer_cv/online_products'
url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'


def _get_online_products():
    data_root = download.get_dataset_directory(root)
    base_path = osp.join(data_root, 'Stanford_Online_Products')
    if osp.exists(base_path):
        # skip downloading
        return base_path

    download_file_path = utils.cached_download(url)

    with zipfile.ZipFile(download_file_path, 'r') as z:
        z.extractall(data_root)
    return base_path


class OnlineProductsDataset(chainer.dataset.DatasetMixin):

    """Simple class to load data from Online Products Dataset [1].

    .. [1] Stanford Online Products dataset
        http://cvgl.stanford.edu/projects/lifted_struct

    Args:
        data_dir (string): Path to the root of the training data. If this is
            'auto', this class will automatically download data for you
            under ``$CHAINER_DATASET_ROOT/pfnet/chainer_cv/pascal_voc``.
    """

    def __init__(self, base_dir='auto', mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(
                'please pick mode from \'train\', \'test\'')
        if base_dir == 'auto':
            base_dir = _get_online_products()
        self.base_dir = base_dir

        id_list_file = osp.join(base_dir, 'Ebay_{}.txt'.format(mode))
        ids_tmp = [id_.strip().split() for id_ in open(id_list_file)][1:]
        self.class_ids = [int(id_[1]) for id_ in ids_tmp]
        self.super_class_ids = [int(id_[2]) for id_ in ids_tmp]
        self.paths = [osp.join(base_dir, id_[3]) for id_ in ids_tmp]

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image, class_id and super_class_id. The image is in CHW
        format.

        Args:
            i (int): The index of the example.
        Returns:
            i-th example
        """
        img = imread(self.paths[i])
        img = img.transpose(2, 0, 1).astype(np.float32)

        class_id = self.class_ids[i]
        super_class_id = self.super_class_ids[i]
        return img, class_id, super_class_id

    def get_raw_data(self, i):
        """Returns the i-th example's image and class data in HWC format.

        The color image that is returned is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example (image, class_id, super_class_id)

        """
        img = imread(self.paths[i])
        class_id = self.class_ids[i]
        super_class_id = self.super_class_ids[i]
        return img, class_id, super_class_id


def get_online_products(base_dir='auto'):
    """Gets the Online Products Dataset.

    This method returns train and test split of Online Products Dataset as
    done in [1].

    .. [1] Deep Metric Learning via Lifted Structured Feature Embedding
        https://arxiv.org/abs/1511.06452

    Args:
        data_dir (string): Path to the root of the training data. If this is
            'auto', this class will automatically download data for you
            under ``$CHAINER_DATASET_ROOT/pfnet/chainer_cv/pascal_voc``.

    Returns:
        A tuple of two datasets of type
        `chainer_cv.datasets.image_retrieval.OnlineProductsDataset`.
        The first dataset is for training and the second is for testing.

    """
    train_dataset = OnlineProductsDataset(base_dir, mode='train')
    test_dataset = OnlineProductsDataset(base_dir, mode='test')
    return train_dataset, test_dataset


if __name__ == '__main__':
    train, test = get_online_products()
