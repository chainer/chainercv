import copy
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

    """Simple class to load data from `Stanford Online Products Dataset`_.

    .. _`Stanford Online Products Dataset`:
        http://cvgl.stanford.edu/projects/lifted_struct

    The :obj:`mode` selects train and test split of the dataset as done in
    [Song]_. The train split contains the first 11318 classes and the test
    split contains the remaining 11316 classes.

    .. [Song] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, Silvio Savarese.
        Deep Metric Learning via Lifted Structured Feature Embedding.
        https://arxiv.org/abs/1511.06452.

    All returned images are in RGB format.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/online_products`.
        mode ({'train', 'test'}): Mode of the dataset.

    """

    def __init__(self, data_dir='auto', mode='train'):
        if data_dir == 'auto':
            data_dir = _get_online_products()
        self.data_dir = data_dir

        self.class_ids = []
        self.super_class_ids = []
        self.paths = []
        # for mode in ['train', 'test']:
        id_list_file = os.path.join(data_dir, 'Ebay_{}.txt'.format(mode))
        ids_tmp = [id_.strip().split() for id_ in open(id_list_file)][1:]
        self.class_ids += [int(id_[1]) for id_ in ids_tmp]
        self.super_class_ids += [int(id_[2]) for id_ in ids_tmp]
        self.paths += [os.path.join(data_dir, id_[3]) for id_ in ids_tmp]

        self.class_ids_dict = self._list_to_dict(self.class_ids)
        self.super_class_ids_dict = self._list_to_dict(self.super_class_ids)

    def _list_to_dict(self, l):
        dict_ = {}
        for i, v in enumerate(l):
            if v not in dict_:
                dict_[v] = []
            dict_[v].append(i)
        return dict_

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

        img = utils.read_image_as_array(self.paths[i])

        if img.ndim == 2:
            img = utils.gray2rgb(img)
        img = img[:, :, ::-1]  # RGB to BGR
        img = img.transpose(2, 0, 1).astype(np.float32)
        return img, class_id, super_class_id

    def get_raw_data(self, i, rgb=True):
        """Returns the i-th example's image and class data in HWC format.

        The color image that is returned is RGB.

        Args:
            i (int): The index of the example.
            rgb (bool): If false, the returned image will be in BGR.

        Returns:
            i-th example (image, class_id, super_class_id)

        """
        img = utils.read_image_as_array(self.paths[i])
        if img.ndim == 2:
            img = utils.gray2rgb(img)
        if not rgb:
            img = img[:, :, ::-1]
        class_id = self.class_ids[i]
        super_class_id = self.super_class_ids[i]
        return img, class_id, super_class_id

    def get_ids(self, class_id):
        """Get indices of examples in the given class.

        Args:
            class_id (int): the class id.

        Returns:
            list of indices of examples whose class ids are `class_id`.

        """
        return copy.copy(self.class_ids_dict[class_id])


if __name__ == '__main__':
    dataset = OnlineProductsDataset(mode='test')
