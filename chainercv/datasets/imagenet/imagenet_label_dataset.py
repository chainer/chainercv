import numpy as np
import os

import chainer
from chainercv import utils
from chainercv.datasets.imagenet import imagenet_utils


class ImageNetLabelDataset(chainer.dataset.DatasetMixin):

    """`ImageNet`_ dataset with annotated class labels.

    .. _`ImageNet`:
        http://image-net.org/challenges/LSVRC/2015/download-images-3j16.php

    When queried by an index, this dataset returns a corresponding
    :obj:`img, label`, a tuple of an image and class id.
    The image is in RGB and CHW format.
    The class id is between 0 and 999.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/imagenet`.

    """

    def __init__(self, data_dir='auto'):
        super(ImageNetLabelDataset, self).__init__()
        if data_dir == 'auto':


        image_class_labels_file = os.path.join(
            self.data_dir, 'image_class_labels.txt')
        labels = [int(d_label.split()[1]) - 1 for
                  d_label in open(image_class_labels_file)]
        self._labels = np.array(labels, dtype=np.int32)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and its label.
            The image is in CHW format and its color channel is ordered in
            RGB.
            If :obj:`return_bb = True`,
            a bounding box is appended to the returned value.
            If :obj:`return_mask = True`,
            a probability map is appended to the returned value.

        """
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.paths[i]),
            color=True)
        label = self._labels[i]

        if not self.return_prob_map:
            if self.return_bb:
                return img, label, self.bbs[i]
            else:
                return img, label

        prob_map = utils.read_image(self.prob_map_paths[i],
                                    dtype=np.uint8, color=False)
        prob_map = prob_map.astype(np.float32) / 255  # [0, 255] -> [0, 1]
        prob_map = prob_map[0]  # (1, H, W) --> (H, W)
        if self.return_bb:
            return img, label, self.bbs[i], prob_map
        else:
            return img, label, prob_map
