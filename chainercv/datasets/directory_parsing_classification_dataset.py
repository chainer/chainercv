import numpy as np
import os

import chainer
from chainercv.utils import read_image


def find_label_names(root):
    label_names = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))]
    label_names.sort()
    return label_names


def _ends_with_img_ext(filename):
    img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp']
    return any(os.path.splitext(filename)[1].lower().endswith(extension) for
               extension in img_extensions)


def _parse_classification_dataset(root, label_names,
                                  check_img_file=_ends_with_img_ext):
    # Use label_name_to_idx for performance.
    label_name_to_idx = {label_names[i]: i for i in range(len(label_names))}

    img_paths = []
    labels = []
    for label_name in os.listdir(root):
        label_dir = os.path.join(root, label_name)
        if not os.path.isdir(label_dir):
            continue

        for cur_dir, _, filenames in sorted(os.walk(label_dir)):
            for filename in filenames:
                if check_img_file(filename):
                    img_paths.append(os.path.join(cur_dir, filename))
                    labels.append(label_name_to_idx[label_name])

    return img_paths, np.array(labels, np.int32)


class DirectoryParsingClassificationDataset(chainer.dataset.DatasetMixin):
    """A data loader that loads images arranged in directory by classes.

    The label names are names of the directories locating a layer below the
    root.
    All images locating under the subdirectory will be categorized with the
    label associated to the subdirectory.
    The image is parsed only when :obj:`check_img_file` returns :obj:`True`
    when the path to the image is given as an argument.
    If this is :obj:`None`, the path with any image extensions will be parsed.

    Example:

        With a directory structure like below.

        .. code::

            root
            |-- class_0
            |   |-- img_0.png
            |   |-- img_1.png
            |
            --- class_1
                |-- img_0.png

        >>> from chainercv.dataset import DirectoryParsingClassificationDataset
        >>> dataset = DirectoryParsingClassificationDataset('root')
        >>> dataset.img_paths
        ['root/class_0/img_0.png', 'root/class_0/img_1.png',
        'root_class_1/img_0.png']
        >>> dataset.labels
        array([0, 0, 1])

    Args:
        root (str): The root directory.
        check_img_file (callable): A function to determine
            if a file should be included in dataset.
        color (bool): If :obj:`True`, read images as color images.

    """

    def __init__(self, root, check_img_file=None, color=True):
        self.color = color
        label_names = find_label_names(root)
        if check_img_file is None:
            check_img_file = _ends_with_img_ext

        self.img_paths, self.labels = _parse_classification_dataset(
            root, label_names, check_img_file)

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        img = read_image(self.img_paths[i], color=self.color)
        label = self.labels[i]
        return img, label
