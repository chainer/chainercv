import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.utils import read_image


def directory_parsing_label_names(root, numerical_sort=False):
    """Get label names from the directories that are named by them.

    The label names are the names of the directories that locate a
    layer below the root directory.

    The label names can be used together with
    :class:`~chainercv.datasets.DirectoryParsingLabelDataset`.
    The index of a label name corresponds to the label id
    that is used by the dataset to refer the label.

    Args:
        root (string): The root directory.
        numerical_sort (bool): Label names are sorted numerically.
            This means that label :obj:`2` is before label :obj:`10`,
            which is not the case when string sort is used.
            The default value is :obj:`False`.

    Returns:
        list of strings:
        Sorted names of classes.

    """
    label_names = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))]

    if not numerical_sort:
        label_names.sort()
    else:
        label_names = sorted(label_names, key=int)
    return label_names


def _check_img_ext(path):
    img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp']
    return any(os.path.splitext(path)[1].lower() == extension for
               extension in img_extensions)


def _parse_label_dataset(root, label_names,
                         check_img_file=_check_img_ext):
    img_paths = []
    labels = []
    for label, label_name in enumerate(label_names):
        label_dir = os.path.join(root, label_name)
        if not os.path.isdir(label_dir):
            continue

        walk_dir = sorted(os.walk(label_dir), key=lambda x: x[0])
        for cur_dir, _, names in walk_dir:
            names = sorted(names)
            for name in names:
                img_path = os.path.join(cur_dir, name)
                if check_img_file(img_path):
                    img_paths.append(img_path)
                    labels.append(label)

    return img_paths, np.array(labels, np.int32)


class DirectoryParsingLabelDataset(GetterDataset):
    """A label dataset whose label names are the names of the subdirectories.

    The label names are the names of the directories that locate a layer below
    the root directory.
    All images locating under the subdirectoies will be categorized to classes
    with subdirectory names.
    An image is parsed only when the function :obj:`check_img_file`
    returns :obj:`True` by taking the path to the image as an argument.
    If :obj:`check_img_file` is :obj:`None`,
    the path with any image extensions will be parsed.

    Example:

        A directory structure should be one like below.

        .. code::

            root
            |-- class_0
            |   |-- img_0.png
            |   |-- img_1.png
            |
            --- class_1
                |-- img_0.png

        >>> from chainercv.datasets import DirectoryParsingLabelDataset
        >>> dataset = DirectoryParsingLabelDataset('root')
        >>> dataset.img_paths
        ['root/class_0/img_0.png', 'root/class_0/img_1.png',
        'root_class_1/img_0.png']
        >>> dataset.labels
        array([0, 0, 1])

    Args:
        root (string): The root directory.
        check_img_file (callable): A function to determine
            if a file should be included in the dataset.
        color (bool): If :obj:`True`, this dataset read images
            as color images. The default value is :obj:`True`.
        numerical_sort (bool): Label names are sorted numerically.
            This means that label :obj:`2` is before label :obj:`10`,
            which is not the case when string sort is used.
            Regardless of this option, string sort is used for the
            order of files with the same label.
            The default value is :obj:`False`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)` [#directory_parsing_1]_", \
        :obj:`float32`, "RGB, :math:`[0, 255]`"
        :obj:`label`, scalar, :obj:`int32`, ":math:`[0, \#class - 1]`"

    .. [#directory_parsing_1] :math:`(1, H, W)` if :obj:`color = False`.
    """

    def __init__(self, root, check_img_file=None, color=True,
                 numerical_sort=False):
        super(DirectoryParsingLabelDataset, self).__init__()

        self.color = color

        label_names = directory_parsing_label_names(
            root, numerical_sort=numerical_sort)
        if check_img_file is None:
            check_img_file = _check_img_ext

        self.img_paths, self.labels = _parse_label_dataset(
            root, label_names, check_img_file)

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        return read_image(self.img_paths[i], color=self.color)

    def _get_label(self, i):
        return self.labels[i]
