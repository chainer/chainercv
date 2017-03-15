import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

import chainer

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.utils import read_image_as_array


class VOCDetectionDataset(chainer.dataset.DatasetMixin):

    """Dataset class for the detection task of PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :obj:`(x_min, y_min, x_max, y_max)`, where the
    four attributes are coordinates of the bottom left and the top right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image. These are integers
    that correspond to object ID which are listed in
    :obj:`VOCDetectionDataset.labels`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/pascal_voc`.
        mode ({'train', 'val', 'trainval', 'test'}): select from dataset splits
            used in VOC. :obj:`test` mode is only available for 2007 dataset.
        year ({'2007', '2012'}): use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If true, use images that are labeled as
            difficult in the original annotation.

    """

    labels = voc_utils.pascal_voc_labels

    def __init__(self, data_dir='auto', mode='train', year='2012',
                 use_difficult=False):
        if data_dir == 'auto' and year in ['2007', '2012']:
            data_dir = voc_utils.get_pascal_voc(year, mode)

        if mode not in ['train', 'trainval', 'val']:
            if not (mode == 'test' and year == '2007'):
                warnings.warn(
                    'please pick mode from \'train\', \'trainval\', \'val\''
                    'for 2012 dataset. For 2007 dataset, you can pick \'test\''
                    ' in addition to the above mentioned modes.'
                )

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(mode))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is BGR.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        for obj in anno.findall('object'):
            # when in not using difficult mode, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.labels.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image_as_array(img_file)  # RGB

        img = img[:, :, ::-1]  # RGB to BGR
        img = img.transpose(2, 0, 1).astype(np.float32)
        return img, bbox, label
