import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

from chainercv.dataset import PickableDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class VOCBboxDataset(PickableDataset):

    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.voc_bbox_label_names`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir='auto', split='train', year='2012',
                 use_difficult=False, return_difficult=False):
        super(VOCBboxDataset, self).__init__()
        if return_difficult:
            self.data_names = ('img', 'bbox', 'label', 'difficult')
        else:
            self.data_names = ('img', 'bbox', 'label')
        self.add_getter('img', self.get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self.get_annotations)

        if data_dir == 'auto' and year in ['2007', '2012']:
            data_dir = voc_utils.get_voc(year, split)

        if split not in ['train', 'trainval', 'val']:
            if not (split == 'test' and year == '2007'):
                warnings.warn(
                    'please pick split from \'train\', \'trainval\', \'val\''
                    'for 2012 dataset. For 2007 dataset, you can pick \'test\''
                    ' in addition to the above mentioned splits.'
                )
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult

    def __len__(self):
        return len(self.ids)

    def get_image(self, i):
        """Returns the i-th image.

        Returns a color image. The image is in CHW and RGB format.

        Args:
            i (int): The index of the example.

        Returns:
            An image.
        """
        id_ = self.ids[i]
        img_path = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def get_annotations(self, i):
        """Returns the annotations of the i-th example.

        Returns bounding boxes, labels and difficulties.

        Args:
            i (int): The index of the example.

        Returns:
            A tuple of three arrays.
        """

        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_utils.voc_bbox_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)
        return bbox, label, difficult
