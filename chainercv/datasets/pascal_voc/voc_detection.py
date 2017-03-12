import glob
import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

import chainer
from chainer.dataset import download

from chainercv.datasets.pascal_voc import voc_utils
from chainercv.utils.dataset_utils import cache_load
from chainercv.utils import read_image_as_array


class VOCDetectionDataset(chainer.dataset.DatasetMixin):

    """Dataset class for the detection task of PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    The bounding box is a two dimensional tensor of shape
    :math:`(R, 5)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :obj:`(x_min, y_min, x_max, y_max, label_id)`, where first
    four attributes are coordinates of the bottom left and the top right
    vertices. The last attribute is the label id, which points to the
    category of the object in the bounding box.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/pascal_voc`.
        mode ({'train', 'val', 'trainval'}): select from dataset splits used
            in VOC.
        year ({'2007', '2012'}): use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If true, use images that are labeled as
            difficult in the original annotation.
        use_cache (bool): If true, use cache of object annotations. This
            is useful in the case when parsing annotation takes time.
            When this is false, the dataset will not write cache.
        delete_cache (bool): Delete the cache described above.

    """

    labels = voc_utils.pascal_voc_labels

    def __init__(self, data_dir='auto', mode='train', year='2012',
                 use_difficult=False,
                 use_cache=False, delete_cache=False):
        if data_dir == 'auto' and year in voc_utils.urls:
            data_dir = voc_utils.get_pascal_voc(year)

        if mode not in ['train', 'trainval', 'val']:
            warnings.warn(
                'please pick mode from \'train\', \'trainval\', \'val\'')

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(mode))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult

        # cache objects
        data_root = download.get_dataset_directory(voc_utils.root)

        pkl_file = os.path.join(
            data_root, 'detection_objects_{}_{}.pkl'.format(year, mode))
        self.objects = cache_load(
            pkl_file, self._collect_objects, delete_cache,
            use_cache, args=(self.data_dir, self.ids, self.use_difficult))
        self.keys = self.objects.keys()

    def _collect_objects(self, data_dir, ids, use_difficult):
        objects = {}
        anno_dir = os.path.join(data_dir, 'Annotations')
        for fn in glob.glob('{}/*.xml'.format(anno_dir)):
            tree = ET.parse(fn)
            filename = tree.find('filename').text
            img_id = os.path.splitext(filename)[0]
            # skip annotation that is not included in ids
            if img_id not in ids:
                continue

            datums = []
            for obj in tree.findall('object'):
                # when in not using difficult mode, and the object is
                # difficult, skipt it.
                if not use_difficult and int(obj.find('difficult').text) == 1:
                    continue

                bbox_ = obj.find('bndbox')  # bndbox is the key used by raw VOC
                bbox = [int(bbox_.find('xmin').text),
                        int(bbox_.find('ymin').text),
                        int(bbox_.find('xmax').text),
                        int(bbox_.find('ymax').text)]
                # make pixel indexes 0-based
                bbox = [float(b - 1) for b in bbox]

                datum = {
                    'filename': filename,
                    'name': obj.find('name').text.lower().strip(),
                    'pose': obj.find('pose').text.lower().strip(),
                    'truncated': int(obj.find('truncated').text),
                    'difficult': int(obj.find('difficult').text),
                    'bbox': bbox,
                }
                datums.append(datum)
            objects[img_id] = datums
        return objects

    def __len__(self):
        return len(self.objects)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is BGR.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        # Load a bbox and its category
        objects = self.objects[self.keys[i]]
        bbox = []
        for obj in objects:
            _bb = obj['bbox']
            name = obj['name']
            label_id = self.labels.index(name)
            bbox_elem = np.asarray([_bb[0], _bb[1], _bb[2], _bb[3], label_id],
                              dtype=np.float32)
            bbox.append(bbox_elem)

        bbox = np.stack(bbox)

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', obj['filename'])
        img = read_image_as_array(img_file)  # RGB

        img = img[:, :, ::-1]  # RGB to BGR
        img = img.transpose(2, 0, 1).astype(np.float32)
        return img, bbox
