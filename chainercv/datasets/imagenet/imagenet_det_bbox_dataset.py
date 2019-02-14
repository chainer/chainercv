import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.imagenet.imagenet_utils import get_ilsvrc_devkit
from chainercv.datasets.imagenet.imagenet_utils import imagenet_det_synset_ids
from chainercv.datasets.voc.voc_utils import parse_voc_bbox_annotation
from chainercv.utils import read_image


class ImageNetDetBboxDataset(GetterDataset):

    """ILSVRC ImageNet detection dataset.

    The data is distributed on the `official Kaggle page`_.

    .. _`official Kaggle page`: https://www.kaggle.com/c/
        imagenet-object-detection-challenge

    Please refer to the readme of ILSVRC2014 dev kit for a comprehensive
    documentation. Note that the detection part of ILSVRC has not changed since
    2014. An overview of annotation process is described in the `paper`_.

    .. _`paper`: http://ai.stanford.edu/~olga/papers/chi2014-MultiLabel.pdf

    Every image in the training set has one or more image-level labels.
    The image-level labels determine the full presence, partial presence or
    absence of one or more object categories.
    Bounding boxes are provided around instances of the present categories.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`,
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/imagenet` is used.
        split ({'train', 'val', 'val1', 'val2'}): Selects a split of the
            dataset.
        year ({'2013', '2014'}): Use a dataset prepared for a challenge
            held in :obj:`year`. The default value is :obj:`2014`.
        return_img_label (bool): If :obj:`True`, this dataset returns
            image-wise labels. This consists of two arrays:
            :obj:`img_label` and :obj:`img_label_type`.
        use_val_blacklist (bool): If :obj:`False`, images that are
            included in the blacklist are avoided when
            the split is :obj:`val`. The default value is :obj:`False`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label`, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`img_label` [#imagenet_det_1]_, ":math:`(M,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`img_label_type` [#imagenet_det_1]_ [#imagenet_det_2]_, \
            ":math:`(M,)`", :obj:`int32`, ":math:`[-1, 1]`"

    .. [#imagenet_det_1] available
        if :obj:`return_img_label = True`.
    .. [#imagenet_det_2] :obj:`-1` means absent. :obj:`1` means present.
        :obj:`0` means partially present. When a category is partially
        present, the image contains at least one
        instance of X, but not all instances of X may be annotated with
        bounding boxes.

    """

    def __init__(self, data_dir='auto', split='train', year='2014',
                 return_img_label=False, use_val_blacklist=False):
        super(ImageNetDetBboxDataset, self).__init__()
        if data_dir == 'auto':
            data_dir = get_ilsvrc_devkit()
        val_blacklist_path = os.path.join(
            data_dir, 'ILSVRC2014_devkit/data/',
            'ILSVRC2014_det_validation_blacklist.txt')

        if year not in ('2013', '2014'):
            raise ValueError('\'year\' has to be either '
                             '\'2013\' or \'2014\'.')
        self.base_dir = os.path.join(data_dir, 'ILSVRC')
        imageset_dir = os.path.join(self.base_dir, 'ImageSets/DET')
        if not os.path.exists(imageset_dir):
            raise ValueError(
                'Images of ImageNet Detection data is not found.'
                'Please download them from the offical kaggle page.')

        if split == 'train':
            img_labels = {}
            for lb in range(0, 200):
                with open(os.path.join(
                        imageset_dir, 'train_{}.txt'.format(lb + 1))) as f:
                    for l in f:
                        id_ = l.split()[0]
                        if 'ILSVRC2014' in id_ and year != '2014':
                            continue

                        anno_type = l.split()[1]
                        if id_ not in img_labels:
                            img_labels[id_] = []
                        img_labels[id_].append((lb, int(anno_type)))
                self.img_labels = img_labels
                self.ids = list(img_labels.keys())
                self.split_type = 'train'
        else:
            if return_img_label:
                raise ValueError('split has to be \'train\' when '
                                 'return_img_label is True')
            if use_val_blacklist:
                blacklist_ids = []
            else:
                ids = []
                with open(os.path.join(
                        imageset_dir, 'val.txt'.format(split))) as f:
                    for l in f:
                        id_ = l.split()[0]
                        ids.append(id_)
                blacklist_ids = []
                with open(val_blacklist_path) as f:
                    for l in f:
                        index = int(l.split()[0])
                        blacklist_ids.append(ids[index])

            ids = []
            with open(os.path.join(
                    imageset_dir, '{}.txt'.format(split))) as f:
                for l in f:
                    id_ = l.split()[0]
                    if id_ not in blacklist_ids:
                        ids.append(id_)
                self.ids = ids
            self.split_type = 'val'

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label'), self._get_inst_anno)
        if return_img_label:
            self.add_getter(
                ('img_label', 'img_label_type'), self._get_img_label)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(
            self.base_dir, 'Data/DET', self.split_type,
            self.ids[i] + '.JPEG')
        img = read_image(img_path, color=True)
        return img

    def _get_inst_anno(self, i):
        if 'extra' not in self.ids[i]:
            anno_path = os.path.join(
                self.base_dir, 'Annotations/DET', self.split_type,
                self.ids[i] + '.xml')
            bbox, label, _ = parse_voc_bbox_annotation(
                anno_path, imagenet_det_synset_ids,
                skip_names_not_in_label_names=True)
        else:
            bbox = np.zeros((0, 4), dtype=np.float32)
            label = np.zeros((0,), dtype=np.int32)
        return bbox, label

    def _get_img_label(self, i):
        img_label = np.array([val[0] for val in self.img_labels[self.ids[i]]],
                             dtype=np.int32)
        img_label_type = np.array(
            [val[1] for val in self.img_labels[self.ids[i]]],
            dtype=np.int32)
        return img_label, img_label_type
