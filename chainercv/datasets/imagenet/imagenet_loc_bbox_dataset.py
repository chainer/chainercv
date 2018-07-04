import os

from chainer.dataset import download

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.imagenet.imagenet_utils import imagenet_loc_synset_ids
from chainercv.datasets.voc.voc_utils import parse_voc_bbox_annotation
from chainercv.utils import read_image


class ImagenetLocBboxDataset(GetterDataset):

    """ILSVRC2012 ImageNet localization dataset.

    The data is distributed on `the official Kaggle page`_.

    .. _`the official Kaggle page`: https://www.kaggle.com/c/
        imagenet-object-localization-challenge

    Please refer to the readme of ILSVRC2012 dev kit for a comprehensive
    documentation. Note that the detection part of ILSVRC has not changed since
    2012.

    Every image in the training and validation sets has a single
    image-level label specifying the presence of one object category.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`,
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/imagenet` is used.
        split ({'train', 'val'}): Selects a split of the dataset.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label`, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"

    """

    def __init__(self, data_dir='auto', split='train'):
        super(ImagenetLocBboxDataset, self).__init__()
        if data_dir == 'auto':
            data_dir = download.get_dataset_directory(
                'pfnet/chainercv/imagenet')
        self.base_dir = os.path.join(data_dir, 'ILSVRC')
        imageset_dir = os.path.join(self.base_dir, 'ImageSets/CLS-LOC')

        ids = []
        if split == 'train':
            imageset_path = os.path.join(imageset_dir, 'train_loc.txt')
        elif split == 'val':
            imageset_path = os.path.join(imageset_dir, 'val.txt')
        with open(imageset_path) as f:
            for l in f:
                id_ = l.split()[0]
                ids.append(id_)
        self.ids = ids
        self.split = split

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label'), self._get_inst_anno)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(
            self.base_dir, 'Data/CLS-LOC', self.split,
            self.ids[i] + '.JPEG')
        img = read_image(img_path, color=True)
        return img

    def _get_inst_anno(self, i):
        anno_path = os.path.join(
            self.base_dir, 'Annotations/CLS-LOC', self.split,
            self.ids[i] + '.xml')
        bbox, label, _ = parse_voc_bbox_annotation(
            anno_path, imagenet_loc_synset_ids,
            skip_names_not_in_label_names=False)
        return bbox, label
