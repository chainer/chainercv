import json
import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.coco.coco_utils import get_coco
from chainercv import utils


def _rgb2id(color):
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


class COCOSemanticSegmentationDataset(GetterDataset):

    """Semantic segmentation dataset for `MS COCO`_.

    Semantic segmentations are generated from panoptic segmentations
    as done in the `official toolkit`_.

    .. _`MS COCO`: http://cocodataset.org/#home

    .. _`official toolkit`: https://github.com/cocodataset/panopticapi/
        blob/master/converters/panoptic2semantic_segmentation.py

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/coco`.
        split ({'train', 'val'}): Select a split of the dataset.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"

    """

    def __init__(self, data_dir='auto', split='train'):
        super(COCOSemanticSegmentationDataset, self).__init__()
        if data_dir == 'auto':
            data_dir = get_coco(split, split, '2017', 'panoptic')

        self.img_root = os.path.join(
            data_dir, 'images', '{}{}'.format(split, 2017))

        self.label_root = os.path.join(
            data_dir, 'annotations', 'panoptic_{}{}'.format(split, 2017))
        anno_path = os.path.join(
            data_dir, 'annotations',
            'panoptic_{}{}.json'.format(split, 2017))

        self.data_dir = data_dir
        annos = json.load(open(anno_path, 'r'))
        self.annos = annos

        self.cat_ids = [cat['id'] for cat in annos['categories']]
        self.img_paths = [ann['file_name'][:-4] + '.jpg'
                          for ann in annos['annotations']]

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

        self.keys = ('img', 'label')

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        img_path = os.path.join(
            self.img_root, self.img_paths[i])
        img = utils.read_image(img_path, dtype=np.float32, color=True)
        return img

    def _get_label(self, i):
        # https://github.com/cocodataset/panopticapi/blob/master/converters/
        # panoptic2semantic_segmentation.py#L58
        anno = self.annos['annotations'][i]
        label_path = os.path.join(self.label_root, anno['file_name'])
        rgb_id_map = utils.read_image(
            label_path,
            dtype=np.uint32, color=True)
        id_map = _rgb2id(rgb_id_map)
        label = -1 * np.ones_like(id_map, dtype=np.int32)
        for inst in anno['segments_info']:
            mask = id_map == inst['id']
            label[mask] = self.cat_ids.index(inst['category_id'])
        return label
