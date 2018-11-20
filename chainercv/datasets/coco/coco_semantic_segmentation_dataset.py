import json
import numpy as np
import os

from chainercv import utils
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.coco.coco_utils import get_coco


def _rgb2id(color):
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


class COCOSemanticSegmentationDataset(GetterDataset):

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

        self.cat_ids = ['-1']  # bg
        self.cat_ids += cat['id'] for cat in annos['categories']]
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
        anno = self.annos['annotations'][i]
        label_path = os.path.join(self.label_root, anno['file_name'])
        rgb_id_map = utils.read_image(
            label_path,
            dtype=np.uint32, color=True)
        id_map = _rgb2id(rgb_id_map)
        label = np.zeros_like(id_map, dtype=np.int32)
        for inst in anno['segments_info']:
            mask = id_map == inst['id']
            label[mask] = self.cat_ids.index(inst['category_id'])
        return label