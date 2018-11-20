import json
import numpy as np
import os

from chainercv import utils
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.coco.coco_utils import get_coco


def _rgb2id(color):
    return color[0] + 256 * color[1] + 256 * 256 * color[2]


class COCOPanopticSegmentationDataset(GetterDataset):

    def __init__(
            self, data_dir='auto', split='train',
            use_crowded=False, return_crowded=False,
            return_area=False):
        super(COCOPanopticSegmentationDataset, self).__init__()
        self.use_crowded = use_crowded

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
        self.add_getter('mask', self._get_mask)
        self.add_getter(
            ['label', 'area', 'crowded'],
            self._get_annotations)
        keys = ('img', 'mask', 'label')
        if return_area:
            keys += ('area',)
        if return_crowded:
            keys += ('crowded',)
        self.keys = keys

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        img_path = os.path.join(
            self.img_root, self.img_paths[i])
        img = utils.read_image(img_path, dtype=np.float32, color=True)
        return img

    def _get_mask(self, i):
        anno = self.annos['annotations'][i]
        label_path = os.path.join(self.label_root, anno['file_name'])
        rgb_id_map = utils.read_image(
            label_path,
            dtype=np.uint32, color=True)
        id_map = _rgb2id(rgb_id_map)

        H, W = id_map.shape
        n_seg = len(anno['segments_info'])
        crowded = []
        mask = np.zeros((n_seg, H, W), dtype=np.bool)
        for i, segm in enumerate(anno['segments_info']):
            mask[i, id_map == segm['id']] = True
            crowded.append(segm['iscrowd'])
        crowded = np.array(crowded, dtype=np.bool)

        if not self.use_crowded:
            not_crowded = np.logical_not(crowded)
            mask = mask[not_crowded]
        return mask

    def _get_annotations(self, i):
        anno = self.annos['annotations'][i]['segments_info']

        label = np.array(
            [self.cat_ids.index(segm['category_id'])
             for segm in anno], dtype=np.int32)
        area = np.array(
            [segm['area'] for segm in anno], dtype=np.float32)
        crowded = np.array(
            [segm['iscrowd'] for segm in anno], dtype=np.bool)

        if not self.use_crowded:
            not_crowded = np.logical_not(crowded)
            label = label[not_crowded]
            area = area[not_crowded]
            crowded = crowded[not_crowded]
        return label, area, crowded
