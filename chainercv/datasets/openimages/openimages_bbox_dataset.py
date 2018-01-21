from __future__ import print_function
from collections import defaultdict
import csv
import numpy as np
import os

import chainer

from chainercv import utils

from chainercv.datasets.openimages.openimages_utils import get_image
from chainercv.datasets.openimages.openimages_utils import get_openimages
from chainercv.datasets.openimages.openimages_utils import openimages_bbox_labels  # NOQA


class OpenImagesBboxDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir='auto', split='train',
                 predownload=False):

        if data_dir == 'auto':
            data_dir = get_openimages(split, predownload)
        self.data_dir = data_dir

        # load image locations
        self.img_root = os.path.join(data_dir, 'images')

        img_fn = os.path.join(data_dir, "images",
                              "2017_11", split, "images.csv")
        self.img_props = dict()
        with open(img_fn) as f:
            reader = csv.reader(f)
            header = next(reader)
            for r in reader:
                d = dict(zip(header, r))
                self.img_props[d["ImageID"]] = d

        anno_fn = os.path.join(data_dir, "annotations",
                               "2017_11", split, "annotations-human-bbox.csv")
        self.anns = defaultdict(list)
        with open(anno_fn) as f:
            reader = csv.reader(f)
            header = next(reader)
            for i in reader:
                d = dict(zip(header, i))
                self.anns[d["ImageID"]] += [d]

        self.label_keys = openimages_bbox_labels.keys()

        self.ids = self.img_props.keys()

    def __len__(self):
        return len(self.ids)

    @property
    def labels(self):
        labels = list()
        for i in range(len(self)):
            labels.append(self._get_annotations(i, 0, 0)[1])
        return labels

    def get_example(self, i):
        img_id = self.ids[i]
        img_fn = get_image(self.img_props[img_id])
        img = utils.read_image(img_fn, dtype=np.float32, color=True)
        _, H, W = img.shape

        bbox, label = self._get_annotations(i, W, H)

        return tuple([img, bbox, label])

    def _get_annotations(self, i, width, height):
        img_id = self.ids[i]
        # [ImageID,Source,LabelName,Confidence,
        #  XMin,XMax,YMin,YMax,
        #  IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside]
        annos = self.anns[img_id]
        if len(annos) == 0:
            bbox = np.zeros((0, 4), dtype=np.float32)
        else:
            bbox = np.array([[
                float(a["YMin"]) * height,
                float(a["XMin"]) * width,
                float(a["YMax"]) * height,
                float(a["XMax"]) * width,
            ] for a in annos], dtype=np.float32)
        label = np.array(
            [self.label_keys.index(a["LabelName"]) for a in annos],
            dtype=np.int32)
        return bbox, label


if __name__ == '__main__':
    from chainercv.datasets.openimages.openimages_utils import openimages_bbox_label_names  # NOQA
    from chainercv.visualizations import vis_bbox
    import matplotlib.pyplot as plt
    import traceback
    dataset = OpenImagesBboxDataset(split='test', predownload=True)
    for i in range(len(dataset)):
        try:
            img, bbox, label = dataset.get_example(i)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
        vis_bbox(img, bbox, label,
                 label_names=openimages_bbox_label_names)
        plt.show()
