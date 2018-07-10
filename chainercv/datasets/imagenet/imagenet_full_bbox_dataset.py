import numpy as np
import os

from chainer.dataset import download

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.imagenet.imagenet_utils import \
    imagenet_full_bbox_synset_ids
from chainercv.datasets.voc.voc_utils import parse_voc_bbox_annotation
from chainercv.utils import read_image


class ImagenetFullBboxDataset(GetterDataset):

    """ImageNet with bounding box annotation.

    There are 3627 categories in this dataset.

    Readable label names can be found by
    :obj:`chainercv.datasets.get_imagenet_full_bbox_label_names`.

    The data needs to be downloaded from the official page.
    There are four steps to prepare data.

    1. Download annotations from \
        http://image-net.org/Annotation/Annotation.tar.gz.
    2. Expand it under :obj:`DATA_DIR/Full/Annotation` with the following \
        command.

    .. code::

        find -name  "*.tar.gz" | while read NAME ; \\
            do tar -xvf "${NAME%.tar.gz}.tar.gz" ; done

    3. Download images using ImageNet API for all synset ids in \
        :obj:`imagenet_full_bbox_synset_ids`. Images for each synset \
        can be downloaded by a command like below. You need to register \
        ImageNet website to use this API.

    .. code::

        wget -O Data/<id>.tar "http://www.image-net.org/download/synset? \\
            wnid=<id>&username=<username>&accesskey=<key>&release=latest& \\
            src=stanford

    4. Expand images under :obj:`DATA_DIR/Full/Data` with the following \
        command.

    .. code::

        find -name  "*.tar" | while read NAME ; do mkdir ${NAME%.tar}; \\
            mv ${NAME%.tar}.tar ${NAME%.tar}; cd ${NAME%.tar}; \\
            tar -xvf ${NAME%.tar}.tar; rm ${NAME%.tar}.tar ; cd ..; done

    Args:
        data_dir (string): Path to the root of the data. If this is
            :obj:`auto`,
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/imagenet` is used.
        return_img_label (bool): If :obj:`True`, this dataset returns
            image-wise labels.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label`, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`img_label` [#imagenet_full_1]_, ":math:`()`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"

    .. [#imagenet_full_1] available
        if :obj:`return_img_label = True`.

    """

    def __init__(self, data_dir='auto',
                 return_img_label=False):
        super(ImagenetFullBboxDataset, self).__init__()
        if data_dir == 'auto':
            data_dir = download.get_dataset_directory(
                'pfnet/chainercv/imagenet')
        self.base_dir = os.path.join(data_dir, 'Full')

        self.paths = []
        self.cls_names = []
        self.img_paths = []
        image_dir = os.path.join(self.base_dir, 'Annotation')
        for cls_name in sorted(os.listdir(image_dir)):
            cls_dir = os.path.join(image_dir, cls_name)
            for name in sorted(os.listdir(cls_dir)):
                img_path = os.path.join(
                    self.base_dir, 'Data', cls_name, name[:-4] + '.JPEG')
                if os.path.exists(img_path):
                    self.paths.append(os.path.join(cls_dir, name))
                    self.img_paths.append(img_path)
                    self.cls_names.append(cls_name)

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label'), self._get_inst_anno)
        self.add_getter('img_label', self._get_img_label)
        if not return_img_label:
            self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.paths)

    def _get_image(self, i):
        img = read_image(self.img_paths[i], color=True)
        return img

    def _get_inst_anno(self, i):
        bbox, label, _ = parse_voc_bbox_annotation(
            self.paths[i], imagenet_full_bbox_synset_ids,
            skip_names_not_in_label_names=False)
        return bbox, label

    def _get_img_label(self, i):
        label = imagenet_full_bbox_synset_ids.index(
            self.cls_names[i])
        return np.array([label], dtype=np.int32)
