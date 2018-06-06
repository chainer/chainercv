import os

from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/coco'
img_urls = {
    'train': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
    'val': 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'
}
anno_urls = {
    'train': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
    'instances_train-val2014.zip',
    'val': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
    'instances_train-val2014.zip',
    'valminusminival': 'https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/'
    'instances_valminusminival2014.json.zip',
    'minival': 'https://dl.dropboxusercontent.com/s/o43o90bna78omob/'
    'instances_minival2014.json.zip'
}


def get_coco(split, img_split):
    url = img_urls[img_split]
    data_dir = download.get_dataset_directory(root)
    img_root = os.path.join(data_dir, 'images')
    created_img_root = os.path.join(img_root, '{}2014'.format(img_split))
    annos_root = os.path.join(data_dir, 'annotations')
    anno_path = os.path.join(annos_root, 'instances_{}2014.json'.format(split))
    if not os.path.exists(created_img_root):
        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, img_root, ext)
    if not os.path.exists(anno_path):
        anno_url = anno_urls[split]
        download_file_path = utils.cached_download(anno_url)
        ext = os.path.splitext(anno_url)[1]
        utils.extractall(download_file_path, annos_root, ext)
    return data_dir


# How you can get the labels
# >>> from pycocotools.coco import COCO
# >>> coco = COCO('instances_train2014.json')
# >>> cat_dict = coco.loadCats(coco.getCatIds())
# >>> coco_bbox_label_names = [c['name'] for c in cat_dict]
coco_bbox_label_names = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')
