import os

from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/coco'
img_urls = {
    '2014': {
        'train': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
        'val': 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'
    },
    '2017': {
        'train': 'http://images.cocodataset.org/zips/train2017.zip',
        'val': 'http://images.cocodataset.org/zips/val2017.zip'
    }
}
instances_anno_urls = {
    '2014': {
        'train': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
        'instances_train-val2014.zip',
        'val': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
        'instances_train-val2014.zip',
        'valminusminival': 'https://dl.dropboxusercontent.com/s/'
        's3tw5zcg7395368/instances_valminusminival2014.json.zip',
        'minival': 'https://dl.dropboxusercontent.com/s/o43o90bna78omob/'
        'instances_minival2014.json.zip'
    },
    '2017': {
        'train': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2017.zip',
        'val': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2017.zip'
    }
}


panoptic_anno_url = 'http://images.cocodataset.org/annotations/' +\
    'panoptic_annotations_trainval2017.zip'


def get_coco(split, img_split, year, mode):
    data_dir = download.get_dataset_directory(root)
    annos_root = os.path.join(data_dir, 'annotations')
    img_root = os.path.join(data_dir, 'images')
    created_img_root = os.path.join(
        img_root, '{}{}'.format(img_split, year))
    if mode == 'instances':
        url = img_urls[year][img_split]
        anno_url = instances_anno_urls[year][split]
        anno_path = os.path.join(
            annos_root, 'instances_{}{}.json'.format(split, year))
    elif mode == 'panoptic':
        url = img_urls[year][img_split]
        anno_url = panoptic_anno_url
        anno_path = os.path.join(
            annos_root, 'panoptic_{}{}.json'.format(split, year))

    if not os.path.exists(created_img_root):
        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, img_root, ext)
    if not os.path.exists(anno_path):
        download_file_path = utils.cached_download(anno_url)
        ext = os.path.splitext(anno_url)[1]
        if split in ['train', 'val']:
            utils.extractall(download_file_path, data_dir, ext)
        elif split in ['valminusminival', 'minival']:
            utils.extractall(download_file_path, annos_root, ext)

    if mode == 'panoptic':
        pixelmap_path = os.path.join(
            annos_root, 'panoptic_{}{}'.format(split, year))
        if not os.path.exists(pixelmap_path):
            utils.extractall(pixelmap_path + '.zip', annos_root, '.zip')
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

# annos = json.load(open('panoptic_val2017'))
# label_names = [cat['name'] for cat in annos['categories']]
coco_semantic_segmentation_label_names = (
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
    'toothbrush',
    'banner',
    'blanket',
    'bridge',
    'cardboard',
    'counter',
    'curtain',
    'door-stuff',
    'floor-wood',
    'flower',
    'fruit',
    'gravel',
    'house',
    'light',
    'mirror-stuff',
    'net',
    'pillow',
    'platform',
    'playingfield',
    'railroad',
    'river',
    'road',
    'roof',
    'sand',
    'sea',
    'shelf',
    'snow',
    'stairs',
    'tent',
    'towel',
    'wall-brick',
    'wall-stone',
    'wall-tile',
    'wall-wood',
    'water-other',
    'window-blind',
    'window-other',
    'tree-merged',
    'fence-merged',
    'ceiling-merged',
    'sky-other-merged',
    'cabinet-merged',
    'table-merged',
    'floor-other-merged',
    'pavement-merged',
    'mountain-merged',
    'grass-merged',
    'dirt-merged',
    'paper-merged',
    'food-other-merged',
    'building-other-merged',
    'rock-merged',
    'wall-other-merged',
    'rug-merged')


coco_instance_segmentation_label_names = coco_bbox_label_names
