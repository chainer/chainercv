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
        'train': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2014.zip',
        'val': 'http://images.cocodataset.org/annotations/'
        'annotations_trainval2014.zip',
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
    img_url = img_urls[year][img_split]
    if mode == 'instances':
        anno_url = instances_anno_urls[year][split]
        anno_path = os.path.join(
            annos_root, 'instances_{}{}.json'.format(split, year))
    elif mode == 'panoptic':
        anno_url = panoptic_anno_url
        anno_path = os.path.join(
            annos_root, 'panoptic_{}{}.json'.format(split, year))

    if not os.path.exists(created_img_root):
        download_file_path = utils.cached_download(img_url)
        ext = os.path.splitext(img_url)[1]
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

# https://raw.githubusercontent.com/cocodataset/panopticapi/master/
# panoptic_coco_categories.json
coco_semantic_segmentation_label_colors = (
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
    (175, 116, 175),
    (250, 0, 30),
    (165, 42, 42),
    (255, 77, 255),
    (0, 226, 252),
    (182, 182, 255),
    (0, 82, 0),
    (120, 166, 157),
    (110, 76, 0),
    (174, 57, 255),
    (199, 100, 0),
    (72, 0, 118),
    (255, 179, 240),
    (0, 125, 92),
    (209, 0, 151),
    (188, 208, 182),
    (0, 220, 176),
    (255, 99, 164),
    (92, 0, 73),
    (133, 129, 255),
    (78, 180, 255),
    (0, 228, 0),
    (174, 255, 243),
    (45, 89, 255),
    (134, 134, 103),
    (145, 148, 174),
    (255, 208, 186),
    (197, 226, 255),
    (171, 134, 1),
    (109, 63, 54),
    (207, 138, 255),
    (151, 0, 95),
    (9, 80, 61),
    (84, 105, 51),
    (74, 65, 105),
    (166, 196, 102),
    (208, 195, 210),
    (255, 109, 65),
    (0, 143, 149),
    (179, 0, 194),
    (209, 99, 106),
    (5, 121, 0),
    (227, 255, 205),
    (147, 186, 208),
    (153, 69, 1),
    (3, 95, 161),
    (163, 255, 0),
    (119, 0, 170),
    (0, 182, 199),
    (0, 165, 120),
    (183, 130, 88),
    (95, 32, 0),
    (130, 114, 135),
    (110, 129, 133),
    (166, 74, 118),
    (219, 142, 185),
    (79, 210, 114),
    (178, 90, 62),
    (65, 70, 15),
    (127, 167, 115),
    (59, 105, 106),
    (142, 108, 45),
    (196, 172, 0),
    (95, 54, 80),
    (128, 76, 255),
    (201, 57, 1),
    (246, 0, 122),
    (191, 162, 208),
    (255, 255, 128),
    (147, 211, 203),
    (150, 100, 100),
    (168, 171, 172),
    (146, 112, 198),
    (210, 170, 100),
    (92, 136, 89),
    (218, 88, 184),
    (241, 129, 0),
    (217, 17, 255),
    (124, 74, 181),
    (70, 70, 70),
    (255, 228, 255),
    (154, 208, 0),
    (193, 0, 92),
    (76, 91, 113),
    (255, 180, 195),
    (106, 154, 176),
    (230, 150, 140),
    (60, 143, 255),
    (128, 64, 128),
    (92, 82, 55),
    (254, 212, 124),
    (73, 77, 174),
    (255, 160, 98),
    (255, 255, 255),
    (104, 84, 109),
    (169, 164, 131),
    (225, 199, 255),
    (137, 54, 74),
    (135, 158, 223),
    (7, 246, 231),
    (107, 255, 200),
    (58, 41, 149),
    (183, 121, 142),
    (255, 73, 97),
    (107, 142, 35),
    (190, 153, 153),
    (146, 139, 141),
    (70, 130, 180),
    (134, 199, 156),
    (209, 226, 140),
    (96, 36, 108),
    (96, 96, 96),
    (64, 170, 64),
    (152, 251, 152),
    (208, 229, 228),
    (206, 186, 171),
    (152, 161, 64),
    (116, 112, 0),
    (0, 114, 143),
    (102, 102, 156),
    (250, 141, 255))


coco_instance_segmentation_label_names = coco_bbox_label_names


coco_keypoint_names = {
    0: [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
}
