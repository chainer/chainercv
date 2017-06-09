import numpy as np
import os

import chainer
from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/cub'
url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'\
    'CUB_200_2011.tgz'
mask_url = 'http://www.vision.caltech.edu/visipedia-data/'\
    'CUB-200-2011/segmentations.tgz'


def get_cub():
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'CUB_200_2011')
    if os.path.exists(base_path):
        # skip downloading
        return base_path

    download_file_path = utils.cached_download(url)
    ext = os.path.splitext(url)[1]
    utils.extractall(download_file_path, data_root, ext)
    return base_path


def get_cub_mask():
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'segmentations')
    if os.path.exists(base_path):
        # skip downloading
        return base_path

    download_file_path_mask = utils.cached_download(mask_url)
    ext_mask = os.path.splitext(mask_url)[1]
    utils.extractall(
        download_file_path_mask, data_root, ext_mask)
    return base_path


class CUBDatasetBase(chainer.dataset.DatasetMixin):

    """Base class for CUB dataset.

    """

    def __init__(self, data_dir='auto', mask_dir='auto', crop_bbox=True):
        if data_dir == 'auto':
            data_dir = get_cub()
        if mask_dir == 'auto':
            mask_dir = get_cub_mask()
        self.data_dir = data_dir
        self.mask_dir = mask_dir

        imgs_file = os.path.join(data_dir, 'images.txt')
        bboxes_file = os.path.join(data_dir, 'bounding_boxes.txt')

        self.filenames = [
            line.strip().split()[1] for line in open(imgs_file)]

        # (x_min, y_min, width, height)
        bboxes = np.array([
            tuple(map(float, line.split()[1:5]))
            for line in open(bboxes_file)])
        # (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
        bboxes[:, 2:] += bboxes[:, :2]
        # (x_min, y_min, width, height) -> (y_min, x_min, y_max, x_max)
        bboxes[:] = bboxes[:, [1, 0, 3, 2]]
        self.bboxes = bboxes.astype(np.float32)

        self.crop_bbox = crop_bbox

    def __len__(self):
        return len(self.filenames)


cub_label_names = (
    'Black_footed_Albatross',
    'Laysan_Albatross',
    'Sooty_Albatross',
    'Groove_billed_Ani',
    'Crested_Auklet',
    'Least_Auklet',
    'Parakeet_Auklet',
    'Rhinoceros_Auklet',
    'Brewer_Blackbird',
    'Red_winged_Blackbird',
    'Rusty_Blackbird',
    'Yellow_headed_Blackbird',
    'Bobolink',
    'Indigo_Bunting',
    'Lazuli_Bunting',
    'Painted_Bunting',
    'Cardinal',
    'Spotted_Catbird',
    'Gray_Catbird',
    'Yellow_breasted_Chat',
    'Eastern_Towhee',
    'Chuck_will_Widow',
    'Brandt_Cormorant',
    'Red_faced_Cormorant',
    'Pelagic_Cormorant',
    'Bronzed_Cowbird',
    'Shiny_Cowbird',
    'Brown_Creeper',
    'American_Crow',
    'Fish_Crow',
    'Black_billed_Cuckoo',
    'Mangrove_Cuckoo',
    'Yellow_billed_Cuckoo',
    'Gray_crowned_Rosy_Finch',
    'Purple_Finch',
    'Northern_Flicker',
    'Acadian_Flycatcher',
    'Great_Crested_Flycatcher',
    'Least_Flycatcher',
    'Olive_sided_Flycatcher',
    'Scissor_tailed_Flycatcher',
    'Vermilion_Flycatcher',
    'Yellow_bellied_Flycatcher',
    'Frigatebird',
    'Northern_Fulmar',
    'Gadwall',
    'American_Goldfinch',
    'European_Goldfinch',
    'Boat_tailed_Grackle',
    'Eared_Grebe',
    'Horned_Grebe',
    'Pied_billed_Grebe',
    'Western_Grebe',
    'Blue_Grosbeak',
    'Evening_Grosbeak',
    'Pine_Grosbeak',
    'Rose_breasted_Grosbeak',
    'Pigeon_Guillemot',
    'California_Gull',
    'Glaucous_winged_Gull',
    'Heermann_Gull',
    'Herring_Gull',
    'Ivory_Gull',
    'Ring_billed_Gull',
    'Slaty_backed_Gull',
    'Western_Gull',
    'Anna_Hummingbird',
    'Ruby_throated_Hummingbird',
    'Rufous_Hummingbird',
    'Green_Violetear',
    'Long_tailed_Jaeger',
    'Pomarine_Jaeger',
    'Blue_Jay',
    'Florida_Jay',
    'Green_Jay',
    'Dark_eyed_Junco',
    'Tropical_Kingbird',
    'Gray_Kingbird',
    'Belted_Kingfisher',
    'Green_Kingfisher',
    'Pied_Kingfisher',
    'Ringed_Kingfisher',
    'White_breasted_Kingfisher',
    'Red_legged_Kittiwake',
    'Horned_Lark',
    'Pacific_Loon',
    'Mallard',
    'Western_Meadowlark',
    'Hooded_Merganser',
    'Red_breasted_Merganser',
    'Mockingbird',
    'Nighthawk',
    'Clark_Nutcracker',
    'White_breasted_Nuthatch',
    'Baltimore_Oriole',
    'Hooded_Oriole',
    'Orchard_Oriole',
    'Scott_Oriole',
    'Ovenbird',
    'Brown_Pelican',
    'White_Pelican',
    'Western_Wood_Pewee',
    'Sayornis',
    'American_Pipit',
    'Whip_poor_Will',
    'Horned_Puffin',
    'Common_Raven',
    'White_necked_Raven',
    'American_Redstart',
    'Geococcyx',
    'Loggerhead_Shrike',
    'Great_Grey_Shrike',
    'Baird_Sparrow',
    'Black_throated_Sparrow',
    'Brewer_Sparrow',
    'Chipping_Sparrow',
    'Clay_colored_Sparrow',
    'House_Sparrow',
    'Field_Sparrow',
    'Fox_Sparrow',
    'Grasshopper_Sparrow',
    'Harris_Sparrow',
    'Henslow_Sparrow',
    'Le_Conte_Sparrow',
    'Lincoln_Sparrow',
    'Nelson_Sharp_tailed_Sparrow',
    'Savannah_Sparrow',
    'Seaside_Sparrow',
    'Song_Sparrow',
    'Tree_Sparrow',
    'Vesper_Sparrow',
    'White_crowned_Sparrow',
    'White_throated_Sparrow',
    'Cape_Glossy_Starling',
    'Bank_Swallow',
    'Barn_Swallow',
    'Cliff_Swallow',
    'Tree_Swallow',
    'Scarlet_Tanager',
    'Summer_Tanager',
    'Artic_Tern',
    'Black_Tern',
    'Caspian_Tern',
    'Common_Tern',
    'Elegant_Tern',
    'Forsters_Tern',
    'Least_Tern',
    'Green_tailed_Towhee',
    'Brown_Thrasher',
    'Sage_Thrasher',
    'Black_capped_Vireo',
    'Blue_headed_Vireo',
    'Philadelphia_Vireo',
    'Red_eyed_Vireo',
    'Warbling_Vireo',
    'White_eyed_Vireo',
    'Yellow_throated_Vireo',
    'Bay_breasted_Warbler',
    'Black_and_white_Warbler',
    'Black_throated_Blue_Warbler',
    'Blue_winged_Warbler',
    'Canada_Warbler',
    'Cape_May_Warbler',
    'Cerulean_Warbler',
    'Chestnut_sided_Warbler',
    'Golden_winged_Warbler',
    'Hooded_Warbler',
    'Kentucky_Warbler',
    'Magnolia_Warbler',
    'Mourning_Warbler',
    'Myrtle_Warbler',
    'Nashville_Warbler',
    'Orange_crowned_Warbler',
    'Palm_Warbler',
    'Pine_Warbler',
    'Prairie_Warbler',
    'Prothonotary_Warbler',
    'Swainson_Warbler',
    'Tennessee_Warbler',
    'Wilson_Warbler',
    'Worm_eating_Warbler',
    'Yellow_Warbler',
    'Northern_Waterthrush',
    'Louisiana_Waterthrush',
    'Bohemian_Waxwing',
    'Cedar_Waxwing',
    'American_Three_toed_Woodpecker',
    'Pileated_Woodpecker',
    'Red_bellied_Woodpecker',
    'Red_cockaded_Woodpecker',
    'Red_headed_Woodpecker',
    'Downy_Woodpecker',
    'Bewick_Wren',
    'Cactus_Wren',
    'Carolina_Wren',
    'House_Wren',
    'Marsh_Wren',
    'Rock_Wren',
    'Winter_Wren',
    'Common_Yellowthroat',
)
