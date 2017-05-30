from collections import defaultdict
from collections import Counter
import json
import os
import pickle
import six
import string

import chainer
from chainer.dataset import download
import numpy as np

from chainercv.datasets.visual_genome.visual_genome_utils import root
from chainercv.datasets.visual_genome.visual_genome_utils import \
    get_extract_data
from chainercv.datasets.visual_genome.visual_genome_utils import \
    get_region_descriptions
from chainercv.datasets.visual_genome.visual_genome_utils import \
    VisualGenomeDatasetBase
from chainercv.transforms import resize
from chainercv.transforms import resize_bbox


def get_region_ids(region_descriptions_path):

    """Returns a dict that maps image ids to region ids.

    For each image id, there is a list of region ids.

    """
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'region_ids.pkl')

    def creator(path):
        print('Caching region IDs')
        region_ids = defaultdict(list)
        with open(region_descriptions_path) as f:
            region_descriptions = json.load(f)
            for region_description in region_descriptions:
                for region in region_description['regions']:
                    img_id = region['image_id']
                    region_id = region['region_id']
                    region_ids[img_id].append(region_id)
        pickle.dump(dict(region_ids), open(base_path, 'wb'))
        return region_ids

    def loader(path):
        return pickle.load(open(base_path, 'rb'))

    return download.cache_or_load_file(base_path, creator, loader)


def get_regions(region_descriptions_path):

    """Returns a dict that maps region ids to a bbox tuples.

    Each bbox tuple is defined as (x_min, y_min, x_max, y_max).

    """
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'regions.pkl')

    def creator(path):
        print('Caching region bounding boxes...')
        regions = {}
        with open(region_descriptions_path) as f:
            region_descriptions = json.load(f)
            for region_description in region_descriptions:
                for region in region_description['regions']:
                    region_id = r['region_id']  # int
                    xmin = region['x']
                    ymin = region['y']
                    xmax = xmin + region['width']
                    ymax = ymin + region['height']
                    regions[region_id] = (xmin, ymin, xmax, ymax)
        pickle.dump(regions, open(base_path, 'wb'))
        return regions

    def loader(path):
        return pickle.load(open(base_path, 'rb'))

    return download.cache_or_load_file(base_path, creator, loader)


def get_phrases(region_descriptions_path, min_token_instances):

    """Return a dict that maps a region_id to its corresponding phrase.

    Phrases in this case are represented as a list of ints of word_ids

    """
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root,
                             'phrases_{}.pkl'.format(min_token_instances))

    def creator(path):
        print('Caching region phrases...')
        phrases = {}
        vocab = retrieve_word_vocabulary(region_descriptions_path,
                                         min_token_instances)
        with open(region_descriptions_path) as f:
            region_descriptions = json.load(f)
            for region_description in region_descriptions:
                for region in region_description['regions']:
                    region_id = region['region_id']
                    word_ids = []
                    for word in preprocess_phrase(region['phrase']).split():
                        if word not in vocab:
                            word = '<unk>'
                        word_id = vocab[word]
                        word_ids.append(word_id)
                    phrases[region_id] = word_ids
        pickle.dump(phrases, open(base_path, 'wb'))
        return phrases

    def loader(path):
        return pickle.load(open(base_path, 'rb'))

    return download.cache_or_load_file(base_path, creator, loader)


def get_vocabulary(region_descriptions_path='auto', min_token_instances=15):

    """Creates a vocabulary based on the region descriptions of Visual Genome.

    A vocabulary is a dictionary that maps each word (str) to its
    corresponding id (int). Rare words are treated as unknown words, i.e.
    <unk> and are excluded for the dictionary.

    Args:
        min_token_instances (int): When words appear less than this times, they
            will be treated as <unk>.

    Returns:
        dict: A dictionary mapping words to their corresponding ids.

    """
    if region_descriptions_path == 'auto':
        region_descriptions_path = get_region_descriptions()
    return retrieve_word_vocabulary(region_descriptions_path,
                                    min_token_instances)


def retrieve_word_vocabulary(region_descriptions_path, min_token_instances):
    def creator(path):
        print('Creating vocabulary (ignoring words that appear less than '
              '{} times)...'.format(min_token_instances))
        words = load_words(region_descriptions_path,
                           min_token_instances=min_token_instances, sort=True)
        vocab = {}  # word (str) -> word_id (int)
        index = 0
        with open(path, 'w') as f:
            for word in words:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
                    f.write(word + '\n')
        return vocab

    def loader(path):
        vocab = {}
        with open(path) as f:
            for i, word in enumerate(f):
                vocab[word.strip()] = i
        return vocab

    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root,
                             'vocab_{}.txt'.format(min_token_instances))
    return download.cache_or_load_file(base_path, creator, loader)


def load_words(region_descriptions_path, min_token_instances=None, sort=True):
    word_counts = Counter()
    with open(region_descriptions_path) as f:
        region_descriptions = json.load(f)
        for region_description in region_descriptions:
            for region in region_description['regions']:
                for word in preprocess_phrase(region['phrase']).split():
                    word_counts[word] += 1

    words = ['<unk>', '<eos>']
    for word, count in six.iteritems(word_counts):
        if min_token_instances is None or count >= min_token_instances:
            words.append(word)

    if sort:
        words = sorted(words)

    return words


def preprocess_phrase(phrase):
    """Preprocess a phrase similar to the DenseCap implementation.

    Certain non-ascii characters are replaced, punctuations are removed and all
    characeters are lower-cased.

    Please refer to the following implementation.
    https://github.com/jcjohnson/densecap/blob/master/preprocess.py

    Args:
        phrse (str): A phrase to process.

    Returns:
        str: A processed phrase.
    """
    replacements = {
        u'\xa2': u'cent',
        u'\xb0': u' degree',
        u'\xbd': u'half',
        u'\xe7': u'c',
        u'\xe8': u'e',
        u'\xe9': u'e',
        u'\xfb': u'u',
        u'\u2014': u'-',
        u'\u2026': u'',
        u'\u2122': u'',
    }

    for k, v in six.iteritems(replacements):
        phrase = phrase.replace(k, v)
    trans = str.maketrans('', '', string.punctuation)
    return str(phrase).lower().translate(trans)



class VisualGenomeRegionDescriptionsDataset(VisualGenomeDatasetBase):

    """Region description class for Visual Genome dataset.

    """

    def __init__(self, data_dir='auto', image_data='auto',
                 region_descriptions='auto', min_token_instances=15,
                 img_size=(720, 720)):
        super(VisualGenomeRegionDescriptionsDataset, self).__init__(
            data_dir=data_dir, image_data=image_data)

        if region_descriptions == 'auto':
            region_descriptions = get_region_descriptions()

        self.region_ids = get_region_ids(region_descriptions)
        self.regions = get_regions(region_descriptions)
        self.phrases = get_phrases(region_descriptions,
                                   min_token_instances=min_token_instances)
        self.img_size = img_size

    def get_example(self, i):
        img_id = self.get_image_id(i)
        img = self.get_image(img_id)

        regions = []
        phrases = []
        for region_id in self.region_ids[img_id]:
            regions.append(self.regions[region_id])
            phrases.append(self.phrases[region_id])
        regions = np.vstack(regions)

        if self.img_size is not None:
            h_orig, w_orig = img.shape[1:]
            img = resize(img, self.img_size)
            h, w = img.shape[1:]
            regions = resize_bbox(regions, (w_orig, h_orig), (w, h))

        return img, regions, phrases
