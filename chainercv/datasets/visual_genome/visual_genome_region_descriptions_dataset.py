from collections import Counter
from collections import defaultdict
import json
import os
import pickle
import six
import string

from chainer.dataset import download
import numpy as np

from chainercv.datasets.visual_genome.visual_genome_utils import \
    get_region_descriptions
from chainercv.datasets.visual_genome.visual_genome_utils import root
from chainercv.datasets.visual_genome.visual_genome_utils import \
    VisualGenomeDatasetBase


def get_vocabulary(region_descriptions='auto', min_token_instances=15):
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
    if region_descriptions == 'auto':
        region_descriptions = get_region_descriptions()
    return _create_word_vocabulary(region_descriptions, min_token_instances)


class VisualGenomeRegionDescriptionsDataset(VisualGenomeDatasetBase):
    """Region description class for Visual Genome dataset.

    """

    def __init__(self, data_dir='auto', image_data='auto',
                 region_descriptions='auto', min_token_instances=15,
                 max_token_length=15, img_size=(720, 720)):
        super(VisualGenomeRegionDescriptionsDataset, self).__init__(
            data_dir=data_dir, image_data=image_data)

        if region_descriptions == 'auto':
            region_descriptions = get_region_descriptions()

        self.img_size = img_size

        self.region_ids = _get_region_ids(region_descriptions)
        self.regions = _get_regions(region_descriptions)
        self.phrases = _get_phrases(region_descriptions, min_token_instances,
                                    max_token_length)

    def get_example(self, i):
        img_id = self.get_image_id(i)
        img = self.get_image(img_id)

        regions = []
        phrases = []
        for region_id in self.region_ids[img_id]:
            # Phrases that are too long are excluded in the preprocessing,
            # so only include regions with actual phrases
            if region_id in self.phrases:
                regions.append(self.regions[region_id])
                phrases.append(self.phrases[region_id])
        regions = np.vstack(regions).astype(np.float32)
        phrases = np.vstack(phrases).astype(np.int32)

        return img, regions, phrases


def _get_region_ids(region_descriptions_path):
    """Image ID (int) -> Region IDs (list of int).

    """
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'region_ids.pkl')

    def creator(path):
        print('Caching Visual Genome region IDs...')
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


def _get_regions(region_descriptions_path):
    """Region ID (int) -> Region bounding box (xmin, ymin).

    """
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'regions.pkl')

    def creator(path):
        print('Caching Visual Genome region bounding boxes...')
        regions = {}
        with open(region_descriptions_path) as f:
            region_descriptions = json.load(f)
            for region_description in region_descriptions:
                for region in region_description['regions']:
                    region_id = region['region_id']  # int
                    x_min = region['x']
                    y_min = region['y']
                    x_max = x_min + region['width']
                    y_max = y_min + region['height']
                    regions[region_id] = (y_min, x_min, y_max, x_max)
        pickle.dump(regions, open(base_path, 'wb'))
        return regions

    def loader(path):
        return pickle.load(open(base_path, 'rb'))

    return download.cache_or_load_file(base_path, creator, loader)


def _get_phrases(region_descriptions_path, min_token_instances,
                 max_token_length):
    """Region ID (int) -> Phrase (list of int).

    """
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root,
                             'phrases_{}.pkl'.format(min_token_instances))

    def creator(path):
        print('Caching Visual Genome region descriptions...')
        phrases = {}
        vocab = _create_word_vocabulary(region_descriptions_path,
                                        min_token_instances)
        with open(region_descriptions_path) as f:
            region_descriptions = json.load(f)
            for region_description in region_descriptions:
                for region in region_description['regions']:
                    region_id = region['region_id']
                    tokens = _preprocess_phrase(region['phrase']).split()
                    if max_token_length > 0 and \
                            len(tokens) < max_token_length - 1:
                        # <bos>, t1, t2,..., tn, <eos>, <eos>,..., <eos>
                        phrase = np.empty(max_token_length, dtype=np.int32)
                        phrase.fill(vocab['<eos>'])
                        phrase[0] = vocab['<bos>']
                        for i, token in enumerate(tokens, 1):
                            if token not in vocab:
                                token = '<unk>'
                            token_id = vocab[token]
                            phrase[i] = token_id
                        phrases[region_id] = phrase

        pickle.dump(phrases, open(base_path, 'wb'))
        return phrases

    def loader(path):
        return pickle.load(open(base_path, 'rb'))

    return download.cache_or_load_file(base_path, creator, loader)


def _create_word_vocabulary(region_descriptions_path, min_token_instances):
    """Word (str) -> Word ID (int).

    """
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root,
                             'vocab_{}.txt'.format(min_token_instances))

    def creator(path):
        print('Creating vocabulary from region descriptions (ignoring words '
              'that appear less than {} times)...'.format(min_token_instances))
        words = _load_words(region_descriptions_path,
                            min_token_instances=min_token_instances)
        vocab = {}
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

    return download.cache_or_load_file(base_path, creator, loader)


def _load_words(region_descriptions_path, min_token_instances):
    # Count the number of occurrences for each word in all region descriptions
    # to only include those words that appear at least a few times
    word_counts = Counter()
    with open(region_descriptions_path) as f:
        region_descriptions = json.load(f)
        for region_description in region_descriptions:
            for region in region_description['regions']:
                for word in _preprocess_phrase(region['phrase']).split():
                    word_counts[word] += 1

    words = []
    for word, count in six.iteritems(word_counts):
        if min_token_instances is None or count >= min_token_instances:
            words.append(word)

    # Sort to make sure that word orders are consistent
    words = sorted(words)

    words.insert(0, '<unk>')
    words.insert(0, '<eos>')
    words.insert(0, '<bos>')

    return words


def _preprocess_phrase(phrase):
    # Preprocess phrases according to the DenseCap implementation
    # https://github.com/jcjohnson/densecap/blob/master/preprocess.py
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
