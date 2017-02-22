import copy
import numpy as np
import os.path as osp
import warnings

import chainer
from chainer.dataset import convert
from chainer import reporter

try:
    import sklearn.neighbors
    _available = True

except ImportError:
    _available = False


def _check_available():
    if not _available:
        warnings.warn('scikit-learn is not installed on your environment, '
                      'so an extension MeasureKRetrieval can not be used.'
                      'Please install scikit-learn.\n\n'
                      '  $ pip install scikit-learn\n')


class MeasureKRetrieval(chainer.training.extension.Extension):

    """Measure Recall@K score for image retrieval task.

    The element in the second index of the returned tuple will be class id.

    Args:
        iterator (chainer.iterators.Iterator): Dataset iterator for the
            images to be embedded. The element in the second index of the
            returned tuple will be class id.
        features_file (string): A file where embedded features are saved.
        ks (list of ints)
        window_size (int)
        n_jobs (int): number of processes run in parallel to compute the
            k-nearest neighbor.
    """

    invoke_before_training = False
    priority = chainer.training.extension.PRIORITY_WRITER

    def __init__(self, iterator, features_file, ks,
                 window_size=10000, n_jobs=6):
        _check_available()
        if not _available:
            return

        self.iterator = iterator
        self.features_file = features_file
        self.ks = ks

        self.window_size = window_size

        n_jobs = n_jobs
        self.max_k = np.max(ks)
        self.nbrs = sklearn.neighbors.NearestNeighbors(n_jobs=n_jobs)

    def __call__(self, trainer):
        if not _available:
            return
        features_file = osp.join(trainer.out, self.features_file)

        iterator = copy.copy(self.iterator)
        features = np.load(features_file)
        optimizer = trainer.updater.get_optimizer('main')

        classes = []
        for v in iterator:
            arrays = convert.concat_examples(v)
            classes.append(arrays[1])
        classes = np.concatenate(classes, axis=0)

        if features.shape[0] != classes.shape[0]:
            raise ValueError(
                'batch size of features and the class array differ')

        n = features.shape[0]
        n_match = {k: [] for k in self.ks}

        self.nbrs.fit(features)
        for i in range(0, n, self.window_size):
            the_slice = slice(
                i * self.window_size, (i + 1) * self.window_size)
            src_features = features[the_slice]
            src_classes = classes[the_slice]

            indices = self.nbrs.kneighbors(
                src_features, n_neighbors=self.max_k + 1,
                return_distance=False)
            indices = indices[:, 1:]
            knbr_classes = classes[indices]  # (window, max_k)

            match = knbr_classes == src_classes[:, None]

            for k in self.ks:
                n_match_k = np.any(match[:, :k], axis=1)
                n_match[k].append(n_match_k)

        for k in self.ks:
            n_match[k] = np.concatenate(n_match[k])

            reporter.report({'recall@{}'.format(k): np.mean(n_match[k])},
                            optimizer.target)
