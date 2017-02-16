import chainer
from chainer_cv.datasets.dataset_wrapper import DatasetWrapper


class CacheDatasetWrapper(DatasetWrapper):

    def __init__(self, dataset):
        self.dataset = dataset

    def get_example(self, i):
        pass
