import os
import shelve
import tempfile


def extend(dataset, transform, method_name='get_example'):
    method = getattr(dataset, method_name)

    def _extended(*args, **kwargs):
        in_data = method(*args, **kwargs)
        return transform(in_data)
    setattr(dataset, method_name, _extended)


def extend_cache(dataset, transform, method_name='get_example'):
    filename = os.path.join(tempfile.mkdtemp(), 'cache.db')
    cache = shelve.open(filename, protocol=2)

    method = getattr(dataset, method_name)

    def _extended(self, i):
        key = str(i)
        if key not in cache:
            in_data = method(self, i)
            cache[key] = transform(in_data)
        return cache[key]
    setattr(dataset, method_name, _extended)


if __name__ == '__main__':
    from chainercv.datasets import VOCSemanticSegmentationDataset
    from chainercv.transforms.random_crop_transform import random_crop
    dataset = VOCSemanticSegmentationDataset()

    def transform(in_data):
        return random_crop(in_data, (None, 256, 256))

    extend(dataset, transform)
    img, label = dataset.get_example(0)
