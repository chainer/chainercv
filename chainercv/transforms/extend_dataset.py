def extend(dataset, transform, method_name='get_example'):
    method = getattr(dataset, method_name)
    def _extended(*args, **kwargs):
        in_data = method(*args, **kwargs)
        return transform(in_data)
    setattr(dataset, method_name, _extended)


if __name__ == '__main__':
    from chainercv.datasets import VOCSemanticSegmentationDataset
    from chainercv.transforms.random_crop_transform import random_crop
    dataset = VOCSemanticSegmentationDataset()

    def transform(in_data):
        return random_crop(in_data, (None, 256, 256))

    extend(dataset, transform)
    img, label = dataset.get_example(0)
