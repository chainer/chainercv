from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper

from region_proporsal.anchor_target_layer import AnchorTargetLayer


class EnumerateAnchorsWrapper(DatasetWrapper):

    def __init__(self, dataset):
        self.anchor_target_layer = AnchorTargetLayer()

    


if __name__ == '__main__':
    from chainer_cv.datasets import VOCDetectionDataset
    from chainer_cv.wrappers import PadWrapper
    from chainer_cv.wrappers import RandomMirrorWrapper

    dataset = VOCDetectionDataset()
    # wrapped = PadWrapper(

    dataset


