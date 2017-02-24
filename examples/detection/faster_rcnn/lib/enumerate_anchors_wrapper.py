from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper

from region_proporsal.anchor_target_layer import AnchorTargetLayer


class EnumerateAnchorsWrapper(DatasetWrapper):

    def __init__(self, dataset, feature_shape):
        super(EnumerateAnchorsWrapper, self).__init__(dataset)
        self.anchor_target_layer = AnchorTargetLayer()
        self.feature_shape = feature_shape

    def _get_example(self, in_data):
        img, bboxes = in_data
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights =\
            self.anchor_target_layer(bboxes, feature_shape, img.shape[2:])
        return img, bboxes, labels, 


if __name__ == '__main__':
    from chainer_cv.datasets import VOCDetectionDataset
    from chainer_cv.wrappers import PadWrapper
    from chainer_cv.wrappers import RandomMirrorWrapper

    dataset = VOCDetectionDataset()
    # wrapped = PadWrapper(

    dataset


