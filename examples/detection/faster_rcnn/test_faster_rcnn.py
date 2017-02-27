import mock
import numpy as np

import chainer

from chainer_cv.datasets import VOCDetectionDataset
from chainer_cv.extensions import DetectionVisReport
from chainer_cv.wrappers import bbox_resize_hook
from chainer_cv.wrappers import SubtractWrapper
from chainer_cv.wrappers import ResizeWrapper
from chainer_cv.wrappers import output_shape_soft_min_hard_max

from faster_rcnn import FasterRCNN


if __name__ == '__main__':
    test_data = VOCDetectionDataset(mode='train', use_cache=True, year='2007',
                                    bgr=True)
    wrappers = [
        lambda d: SubtractWrapper(
            d, value=np.array([103.939, 116.779, 123.68])),
        lambda d: ResizeWrapper(
            d, preprocess_idx=0,
            output_shape=output_shape_soft_min_hard_max(600, 1200),
            hook=bbox_resize_hook(1)),
    ]
    for wrapper in wrappers:
        test_data = wrapper(test_data)

    model = FasterRCNN()
    chainer.serializers.load_npz('VGG16_faster_rcnn_final.model', model)
    trainer = mock.MagicMock()
    trainer.out = 'result'
    trainer.updater.iteration = 0

    extension = DetectionVisReport(
        [3, 4, 5, 6, 7, 8],
        test_data, model, predict_func=model.predict_bboxes)
    extension(trainer)
