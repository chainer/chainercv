import numpy as np
import unittest

import chainer

from chainercv.utils import assert_is_detection_link
from chainercv.utils import generate_random_bbox
from chainercv.utils import testing


class DetectionLink(chainer.Link):

    def predict(self, imgs):
        bboxes = []
        labels = []
        scores = []

        for img in imgs:
            n_bbox = np.random.randint(1, 10)
            bboxes.append(generate_random_bbox(
                n_bbox, img.shape[1:], 4, 12))
            labels.append(np.random.randint(
                0, 20, size=n_bbox).astype(np.int32))
            scores.append(np.random.uniform(
                0, 1, size=n_bbox).astype(np.float32))

        return bboxes, labels, scores


class InvalidPredictionSizeLink(DetectionLink):

    def predict(self, imgs):
        bboxes, labels, scores = super(
            InvalidPredictionSizeLink, self).predict(imgs)
        return bboxes[1:], labels[1:], scores[1:]


class InvalidLabelSizeLink(DetectionLink):

    def predict(self, imgs):
        bboxes, labels, scores = super(
            InvalidLabelSizeLink, self).predict(imgs)
        return bboxes, [label[1:] for label in labels], scores


class InvalidLabelValueLink(DetectionLink):

    def predict(self, imgs):
        bboxes, labels, scores = super(
            InvalidLabelValueLink, self).predict(imgs)
        return bboxes, [label + 1000 for label in labels], scores


class InvalidScoreSizeLink(DetectionLink):

    def predict(self, imgs):
        bboxes, labels, scores = super(
            InvalidScoreSizeLink, self).predict(imgs)
        return bboxes, labels, [score[1:] for score in scores]


@testing.parameterize(
    {'link': DetectionLink(), 'valid': True},
    {'link': InvalidPredictionSizeLink(), 'valid': False},
    {'link': InvalidLabelSizeLink(), 'valid': False},
    {'link': InvalidLabelValueLink(), 'valid': False},
    {'link': InvalidScoreSizeLink(), 'valid': False},
)
class TestAssertIsDetectionLink(unittest.TestCase):

    def test_assert_is_detection_link(self):
        if self.valid:
            assert_is_detection_link(self.link, 20)
        else:
            with self.assertRaises(AssertionError):
                assert_is_detection_link(self.link, 20)


testing.run_module(__name__, __file__)
