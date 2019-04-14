import numpy as np
import unittest

import chainer

from chainercv.utils import assert_is_semantic_segmentation_link
from chainercv.utils import testing


class SemanticSegmentationLink(chainer.Link):

    def predict(self, imgs):
        labels = []

        for img in imgs:
            labels.append(np.random.randint(
                0, 21, size=img.shape[1:]).astype(np.int32))

        return labels


class InvalidPredictionSizeLink(SemanticSegmentationLink):

    def predict(self, imgs):
        labels = super(
            InvalidPredictionSizeLink, self).predict(imgs)
        return labels[1:]


class InvalidLabelSizeLink(SemanticSegmentationLink):

    def predict(self, imgs):
        labels = super(
            InvalidLabelSizeLink, self).predict(imgs)
        return [label[1:] for label in labels]


class InvalidLabelValueLink(SemanticSegmentationLink):

    def predict(self, imgs):
        labels = super(
            InvalidLabelValueLink, self).predict(imgs)
        return [label + 1000 for label in labels]


@testing.parameterize(
    {'link': SemanticSegmentationLink(), 'valid': True},
    {'link': InvalidPredictionSizeLink(), 'valid': False},
    {'link': InvalidLabelSizeLink(), 'valid': False},
    {'link': InvalidLabelValueLink(), 'valid': False},
)
class TestAssertIsSemanticSegmentationLink(unittest.TestCase):

    def test_assert_is_semantic_segmentation_link(self):
        if self.valid:
            assert_is_semantic_segmentation_link(self.link, 21)
        else:
            with self.assertRaises(AssertionError):
                assert_is_semantic_segmentation_link(self.link, 21)


testing.run_module(__name__, __file__)
