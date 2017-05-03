import numpy as np
import unittest

import chainer
from chainer import testing

from chainercv.utils import StubLink


@testing.parameterize(*testing.product({
    'shapes': [(3,), ((4, 5),), (3, (4, 5))],
    'value': ['uniform', 1, 1.5],
    'dtype': [np.float32, np.int32],
}))
class TestStubLink(unittest.TestCase):

    def setUp(self):
        self.link = StubLink(*self.shapes, value=self.value, dtype=self.dtype)

    def test_stub_link(self):
        self.assertIsInstance(self.link, chainer.Link)

        values = self.link('ignored', -1, 'values', 1.0)

        if len(self.shapes) == 1:
            values = (values,)

        for shape, value in zip(self.shapes, values):
            self.assertIsInstance(value, chainer.Variable)
            if isinstance(shape, int):
                shape = (shape,)
            self.assertEqual(value.shape, shape)
            self.assertEqual(value.dtype, self.dtype)

            if isinstance(self.value, (int, float)):
                value.to_cpu()
                np.testing.assert_equal(value.data, self.dtype(self.value))


class TestStubLinkInvalidArgument(unittest.TestCase):

    def test_no_shapes(self):
        with self.assertRaises(ValueError):
            StubLink()

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            StubLink((3, 4), value='invalid')


testing.run_module(__name__, __file__)
