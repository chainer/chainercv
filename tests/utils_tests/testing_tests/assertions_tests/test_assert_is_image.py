import numpy as np
import unittest

from chainer import testing

from chainercv.utils import assert_is_image


@testing.parameterize(
    {
        'img': np.random.randint(0, 256, size=(3, 48, 64)),
        'color': True, 'check_range': True, 'valid': True},
    {
        'img': np.random.randint(0, 256, size=(1, 48, 64)),
        'color': True, 'check_range': True, 'valid': False},
    {
        'img': np.random.randint(0, 256, size=(4, 48, 64)),
        'color': True, 'check_range': True, 'valid': False},
    {
        'img': np.ones((3, 48, 64)) * 256,
        'color': True, 'check_range': True, 'valid': False},
    {
        'img': np.ones((3, 48, 64)) * -1,
        'color': True, 'check_range': True, 'valid': False},
    {
        'img': np.ones((3, 48, 64)) * 256,
        'color': True, 'check_range': False, 'valid': True},

    {
        'img': np.random.randint(0, 256, size=(1, 48, 64)),
        'color': False, 'check_range': True, 'valid': True},
    {
        'img': np.random.randint(0, 256, size=(3, 48, 64)),
        'color': False, 'check_range': True, 'valid': False},
    {
        'img': np.ones((1, 48, 64)) * 256,
        'color': False, 'check_range': True, 'valid': False},
    {
        'img': np.ones((1, 48, 64)) * -1,
        'color': False, 'check_range': True, 'valid': False},
    {
        'img': np.ones((1, 48, 64)) * 256,
        'color': False, 'check_range': False, 'valid': True},

    {
        'img': (((0, 1), (2, 3)), ((4, 5), (6, 7)), ((8, 9), (10, 11))),
        'color': True, 'check_range': True, 'valid': False},
)
class TestAssertIsImage(unittest.TestCase):

    def test_assert_is_image(self):
        if self.valid:
            assert_is_image(self.img, self.color, self.check_range)
        else:
            with self.assertRaises(AssertionError):
                assert_is_image(self.img, self.color, self.check_range)


testing.run_module(__name__, __file__)
