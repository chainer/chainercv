import numpy as np
import unittest

from chainer import testing

from chainercv.links.model.ssd import generate_default_bbox


@testing.parameterize(
    {
        'grids': (32,),
        'aspect_ratios': ((2,),),
        'steps': (1,),
        'sizes': (1, 2),
    },
    {
        'grids': (32, 16, 8, 4, 2),
        'aspect_ratios': ((2,), (3, 4), (5,), (6, 7, 8), (9,)),
        'steps': (1, 2, 4, 8, 16),
        'sizes': (1, 2, 3, 4, 5, 6),
    },
)
class TestGenerateDefaultBbox(unittest.TestCase):

    def test_generate_default_bbox(self):
        default_bbox = generate_default_bbox(
            self.grids, self.aspect_ratios, self.steps, self.sizes)
        self.assertIsInstance(default_bbox, np.ndarray)

        expect_n_bbox = sum(
            grid * grid * (len(ar) + 1) * 2
            for grid, ar in zip(self.grids, self.aspect_ratios))
        self.assertEqual(default_bbox.shape, (expect_n_bbox, 4))


class TestGenerateDefaultBboxMismatchArgument(unittest.TestCase):

    def setUp(self):
        self.grids = (32, 16, 8, 4, 2),
        self.aspect_ratios = ((2,), (3, 4), (5,), (6, 7, 8), (9,))
        self.steps = (1, 2, 4, 8, 16)
        self.sizes = (1, 2, 3, 4, 5, 6)

    def test_mismatch_aspect_ratios(self):
        with self.assertRaises(ValueError):
            generate_default_bbox(
                self.grids, self.aspect_ratios[:-1], self.steps, self.sizes)

        with self.assertRaises(ValueError):
            generate_default_bbox(
                self.grids, self.aspect_ratios + (10,), self.steps, self.sizes)

    def test_mismatch_steps(self):
        with self.assertRaises(ValueError):
            generate_default_bbox(
                self.grids, self.aspect_ratios, self.steps[:-1], self.sizes)

        with self.assertRaises(ValueError):
            generate_default_bbox(
                self.grids, self.aspect_ratios, self.steps + (32,), self.sizes)

    def test_mismatch_sizes(self):
        with self.assertRaises(ValueError):
            generate_default_bbox(
                self.grids, self.aspect_ratios, self.steps, self.sizes[:-1])

        with self.assertRaises(ValueError):
            generate_default_bbox(
                self.grids, self.aspect_ratios, self.steps, self.sizes + (7,))


testing.run_module(__name__, __file__)
