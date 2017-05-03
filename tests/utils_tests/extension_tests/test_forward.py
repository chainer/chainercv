import mock
import numpy as np
import unittest

import chainer
from chainer import testing

from chainercv.utils import forward


@testing.parameterize(*testing.product({
    'in_shapes': [((3, 4),), ((3, 4), (5,))],
    'out_shapes': [((3, 4),), ((3, 4), (5,))],
    'variable': [True, False],
}))
class TestForward(unittest.TestCase):

    def setUp(self):
        self.xp = np

        self.mocked_model = mock.MagicMock()
        self.mocked_model.xp = self.xp

        self.inputs = tuple(np.empty(shape) for shape in self.in_shapes)
        if len(self.inputs) == 1:
            self.inputs = self.inputs[0]

        self.outputs = tuple(
            self.xp.array(np.empty(shape)) for shape in self.out_shapes)
        if self.variable:
            self.outputs = tuple(
                chainer.Variable(output) for output in self.outputs)
        if len(self.outputs) == 1:
            self.outputs = self.outputs[0]

    def _check_inputs(self, inputs):
        if isinstance(self.inputs, tuple):
            orig_inputs = self.inputs
        else:
            orig_inputs = self.inputs,

        for orig, in_ in zip(orig_inputs, inputs):
            self.assertIsInstance(in_, chainer.Variable)
            self.assertEqual(chainer.cuda.get_array_module(in_.data), self.xp)

            in_ = chainer.cuda.to_cpu(in_.data)
            np.testing.assert_equal(in_, orig)

    def _check_outputs(self, outputs):
        if len(outputs) == 1:
            outputs = outputs,

        for orig, out in zip(self.outputs, outputs):
            self.assertIsInstance(out, np.ndarray)

            if self.variable:
                orig = orig.data
            orig = chainer.cuda.to_cpu(orig)
            np.testing.assert_equal(out, orig)

    def test_forward(self):
        def _call(*inputs):
            self._check_inputs(inputs)
            return self.outputs
        self.mocked_model.side_effect = _call

        outputs = forward(self.mocked_model, self.inputs)
        self._check_outputs(outputs)


testing.run_module(__name__, __file__)
