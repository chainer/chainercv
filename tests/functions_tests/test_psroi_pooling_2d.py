import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import numpy as np
import unittest

from chainercv import functions


class TestPSROIPolling2D(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.group_size = 2
        self.out_c = 2
        self.n_channels = self.group_size * self.group_size * self.out_c
        self.x = np.arange(
            self.N * self.n_channels * 10 * 12,
            dtype=np.float32).reshape((self.N, self.n_channels, 10, 12))
        np.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        self.x = self.x.astype(np.float32)
        self.rois = np.array(
            [[0, 0, 7, 7],
             [1, 0, 5, 12],
             [0, 1, 10, 5],
             [3, 3, 4, 4]],
            dtype=np.float32
        )
        self.roi_indices = np.array([0, 2, 1, 0], dtype=np.int32)
        self.n_roi = self.rois.shape[0]
        self.out_h, self.out_w = 4, 4
        self.spatial_scale = 1.0
        self.gy = np.random.uniform(
            -1, 1, (self.n_roi, self.out_c, self.out_h, self.out_w))
        self.gy = self.gy.astype(np.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, roi_data, roi_index_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        roi_indices = chainer.Variable(roi_index_data)
        y = functions.psroi_pooling_2d(
            x, rois, roi_indices, self.out_c, self.out_h, self.out_w,
            self.spatial_scale, self.group_size)
        self.assertEqual(y.data.dtype, np.float32)
        y_data = cuda.to_cpu(y.data)
        self.assertEqual(
            (self.n_roi, self.out_c, self.out_h, self.out_w), y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.rois, self.roi_indices)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices))

    def check_backward(self, x_data, roi_data, roi_index_data, y_grad_data):
        gradient_check.check_backward(
            functions.PSROIPooling2D(
                self.out_c, self.out_h, self.out_w,
                self.spatial_scale, self.group_size),
            (x_data, roi_data, roi_index_data), y_grad_data,
            no_grads=[False, True, True], **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.rois, self.roi_indices, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices), cuda.to_gpu(self.gy))

    def apply_backward(self, x_data, roi_data, roi_index_data, y_grad_data):
        x = chainer.Variable(x_data)
        rois = chainer.Variable(roi_data)
        roi_indices = chainer.Variable(roi_index_data)
        y = functions.psroi_pooling_2d(
            x, rois, roi_indices, self.out_c, self.out_h, self.out_w,
            self.spatial_scale, self.group_size)
        x.cleargrad()
        y.grad = y_grad_data
        y.backward()
        return x, y

    @attr.gpu
    @condition.retry(3)
    def test_consistency_with_gpu(self):
        x_cpu, y_cpu = self.apply_backward(
            self.x, self.rois, self.roi_indices, self.gy)
        x_gpu, y_gpu = self.apply_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.rois),
            cuda.to_gpu(self.roi_indices), cuda.to_gpu(self.gy))
        testing.assert_allclose(y_cpu.data, y_gpu.data)
        testing.assert_allclose(x_cpu.grad, x_gpu.grad)


testing.run_module(__name__, __file__)
