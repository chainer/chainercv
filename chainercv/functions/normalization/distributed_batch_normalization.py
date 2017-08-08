# This code is first written by Takuya Akiba 2017

import cffi
import chainer
import numpy
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import type_check

import mpi4py.MPI

ffi = cffi.FFI()


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu


def array_to_buffer_object(array):
    if cuda.get_array_module(array) is numpy:
        return array
    else:
        return (ffi.buffer(ffi.cast('void *', array.data.ptr), array.nbytes),
                mpi4py.MPI.FLOAT)


class DistributedBatchNormalizationFunction(function.Function):

    def __init__(self, comm, eps=2e-5, mean=None, var=None, decay=0.9):
        self.comm = comm
        self.running_mean = mean
        self.running_var = var

        # Note: cuDNN v5 requires that eps be greater than 1e-5. Otherwise, an
        # error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if chainer.should_use_cudnn('>=auto'):
            if eps < 1e-5:
                msg = 'cuDNN does not allow an eps value less than 1e-5.'
                raise RuntimeError(msg)
        self.mean_cache = None
        self.decay = decay

    def check_type_forward(self, in_types):
        n_in = type_check.eval(in_types.size())
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))
        x_type, gamma_type, beta_type = in_types[:3]
        M = type_check.eval(gamma_type.ndim)
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )
        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == x_type.dtype,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == x_type.dtype,
                var_type.shape == gamma_type.shape,
            )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]
        if configuration.config.train:
            if self.running_mean is None:
                self.running_mean = xp.zeros_like(gamma)
                self.running_var = xp.zeros_like(gamma)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        elif len(inputs) == 5:
            self.fixed_mean = inputs[3]
            self.fixed_var = inputs[4]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        # cuDNN only supports these tensor dimensions because they are
        # the most commonly used. If there is a need to support other
        # dimensions with cuDNN, we could consider reshaping the input
        # into a 2-dim array with channels as second dim and m=<product
        # of all dimensions except the 2nd dimension> as the first
        # dimension.
        cudnn_dim_ok = x.ndim == 2 or (x.ndim == 4 and head_ndim == 2)
        # TODO(bkvogel): Check for float16 support again in next cuDNN version.
        # cuDNN v5 batch normalization does not seem to support float16.
        self._can_use_cudnn = cudnn_dim_ok and x[0].dtype != numpy.float16

        cudnn_updated_running_stats = False

        if configuration.config.train:
            axis = (0,) + tuple(range(head_ndim, x.ndim))
            """
            # Normal
            mean = x.mean(axis=axis)
            var = x.var(axis=axis)
            var += self.eps
            """

            """
            # Naive implementation
            mpi_comm = self.comm.mpi_comm
            mean = x.mean(axis=axis)
            mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, array_to_buffer_object(mean))
            mean *= 1.0 / mpi_comm.size

            print(x.mean(axis=axis), mean)

            var = xp.square(x - mean[expander]).mean(axis=axis)
            var = xp.ascontiguousarray(var)
            mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, array_to_buffer_object(var))
            var *= 1.0 / mpi_comm.size
            print(x.var(axis=axis), var)

            var += self.eps
            """

            mpi_comm = self.comm.mpi_comm
            tmp = xp.empty(gamma.size * 2, dtype=x.dtype)
            x.mean(axis=axis, out=tmp[:gamma.size])
            xp.square(x).mean(axis=axis, out=tmp[gamma.size:])
            mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, array_to_buffer_object(tmp))
            tmp *= 1.0 / mpi_comm.size

            mean = tmp[:gamma.size]
            sqmean = tmp[gamma.size:]
            var = sqmean - xp.square(mean)

            var += self.eps
        else:
            mean = self.fixed_mean
            var = self.fixed_var + self.eps
        self.std = xp.sqrt(var, dtype=var.dtype)
        if xp is numpy:
            self.x_hat = _xhat(x, mean, self.std, expander)
            y = gamma * self.x_hat
            y += beta
        else:
            self.x_hat, y = cuda.elementwise(
                'T x, T mean, T std, T gamma, T beta', 'T x_hat, T y',
                '''
                x_hat = (x - mean) / std;
                y = gamma * x_hat + beta;
                ''',
                'bn_fwd')(x, mean[expander], self.std[expander], gamma,
                          beta)

        if configuration.config.train and (not cudnn_updated_running_stats):
            # Note: If in training mode, the cuDNN forward training function
            # will do this for us, so
            # only run following code if cuDNN was not used.
            # Update running statistics:
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_mean *= self.decay
            temp_ar = xp.array(mean)
            temp_ar *= (1 - self.decay)
            self.running_mean += temp_ar
            del temp_ar
            self.running_var *= self.decay
            temp_ar = xp.array(var)
            temp_ar *= (1 - self.decay) * adjust
            self.running_var += temp_ar
            del temp_ar
        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)
        if len(inputs) == 5:
            # This case is unlikely to be used in practice and so does not
            # need to be optimized for performance.
            mean = inputs[3]
            var = inputs[4]
            std = xp.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            gbeta = gy.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            ggamma = (gy * x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, gmean, gvar

        # Note: If length of inputs is not 5, we must be in train mode.
        assert configuration.config.train

        # gbeta = gy.sum(axis=axis)
        # ggamma = (gy * self.x_hat).sum(axis=axis)

        # It is wrong to multiply m by mpi_comm.size (instead of multiplying 1/size to gbeta, ggamma)
        mpi_comm = self.comm.mpi_comm
        tmp = xp.empty(gamma.size * 2, dtype=x.dtype)
        gy.sum(axis=axis, out=tmp[:gamma.size])
        (gy * self.x_hat).sum(axis=axis, out=tmp[gamma.size:])
        mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, array_to_buffer_object(tmp))
        tmp *= 1.0 / mpi_comm.size
        gbeta = tmp[:gamma.size]
        ggamma = tmp[gamma.size:]

        if xp is numpy:
            gx = (gamma / self.std)[expander] * (
                gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
        else:
            inv_m = numpy.float32(1) / m
            gx = cuda.elementwise(
                'T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, \
                T inv_m',
                'T gx',
                'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * \
                inv_m)',
                'bn_bwd')(gy, self.x_hat, gamma[expander],
                          self.std[expander], ggamma[expander],
                          gbeta[expander], inv_m)

        return gx, ggamma, gbeta
