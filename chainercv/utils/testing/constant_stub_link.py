import numpy as np

import chainer


class ConstantStubLink(chainer.Link):
    """A chainer.Link that returns constant value(s).

    This is a :obj:`chainer.Link` that returns constant
    :obj:`chainer.Variable` (s) when :meth:`forward` method is called.

    Args:
        outputs (~numpy.ndarray or tuple or ~numpy.ndarray):
            The value(s) of variable(s) returned by :meth:`forward`.
            If an array is specified, :meth:`forward` returns
            a :obj:`chainer.Variable`. Otherwise, it returns a tuple of
            :obj:`chainer.Variable`.
    """

    def __init__(self, outputs):
        super(ConstantStubLink, self).__init__()

        if isinstance(outputs, tuple):
            self._tuple = True
        else:
            self._tuple = False
            outputs = outputs,

        self._outputs = []
        for output in outputs:
            if not isinstance(output, np.ndarray):
                raise ValueError(
                    'output must be numpy.ndarray or tuple of numpy.ndarray')
            self._outputs.append(chainer.Variable(output))
        self._outputs = tuple(self._outputs)

    def to_cpu(self):
        super(ConstantStubLink, self).to_cpu()
        for output in self._outputs:
            output.to_cpu()

    def to_gpu(self, device=None):
        super(ConstantStubLink, self).to_gpu(device)
        for output in self._outputs:
            output.to_gpu(device)

    def forward(self, *_):
        """Returns value(s).

        Args:
            This method can take any values as its arguments.
            This function returns values independent of the arguments.

        Returns:
            chainer.Variable or tuple of chainer.Variable:
            If :obj:`outputs` is an array, this method returns
            a :obj:`chainer.Variable`. Otherwise, this returns a
            tuple of :obj:`chainer.Variable`.
        """

        if self._tuple:
            return self._outputs
        else:
            return self._outputs[0]
