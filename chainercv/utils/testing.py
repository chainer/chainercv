import numpy as np

import chainer


class StubLink(chainer.Link):
    """A chainer.Link that returns dummy value(s).

    This is a :obj:`chainer.Link` that returns dummy
    :obj:`chainer.Variable` when :meth:`__call__` method is called.

    Args:
        shape (int or tuple of int): The shape of returned variable.
            This argument can be specified more than once.
            In this case, :meth:`__call__` returns a tuple of
            :obj:`chainer.Variable`.
        value (:obj:`'uniform'`, int or float): The value of returned
            variable. If this is :obj:`'uniform'`, the values of the variable
            are drawn from an uniformed distribution. Otherwise, They are
            initalized with the specified value.
            The default value is :obj:`'uniform'`.
        dtype: The type of returned variable. The default value is
            :obj:`~numpy.float32`.
    """

    def __init__(self, *shape, **kwargs):
        super(StubLink, self).__init__()

        if len(shape) == 0:
            raise ValueError('At least, one shape must be specified')
        self.shapes = shape

        value = 'uniform'
        if 'value' in kwargs:
            value = kwargs['value']
            del kwargs['value']

        dtype = np.float32
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
            del kwargs['dtype']

        if len(kwargs) > 0:
            raise ValueError('Unknown arguments {}'.format(kwargs))

        if value == 'uniform':
            def _get_array(shape):
                return np.random.uniform(size=shape).astype(dtype)
        elif isinstance(value, (int, float)):
            def _get_array(shape):
                array = np.empty(shape, dtype=dtype)
                array[:] = value
                return array
        else:
            raise ValueError('value must be \'uniform\', int or float')

        self._get_array = _get_array

    def __call__(self, *_):
        """Returns dummy value(s).

        Args:
            This method can take any values as argument.
            All of them are ignored.

        Returns:
            chainer.Variable or tuple of chainer.Variable:
            If only one :obj:`shape` is given to :meth:`__init__`, this method
            returns a :obj: `chainer.Variable`. Otherwise, this returns a
            tuple of :obj:`chainer.Variable`.
        """

        if len(self.shapes) == 1:
            return chainer.Variable(
                self.xp.asarray(self._get_array(self.shapes[0])))
        else:
            return tuple(
                chainer.Variable(self.xp.asarray(self._get_array(shape)))
                for shape in self.shapes)
