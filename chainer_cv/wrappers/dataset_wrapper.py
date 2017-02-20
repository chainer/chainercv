import six

import numpy as np

import chainer
from chainer.utils import type_check


class DatasetWrapper(chainer.dataset.DatasetMixin):

    def __init__(self, dataset):
        self.dataset = dataset
        self._update_wrapper_stack()

    def _update_wrapper_stack(self):
        """
        Keep a list of all the wrappers that have been appended to the stack.
        """
        self._wrapper_stack = getattr(self.dataset, '_wrapper_stack', [])
        self._wrapper_stack.append(self)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, attr):
        if attr == 'get_example':
            return self.get_example
        elif attr == '__getitem__':
            return self.__getitem__
        orig_attr = getattr(self.dataset, attr)
        return orig_attr

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        # this can be overridden
        in_data = self.dataset[i]
        self._check_data_type_get_example(in_data)
        return self._get_example(in_data)

    def _get_example(self, in_data):
        """Returns the i-th example given values from the wrapped dataset.

        Args:
            in_data: The i-th example of the wrapped dataset.

        Returns:
            The i-th example.

        """
        raise NotImplementedError

    def _check_data_type_get_example(self, in_data):
        in_data = tuple([np.array(v) for v in in_data])
        in_type = type_check.get_types(in_data, 'in_types', False)
        try:
            self.check_type_get_example(in_type)
        except type_check.InvalidType as e:
            msg = """
Invalid operation is performed in: {0} (get_example)

{1}""".format(self.label, str(e))
            six.raise_from(
                type_check.InvalidType(e.expect, e.actual, msg=msg), None)

    def check_type_get_example(self, in_types):
        """Checks types of input data before forward propagation.

        Before :meth:`get_example` is called, this function is called.
        You need to validate types of input data in this function
        using :ref:`the type checking utilities <type-check-utils>`.

        Args:
            in_types (~chainer.utils.type_check.TypeInfoTuple): The type
                information of input data for :meth:`get_example`.
        """
        pass

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.dataset)

    def __repr__(self):
        return str(self)
