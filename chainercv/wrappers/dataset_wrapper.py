import six

import numpy as np

import chainer
from chainer.utils import type_check


class DatasetWrapper(chainer.dataset.DatasetMixin):
    """Wrap dataset class to add functionalities.

    This class is wrapped around a dataset or another wrapper to add a
    functionality.

    The method `_get_example` should contain a code that are necessary to add
    a functionality.

    If an output of the wrapped dataset is not a tuple, the wrapper class
    forces it to be a tuple when passed to `_get_example`. In that case, the
    final value that is returned by `get_example` is forced back to a
    non-tuple.

    Args:
        dataset: a dataset or a wrapper that this wraps.

    """

    def __init__(self, dataset):
        self._dataset = dataset
        self._update_wrapper_stack()

    def _update_wrapper_stack(self):
        """Keep a list of all the wrappers that have been appended to the stack.

        """
        self._wrapper_stack = getattr(self._dataset, '_wrapper_stack', [])
        self._wrapper_stack.append(self)

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, name):
        if name == 'get_example':
            return self.get_example
        elif name == '__getitem__':
            return self.__getitem__
        orig_attr = getattr(self._dataset, name)
        return orig_attr

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        # this can be overridden
        in_data = self._dataset[i]

        # check if input is tuple
        converted_tuple = False
        if not isinstance(in_data, tuple):
            in_data = (in_data,)
            converted_tuple = True

        # check type
        self._check_data_type_get_example(in_data)

        # convert back to non tuple if necessary
        out = self._get_example(in_data)
        if converted_tuple:
            out = out[0]
        return out

    def _get_example(self, in_data):
        """Returns the i-th example given values from the wrapped dataset.

        Args:
            in_data (tuple): The i-th example from the wrapped dataset.

        Returns:
            The i-th example.

        """
        return in_data

    def _check_data_type_get_example(self, in_data):
        """Internal function called before checking types.

        Args:
            in_data (tuple)
        """
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
        """Checks types of input data before calling `_get_example`.

        Before :meth:`_get_example` is called, this function is called.
        You need to validate types of input data in this function
        using :ref:`the type checking utilities <type-check-utils>`.

        Args:
            in_types (~chainer.utils.type_check.TypeInfoTuple): The type
                information of input data for :meth:`_get_example`.
        """
        pass

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self._dataset)

    def __repr__(self):
        return str(self)
