import chainer


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
        if attr != 'get_example':
            orig_attr = self.dataset.__getattribute__(attr)
            return orig_attr
        return self.get_example

    def get_example(self, i):
        return self.dataset.get_example(i)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.dataset)

    def __repr__(self):
        return str(self)
