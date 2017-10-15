import chainer


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class PickedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, base, pick):
        self._base = base
        self._pick = pick

    def __len__(self):
        return len(self._base)

    def get_example(self, i):
        return self._base._pick_example(i, self._pick)


class PickableDataset(chainer.dataset.DatasetMixin):

    def __init__(self):
        self._getters = dict()

    def __len__(self):
        raise NotImplementedError

    def add_getter(self, names, getter):
        names = _as_tuple(names)
        for i, name in enumerate(names):
            self._getters[name] = (getter, i)

    def _pick_example(self, i, pick):
        pick = _as_tuple(pick)

        example = list()
        cache = dict()

        for name in pick:
            getter, index = self._getters[name]
            if getter not in cache:
                cache[getter] = _as_tuple(getter(i))
            example.append(cache[getter][index])

        if len(pick) > 1:
            return tuple(example)
        else:
            return example[0]

    def get_example(self, i):
        return self._pick_example(i, self.data_names)

    def pick(self, pick):
        return PickedDataset(self, pick)
