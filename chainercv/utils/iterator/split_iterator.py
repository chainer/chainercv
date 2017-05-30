class IteratorHub(object):

    def __init__(self, iterator):
        self.iterator = iterator

        sample = next(self.iterator)
        self.children = tuple(BufferedIterator(self) for _ in sample)
        for child, value in zip(self.children, sample):
            child.append(value)

    def feed(self):
        sample = next(self.iterator)
        for child, value in zip(self.children, sample):
            child.append(value)


class BufferedIterator(object):

    def __init__(self, hub):
        self.hub = hub
        self.buffer = list()

    def append(self, value):
        self.buffer.append(value)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.buffer.pop(0)
        except:
            self.hub.feed()
            return self.buffer.pop(0)


def split_iterator(iterator):
    """Converts an iterator of tuples into a tuple of iterators.

    This function converts an iterator of tuples into a tuple of iterators.
    This is an inverse function of :func:`zip`.

    >>> input = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]
    >>> int_iter, str_iter = split_iterator(input)
    >>>
    >>> next(int_iter)  # 0
    >>> next(int_iter)  # 1
    >>> next(int_iter)  # 2
    >>>
    >>> next(str_iter)  # 'a'
    >>> next(str_iter)  # 'b'
    >>> next(str_iter)  # 'c'

    Args:
        iterator (iterator): An iterator of tuples. All tuples should have same
            length.

    Returns:
        tuple of iterators:
        Each iterator corresponds to each element of input tuple.
    """

    hub = IteratorHub(iterator)
    return hub.children
