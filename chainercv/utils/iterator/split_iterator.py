class BufferedIterator(object):

    def __init__(self, feed):
        self.feed = feed
        self.buffer = list()

    def push(self, value):
        self.buffer.append(value)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.buffer.pop(0)
        except IndexError:
            self.feed()
            return self.buffer.pop(0)

    next = __next__


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

    def feed(values=None):
        if values is None:
            values = next(iterator)
        for it, value in zip(iters, values):
            it.push(value)

    values = next(iterator)
    iters = tuple(BufferedIterator(feed) for _ in values)
    feed(values)

    return iters
