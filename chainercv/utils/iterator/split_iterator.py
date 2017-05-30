class BufferedIterator(object):

    def __init__(self, iterator, buffers, index):
        self.iterator = iterator
        self.buffers = buffers
        self.index = index

    def __del__(self):
        self.buffers[self.index] = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.buffers[self.index].pop(0)
        except IndexError:
            values = next(self.iterator)
            for buf, val in zip(self.buffers, values):
                if buf is not None:
                    buf.append(val)
            return self.buffers[self.index].pop(0)

    next = __next__


def split_iterator(iterator):
    """Converts an iterator of tuples into a tuple of iterators.

    This function converts an iterator of tuples into a tuple of iterators.
    This is an inverse function of :func:`zip`.

    >>> from chainercv.utils.iterator import split_iterator
    >>> in_iter = iter([(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')])
    >>> int_iter, str_iter = split_iterator(in_iter)
    >>>
    >>> next(int_iter)  # 0
    >>> next(int_iter)  # 1
    >>> next(int_iter)  # 2
    >>>
    >>> next(str_iter)  # 'a'
    >>> next(str_iter)  # 'b'
    >>> next(str_iter)  # 'c'

    Args:
        iterator (iterator): An iterator of tuples. All tuples should have
            the same length.

    Returns:
        tuple of iterators:
        Each iterator corresponds to each element of input tuple.
    """

    values = next(iterator)
    buffers = [[val] for val in values]
    return tuple(
        BufferedIterator(iterator, buffers, index)
        for index in range(len(buffers)))
