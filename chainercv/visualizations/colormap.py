def voc_colormap(label):
    """Color map used in PASCAL VOC

    Args:
        label (int): Class id.

    """
    r, g, b = 0, 0, 0
    i = label
    for j in range(8):
        if i & (1 << 0):
            r |= 1 << (7 - j)
        if i & (1 << 1):
            g |= 1 << (7 - j)
        if i & (1 << 2):
            b |= 1 << (7 - j)
        i >>= 3
    return r, g, b
