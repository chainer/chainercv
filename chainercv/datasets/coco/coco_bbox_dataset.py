from chainercv.datasets.coco.coco_instances_base_dataset import \
    COCOInstancesBaseDataset


class COCOBboxDataset(COCOInstancesBaseDataset):

    """Bounding box dataset for `MS COCO`_.

    .. _`MS COCO`: http://cocodataset.org/#home

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/coco`.
        split ({'train', 'val', 'minival', 'valminusminival'}): Select
            a split of the dataset.
        year ({'2014', '2017'}): Use a dataset released in :obj:`year`.
            Splits :obj:`minival` and :obj:`valminusminival` are only
            supported in year :obj:`2014`.
        use_crowded (bool): If true, use bounding boxes that are labeled as
            crowded in the original annotation. The default value is
            :obj:`False`.
        return_area (bool): If true, this dataset returns areas of masks
            around objects. The default value is :obj:`False`.
        return_crowded (bool): If true, this dataset returns a boolean array
            that indicates whether bounding boxes are labeled as crowded
            or not. The default value is :obj:`False`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox` [#coco_bbox_1]_, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label` [#coco_bbox_1]_, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`area` [#coco_bbox_1]_ [#coco_bbox_2]_, ":math:`(R,)`", \
        :obj:`float32`, --
        :obj:`crowded` [#coco_bbox_3]_, ":math:`(R,)`", :obj:`bool`, --

    .. [#coco_bbox_1] If :obj:`use_crowded = True`, :obj:`bbox`, \
        :obj:`label` and :obj:`area` contain crowded instances.
    .. [#coco_bbox_2] :obj:`area` is available \
        if :obj:`return_area = True`.
    .. [#coco_bbox_3] :obj:`crowded` is available \
        if :obj:`return_crowded = True`.

    When there are more than ten objects from the same category,
    bounding boxes correspond to crowd of instances instead of individual
    instances. Please see more detail in the Fig. 12 (e) of the summary
    paper [#]_.

    .. [#] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, \
        Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, \
        C. Lawrence Zitnick, Piotr Dollar.
        `Microsoft COCO: Common Objects in Context \
        <https://arxiv.org/abs/1405.0312>`_. arXiv 2014.

    """

    def __init__(self, data_dir='auto', split='train', year='2017',
                 use_crowded=False, return_area=False, return_crowded=False):
        super(COCOBboxDataset, self).__init__(
            data_dir, split, year, use_crowded)
        keys = ('img', 'bbox', 'label')
        if return_area:
            keys += ('area',)
        if return_crowded:
            keys += ('crowded',)
        self.keys = keys
