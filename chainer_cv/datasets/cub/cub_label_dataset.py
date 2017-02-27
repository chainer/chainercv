import os.path as osp
from skimage.io import imread

from chainer_cv.datasets.cub.cub_utils import CUBDatasetBase


class CUBLabelDataset(CUBDatasetBase):

    """CUB200-2011 dataset that returns an image and a label.

    There are 200 labels of birds.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            'auto', this class will automatically download data for you
            under ``$CHAINER_DATASET_ROOT/yuyu2172/chainer-cv/cub``.
        crop_bbox (bool): If true, this class returns an image cropped
            by the bounding box of the bird inside it.
        bgr (bool): If true, method `get_example` will return an image in BGR
            format.

    """

    def __init__(self, data_dir='auto', crop_bbox=True, bgr=True):
        super(CUBLabelDataset, self).__init__(
            data_dir=data_dir, crop_bbox=crop_bbox)

        classes_file = osp.join(self.data_dir, 'classes.txt')
        image_class_labels_file = osp.join(
            self.data_dir, 'image_class_labels.txt')
        self.labels = [label.split()[1] for label in open(classes_file)]
        self._data_labels = [int(d_label.split()[1]) - 1 for
                             d_label in open(image_class_labels_file)]

        self.bgr = bgr

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        If `self.bgr` is True, the image is in BGR. If not, it is in RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and its label.

        """
        img, label = self.get_raw_data(i)
        if self.bgr:
            img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)
        return img, label

    def get_raw_data(self, i):
        """Returns the i-th example.

        This returns a color image and its label. The image is in HWC foramt.
        Also, the image is in RGB.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example (image, label)

        """
        img = imread(osp.join(self.data_dir, 'images', self.fns[i]))  # RGB

        if self.crop_bbox:
            bbox = self.bboxes[i]  # (x, y, width, height)
            img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        label = self._data_labels[i]
        return img, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    for i in range(1000, 1020):
        plt.figure()
        dataset = CUBLabelDataset(crop_bbox=False)
        img, label = dataset.get_raw_data(i)
        dataset = CUBLabelDataset(crop_bbox=True)
        cropped, label = dataset.get_raw_data(i)
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.subplot(2, 1, 2)
        plt.imshow(cropped)
        plt.show()
