from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

from chainercv.experimental.links import YOLOv2Tiny
from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.visualizations import vis_bbox


def main():
    dataset = VOCBboxDataset(year='2007', split='test') \
        .slice[[29, 301, 189, 229], 'img']
    models = [
        ('Faster R-CNN', FasterRCNNVGG16(pretrained_model='voc07')),
        ('SSD300', SSD300(pretrained_model='voc0712')),
        ('SSD512', SSD512(pretrained_model='voc0712')),
        ('YOLOv2', YOLOv2(pretrained_model='voc0712')),
        ('YOLOv2 tiny', YOLOv2Tiny(pretrained_model='voc0712')),
        ('YOLOv3', YOLOv3(pretrained_model='voc0712')),
    ]

    fig = plt.figure(figsize=(30, 20))
    for i, img in enumerate(dataset):
        for j, (name, model) in enumerate(models):
            bboxes, labels, scores = model.predict([img])
            bbox, label, score = bboxes[0], labels[0], scores[0]

            ax = fig.add_subplot(
                len(dataset), len(models), i * len(models) + j + 1)
            vis_bbox(
                img, bbox, label, score,
                label_names=voc_bbox_label_names, ax=ax
            )

            # Set MatplotLib parameters
            ax.set_aspect('equal')
            if i == 0:
                font = FontProperties()
                font.set_family('serif')
                font.set_size(35)
                ax.set_title(name, y=1.03, fontproperties=font)
            plt.axis('off')
            plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
