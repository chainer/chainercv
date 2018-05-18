from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.visualizations import vis_bbox


def main():
    dataset = VOCBboxDataset(year='2007', split='test')
    models = [
        ('Faster R-CNN', FasterRCNNVGG16(pretrained_model='voc07')),
        ('SSD300', SSD300(pretrained_model='voc0712')),
        ('SSD512', SSD512(pretrained_model='voc0712')),
    ]
    indices = [29, 301, 189, 229]

    fig = plt.figure(figsize=(30, 30))
    for i, idx in enumerate(indices):
        for j, (name, model) in enumerate(models):
            img, _, _ = dataset[idx]
            bboxes, labels, scores = model.predict([img])
            bbox, label, score = bboxes[0], labels[0], scores[0]

            ax = fig.add_subplot(
                len(indices), len(models), i * len(models) + j + 1)
            vis_bbox(
                img, bbox, label, score,
                label_names=voc_bbox_label_names, ax=ax
            )

            # Set MatplotLib parameters
            ax.set_aspect('equal')
            if i == 0:
                font = FontProperties()
                font.set_family('serif')
                ax.set_title(name, fontsize=35, y=1.03, fontproperties=font)
            plt.axis('off')
            plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
