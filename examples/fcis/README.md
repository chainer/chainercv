# Example codes for FCIS [1]

## Performance

### SBD Train & Test

| Model | mAP@0.5 (Original [1]) | mAP@0.7 (Original [1]) | mAP@0.5 (weight conversion) | mAP@0.7 (weight conversion) |  mAP@0.5 (train) | mAP@0.7 (train) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FCIS ResNet101| 65.7 | 52.1 | 64.2 | 51.2 | 64.1 (1 GPU) | 51.2 (1 GPU) |

### COCO Train & Test

| Model | mAP/iou@[0.5:0.95] (original [1])| mAP/iou@0.5 (original [1])| mAP/iou@[0.5:0.95] (weight conversion)| mAP/iou@0.5 (weight conversion)| mAP/iou@[0.5:0.95] (train)| mAP/iou@0.5 (train)|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FCIS ResNet101 | 29.2 | 49.5 | 27.9 | 46.3 | 24.3 (3 GPU) | 42.6 (3 GPU) |

\*: We use random sampling for sampling strategy. The original paper used OHEM sampling strategy.

## Demo
Segment objects in an given image. This demo downloads SBD pretrained model automatically if a pretrained model path is not given.

```bash
python demo.py [--dataset sbd|coco] [--gpu <gpu>] [--pretrained-model <model_path>] <image.jpg>
```

## Evaluation
The evaluation for sbd dataset can be conducted using [`chainercv/examples/instance_segmentation/eval_sbd.py`](https://github.com/chainer/chainercv/blob/master/examples/instance_segmentation)
and the one for coco dataset can be conducted using [`chainercv/examples/instance_segmentation/eval_coco.py`](https://github.com/chainer/chainercv/blob/master/examples/instance_segmentation).

## Train
You can train the model with the following code.
Note that this code requires `SciPy` module.

### SBD Train with single GPU

```bash
python train_sbd.py [--gpu <gpu>]
```

### SBD Train with multiple GPUs

If you want to use multiple GPUs, use `train_sbd_multi.py`.
Note that this code requires `chainermn` module.

```bash
mpiexec -n <n_gpu> python train_sbd_multi.py --lr  <n_gpu>*0.0005
```

You can download weights that were trained by ChainerCV.
- [FCIS ResNet101 SBD Trained](https://chainercv-models.preferred.jp/fcis_resnet101_sbd_trained_2018_06_22.npz)

### COCO Train with multiple GPUs

If you want to use multiple GPUs, use `train_coco_multi.py`.
Note that this code requires `chainermn` module.

```bash
mpiexec -n <n_gpu> python train_coco_multi.py --lr  <n_gpu>*0.0005
```

You can download weights that were trained by ChainerCV.
- [FCIS ResNet101 COCO Trained](https://chainercv-models.preferred.jp/fcis_resnet101_coco_trained_2019_01_30.npz)


## Convert Mxnet model
Convert `*.params` to `*.npz`.
Note that the number of classes and network structure is specified by `--dataset`.

```bash
python mxnet2npz.py [--dataset sbd|coco] [--out <npz filename>] <param filename>
```
You can download weights that were converted the script.
- [FCIS ResNet101 SBD Converted](https://chainercv-models.preferred.jp/fcis_resnet101_sbd_converted_2018_07_02.npz)
- [FCIS ResNet101 COCO Converted](https://chainercv-models.preferred.jp/fcis_resnet101_coco_converted_2019_01_30.npz)

#
## References
1. Yi Li et al. "Fully Convolutional Instance-aware Semantic Segmentation" CVPR 2017.
