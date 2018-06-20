# Example codes for FCIS [1]

## Performance
SBD Train & Test

| Model | mAP@0.5 in Original [1] | mAP@0.7 in Original [1] | mAP@0.5 in ChainerCV | mAP@0.7 in ChainerCV |
|:-:|:-:|:-:|:-:|:-:|
| FCIS ResNet101| 65.7 | 52.1 | 64.2 (1 GPU) | 51.4 (1 GPU) |

## Demo
Segment objects in an given image. This demo downloads SBD pretrained model automatically if a pretrained model path is not given.

```bash
python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] <image.jpg>
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/instance_segmentation/eval_sbd.py`](https://github.com/chainer/chainercv/blob/master/examples/instance_segmentation)

## Train
You can train the model with the following code.
Note that this code requires `SciPy` module.

```bash
python train.py [--gpu <gpu>]
```

If you want to use multiple GPUs, use `train_multi.py`.
Note that this code requires `chainermn` module.

```bash
mpi4exec -n <n_gpu> python train_multi.py

```
You can download weights that were trained by ChainerCV.
- [FCIS ResNet101](https://chainercv-models.preferred.jp/fcis_resnet101_sbd_trained_2018_04_14.npz)

## References
1. Yi Li et al. "Fully Convolutional Instance-aware Semantic Segmentation" CVPR 2017.
