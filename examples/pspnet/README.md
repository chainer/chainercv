# Examples of Pyramid Scene Parsing Network (PSPNet)

## Demo
This demo downloads a pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] [--input-size <size>] <image>.jpg
```

## Weight Covnersion

Convert `*.caffemodel` to `*.npz`. Some layers are renamed to fit ChainerCV.
```
$ python caffe2npz.py <source>.caffemodel <target>.npz
```

The converted weight can be downloaded from [here](https://chainercv-models.preferred.jp/pspnet_resnet101_cityscapes_converted_2018_05_22.npz).

The performance on the Cityscapes dataset is as follows with single scale inference.
Scores are measured by mean Intersection over Union (mIoU).

| Dataset | Scale | Original | Ours (weight conversion) |
|:-:|:-:|:-:|:-:|
| Cityscapes | Single scale | 79.70 % [1] | 79.03 % |

## Training model

The model can be trained with a script `train_mutli.py`.

### Cityscapes

The following table shows the performance of the models trained with our scripts.

| Model | Training Data |  Original | Ours |
|:-:|:-:|:-:|:-:|
| PSPNet w/ Dilated ResNet50 | fine only (3K) | 76.9 % [2] |  73.99 % |
| PSPNet w/ Dilated ResNet101 | fine only (3K) |  77.9 % [2] | 76.01 % |

Here are the commands used to train the models included in the table.

```
$ mpiexec -n 8 python3 train_multi.py --dataset cityscapes --model pspnet_resnet50 --iteration 90000
$ mpiexec -n 8 python3 train_multi.py --dataset cityscapes --model pspnet_resnet101 --iteration 90000
```

### ADE20K

The following table shows the performance of the models trained with our scripts.

| Model |  Original | Ours |
|:-:|:-:|:-:|
| PSPNet w/ Dilated ResNet50 | 41.68 % [1] |  34.97 % |
| PSPNet w/ Dilated ResNet101 |  | 36.55 % |

Here are the commands used to train the models included in the table.

```
$ mpiexec -n 8 python3 train_multi.py --dataset ade20k --model pspnet_resnet50 --iteration 150000
$ mpiexec -n 8 python3 train_multi.py --dataset ade20k --model pspnet_resnet101 --iteration 150000
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/semantic_segmentation/eval_semantic_segmentation.py`](https://github.com/chainer/chainercv/blob/master/examples/semantic_segmentation).


## References
1. Hengshuang Zhao et al. "Pyramid Scene Parsing Network" CVPR 2017.
2. https://github.com/holyseven/PSPNet-TF-Reproduce (Validation scores for Cityscapes are lacking in the original paper)
3. [chainer-pspnet by mitmul](https://github.com/mitmul/chainer-pspnet)
