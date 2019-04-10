# Examples of Semantic Segmentation


## Supported models
- SegNet
- PSPNet

For the details, please check the documents and examples of each model.

## Performance

The scores are mIoU.

### Cityscapes

| Model | Training Data | Original | Ours  |
|:-:|:-:|:-:|:-:|
| PSPNet w/ Dilated ResNet50 | fine only (3K) | 76.9 % [2] |  73.99 % |
| PSPNet w/ Dilated ResNet101 | fine only (3K) |  77.9 % [2] | 76.01 % |


Example

```
$ python eval_semantic_segmentation.py --gpu <GPU> --dataset cityscapes --model pspnet_resnet101
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_semantic_segmentation_multi.py --dataset cityscapes --model pspnet_resnet101
```

### ADE20k

| Base model |  Original | Ours |
|:-:|:-:|:-:|
| Dilated ResNet50 | 41.68 % [1] |  34.97 % |
| Dilated ResNet101 |  | 36.55 % |

```
$ python eval_semantic_segmentation.py --gpu <GPU> --dataset ade20k --model pspnet_resnet101
```


### CamVid

| Model | Original | Ours |
|:-:|:-:|:-:|
| SegNet | 46.3 % [3] | 49.4 % |

```
$ python eval_semantic_segmentation.py --gpu <GPU> --dataset camvid --model segnet
```


# Reference

1. Hengshuang Zhao et al. "Pyramid Scene Parsing Network" CVPR 2017.
2. https://github.com/holyseven/PSPNet-TF-Reproduce (Validation scores for Cityscapes are lacking in the original paper)
3. Vijay Badrinarayanan et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.
