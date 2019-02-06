# Examples of Semantic Segmentation


## Supported models
- SegNet
- PSPNet

For the details, please check the documents and examples of each model.

## Performance

The scores are mIoU.

### Cityscapes

| Model | Reference | ChainerCV (weight conversion) |
|:-:|:-:|:-:|
| PSPNet with ResNet101 (single scale) | 79.70 % [1] | 79.03 % |

```
$ python eval_semantic_segmentation.py --gpu <GPU> --dataset cityscapes --model pspnet_resnet101
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_semantic_segmentation_multi.py --dataset cityscapes --model pspnet_resnet101
```


### CamVid

| Model | Reference | ChainerCV |
|:-:|:-:|:-:|
| SegNet | 46.3 % [2] | 49.4 % |

```
$ python eval_semantic_segmentation.py --gpu <GPU> --dataset camvid --model segnet
```


# Reference

1. Hengshuang Zhao et al. "Pyramid Scene Parsing Network" CVPR 2017.
2. Vijay Badrinarayanan et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.
