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
$ python eval_cityscapes.py [--model pspnet_resnet101] [--gpu <gpu>] [--pretrained-model <model_path>]
```

You can conduct evaluation with multiple GPUs by `eval_cityscapes_multi.py`.
Note that this script requires ChainerMN.

```
$ mpiexec -n <#gpu> python eval_cityscapes_multi.py [--model pspnet_resnet101] [--pretrained-model <model_path>]
```

### CamVid

| Model | Reference | ChainerCV |
|:-:|:-:|:-:|
| SegNet | 46.3 % [2] | 49.4 % |

```
$ python eval_camvid.py [--gpu <gpu>] [--pretrained-model <model_path>] [--batchsize <batchsize>]
```


# Reference

1. Hengshuang Zhao et al. "Pyramid Scene Parsing Network" CVPR 2017.
2. Vijay Badrinarayanan et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.
