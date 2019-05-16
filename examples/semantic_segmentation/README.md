# Examples of Semantic Segmentation


## Supported models
- SegNet
- PSPNet

For the details, please check the documents and examples of each model.

## Performance

The scores are mIoU.

### Cityscapes

| Model | Training Data | Reference | ChainerCV  |
|:-:|:-:|:-:|:-:|
| PSPNet w/ Dilated ResNet50 | fine only (3K) | 76.9 % [2] |  73.99 % |
| PSPNet w/ Dilated ResNet101 | fine only (3K) |  77.9 % [2] | 76.01 % |
| DeepLab V3+ w/ Xception65 | fine only (3K) |  79.12 % [4] | 79.14 % |

You can reproduce these scores by the following command.

```
$ python eval_semantic_segmentation.py --dataset cityscapes [--model pspnet_resnet50|pspnet_resnet101|deeplab_v3plus_xception65] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_semantic_segmentation_multi.py --dataset cityscapes [--model pspnet_resnet50|pspnet_resnet101|deeplab_v3plus_xception65] [--pretrained-model <model_path>] [--batchsize <batchsize>]
```

### ADE20k

| Base model |  Reference | ChainerCV |
|:-:|:-:|:-:|
| PSPNet w/ Dilated ResNet50 | 41.68 % [1] |  34.97 % |
| PSPNet w/ Dilated ResNet101 |  | 36.55 % |
| DeepLab V3+ w/ Xception65 |  | 42.52 % |

You can reproduce these scores by the following command.

```
$ python eval_semantic_segmentation.py --dataset ade20k [--model pspnet_resnet50|pspnet_resnet101|deeplab_v3plus_xception65] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_semantic_segmentation_multi.py --dataset ade20k [--model pspnet_resnet50|pspnet_resnet101|deeplab_v3plus_xception65] [--pretrained-model <model_path>] [--batchsize <batchsize>]
```

### VOC2012 val

| Base model |  Reference | ChainerCV |
|:-:|:-:|:-:|
| DeepLab V3+ w/ Xception65 | 82.36 % [4] | 82.36 % |

You can reproduce these scores by the following command.

```
$ python eval_semantic_segmentation.py --dataset voc [--model deeplab_v3plus_xception65] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_semantic_segmentation_multi.py --dataset voc [--model deeplab_v3plus_xception65] [--pretrained-model <model_path>] [--batchsize <batchsize>]
```

### CamVid

| Model | Reference | ChainerCV |
|:-:|:-:|:-:|
| SegNet | 46.3 % [3] | 49.4 % |

You can reproduce these scores by the following command.

```
$ python eval_semantic_segmentation.py --dataset camvid [--model segnet] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>]
# with multiple GPUs
$ mpiexec -n <#gpu> python eval_semantic_segmentation_multi.py --dataset camvid [--model segnet] [--pretrained-model <model_path>] [--batchsize <batchsize>]
```


# Reference

1. Hengshuang Zhao et al. "Pyramid Scene Parsing Network" CVPR 2017.
2. https://github.com/holyseven/PSPNet-TF-Reproduce (Validation scores for Cityscapes are lacking in the original paper)
3. Vijay Badrinarayanan et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.
4. Liang-Chieh Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" ECCV, 2018.
