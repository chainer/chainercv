# Examples of Pyramid Scene Parsing Network (PSPNet)

## Performance

| Model | Reference | ChainerCV (weight conversion) |
|:-:|:-:|:-:|
| Cityscapes (single scale) | 79.70 % * | 79.03 % |

Scores are mean Intersection over Union (mIoU).


## Demo
This demo downloads Cityscapes pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--gpu <gpu>] [--pretrained_model <model_path>] <image>.jpg
```

## Convert Caffe model
Convert `*.caffemodel` to `*.npz`. Some layers are renamed to fit ChainerCV.
```
$ python caffe2npz.py <source>.caffemodel <target>.npz
```

## Evaluation
The evaluation can be conducted using [`chainercv/examples/semantic_segmentation/eval_cityscapes.py`](https://github.com/chainer/chainercv/blob/master/examples/semantic_segmentation).

## References
1. Hengshuang Zhao et al. "Pyramid Scene Parsing Network" CVPR 2017.
2. [chainer-pspnet by mitmul](https://github.com/mitmul/chainer-pspnet)
