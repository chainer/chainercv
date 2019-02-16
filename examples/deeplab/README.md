# Examples of DeepLab

## Performance
DeepLab V3+

| Network backborn | Training | Evaluation | Eval scales | Reference | ChainerCV (weight conversion) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Xception65 | VOC2012 trainaug | VOC2012 val | (1.0,) |  82.36 % * |  82.36 % |
| Xception65 | Cityscapes train fine | Cityscapes val fine | (1.0,) | 79.12 % * | 79.14 % |

Scores are measured by mean Intersection over Union (mIoU).  
\*: Although the official repository reports a score of multi-scale prediciton, public pretrained graph is for single-scale prediction.
So we evaluated the pretrained graph using `eval_semantic_segmentation` in ChainerCV.

## Demo
This demo downloads Citåyscapes pretrained model automatically if a pretrained model path is not given.
```
$ python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] [--input-size <size>] <image>.jpg
```


## Convert Tensorflow Frozen Graph
Convert `frozen_inference_graph.pb` distributed in official repository to `*.npz`. Some layers are renamed to fit ChainerCV.
Official repository is [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).

```
$ python tf2npz.py <task: {voc, cityscapes, ade20k}> path/to/frozen_inference_graph.pb <target>.npz
```


## Evaluation
The evaluation can be conducted using [`chainercv/examples/semantic_segmentation/eval_cityscapes.py`](https://github.com/chainer/chainercv/blob/master/examples/semantic_segmentation).


## References
1. Liang-Chieh Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" ECCV, 2018.
2. [official repository](https://github.com/tensorflow/models/tree/master/research/deeplab)
