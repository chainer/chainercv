cd examples/fpn
curl -L https://cloud.githubusercontent.com/assets/2062128/26187667/9cb236da-3bd5-11e7-8bcf-7dbd4302e2dc.jpg \
     -o sample.jpg

$PFNCI_SKIP $PYTHON demo.py --model faster_rcnn_fpn_resnet50 sample.jpg
$PYTHON demo.py --model faster_rcnn_fpn_resnet50 --gpu 0 sample.jpg
$PFNCI_SKIP $PYTHON demo.py --model faster_rcnn_fpn_resnet101 sample.jpg
$PYTHON demo.py --model faster_rcnn_fpn_resnet101 --gpu 0 sample.jpg
