cd examples/fpn

$FLEXCI_SKIP $PYTHON demo.py --model faster_rcnn_fpn_resnet50 $SAMPLE_IMAGE
$PYTHON demo.py --model faster_rcnn_fpn_resnet50 --gpu 0 $SAMPLE_IMAGE
$FLEXCI_SKIP $PYTHON demo.py --model faster_rcnn_fpn_resnet101 $SAMPLE_IMAGE
$PYTHON demo.py --model faster_rcnn_fpn_resnet101 --gpu 0 $SAMPLE_IMAGE

$FLEXCI_SKIP $PYTHON demo.py --model mask_rcnn_fpn_resnet50 $SAMPLE_IMAGE
$PYTHON demo.py --model mask_rcnn_fpn_resnet50 --gpu 0 $SAMPLE_IMAGE
$FLEXCI_SKIP $PYTHON demo.py --model mask_rcnn_fpn_resnet101 $SAMPLE_IMAGE
$PYTHON demo.py --model mask_rcnn_fpn_resnet101 --gpu 0 $SAMPLE_IMAGE
