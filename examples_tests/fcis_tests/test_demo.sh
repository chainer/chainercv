cd examples/fcis

$FLEXCI_SKIP $PYTHON demo.py --dataset sbd $SAMPLE_IMAGE
$PYTHON demo.py --dataset sbd --gpu 0 $SAMPLE_IMAGE
$FLEXCI_SKIP $PYTHON demo.py --dataset coco $SAMPLE_IMAGE
$PYTHON demo.py --dataset coco --gpu 0 $SAMPLE_IMAGE
