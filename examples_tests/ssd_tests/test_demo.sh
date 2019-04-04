cd examples/ssd

$PYTHON demo.py --model ssd300 $SAMPLE_IMAGE
$PYTHON demo.py --model ssd300 --gpu 0 $SAMPLE_IMAGE
$PYTHON demo.py --model ssd512 $SAMPLE_IMAGE
$PYTHON demo.py --model ssd512 --gpu 0 $SAMPLE_IMAGE
