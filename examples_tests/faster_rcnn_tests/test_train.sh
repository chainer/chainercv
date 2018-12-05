cd examples/faster_rcnn
sed -e 's/return_difficult=True)/return_difficult=True).slice[:20]/' -i train.py

$PYTHON train.py --dataset voc07 --gpu 0 --step-size 5 --iteration 7
$PYTHON train.py --dataset voc0712 --gpu 0 --step-size 5 --iteration 7

