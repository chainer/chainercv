cd examples/ssd
sed -e 's/return_difficult=True)/return_difficult=True).slice[:20]/' -i train.py

$PYTHON train.py --model ssd300 --batchsize 2 --iteration 12 --step 8 10 --gpu 0
$PYTHON train.py --model ssd512 --batchsize 2 --iteration 12 --step 8 10 --gpu 0
