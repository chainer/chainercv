cd examples/ssd
sed -e 's/return_difficult=True)/return_difficult=True).slice[:20]/' -i train_multi.py

$MPIEXEC $PYTHON train_multi.py --model ssd300 --batchsize 4 --test-batchsize 2 --iteration 12 --step 8 10
$MPIEXEC $PYTHON train_multi.py --model ssd512 --batchsize 4 --test-batchsize 2 --iteration 12 --step 8 10
