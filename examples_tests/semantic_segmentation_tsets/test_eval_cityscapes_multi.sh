cd examples/semantic_segmentation
sed -e 's/label_resolution='\''fine'\'')/label_resolution='\''fine'\'').slice[:20]/' \
    -i eval_cityscapes_multi.py

$MPIEXEC $PYTHON eval_cityscapes_multi.py
