cd examples/semantic_segmentation

sed -e 's/CamVidDataset(split='\''test'\'')/CamVidDataset(split='\''test'\'').slice[:20]/' -i eval_semantic_segmentation.py
$MPIEXEC $PYTHON eval_semantic_segmentation_multi.py --dataset camvid --model segnet

sed -e 's/label_resolution='\''fine'\'')/label_resolution='\''fine'\'').slice[:20]/' \
    -i eval_semantic_segmentation.py
$MPIEXEC $PYTHON eval_semantic_segmentation_multi.py --dataset cityscapes --model pspnet_resnet101
$MPIEXEC $PYTHON eval_semantic_segmentation_multi.py --dataset cityscapes --model deeplab_v3plus_xception65
