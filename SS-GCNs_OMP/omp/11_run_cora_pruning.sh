GPU=$1
s1=1e-6
s2=5e-5
for i in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do    
    echo syd ------------------------------------------------------
    echo syd s1: $s1 s2: $s2 sparsity: $i
    CUDA_VISIBLE_DEVICES=${GPU} \
    python -u main_pruning.py \
    --dataset cora \
    --embedding-dim 1433 16 7 \
    --lr 0.008 \
    --weight-decay 8e-5 \
    --pruning_percent ${i} \
    --total_epoch 500 \
    --s1 $s1 \
    --s2 $s2
    echo syd ------------------------------------------------------
done