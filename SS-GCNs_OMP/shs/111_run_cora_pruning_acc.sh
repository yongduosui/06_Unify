GPU=$1
i=0.1
# echo ---------
# echo syd ${i}
# CUDA_VISIBLE_DEVICES=${GPU} \
# python -u main_pruning_acc.py \
# --dataset cora \
# --embedding-dim 1433 16 7 \
# --lr 0.008 \
# --weight-decay 8e-5 \
# --pruning_percent ${i} \
# --total_epoch 500 \
# --s1 5e-4 \
# --s2 1e-4
for s1 in 1e-4 5e-5 1e-5 5e-6 1e-6
do
    for s2 in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6
    do
        echo syd ------------------------------------------------------
        echo syd s1: $s1 s2: $s2
        CUDA_VISIBLE_DEVICES=${GPU} \
        python -u main_pruning_acc.py \
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
done