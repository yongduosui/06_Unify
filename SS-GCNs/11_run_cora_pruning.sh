GPU=$1

i=0.1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning.py \
--dataset cora \
--embedding-dim 1433 16 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent ${i} \
--total_epoch 300

# i=0.1
# for s1 in 1e-4 5e-5 1e-5 5e-6 1e-6
# do
#     for s2 in 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6
#     do
#         echo syd ------------------------------------------------------
#         echo syd s1: $s1 s2: $s2
#         CUDA_VISIBLE_DEVICES=${GPU} \
#         python -u main_pruning.py \
#         --dataset cora \
#         --embedding-dim 1433 16 7 \
#         --lr 0.008 \
#         --weight-decay 8e-5 \
#         --pruning_percent ${i} \
#         --total_epoch 300 \
#         --s1 $s1 \
#         --s2 $s2
#         echo syd ------------------------------------------------------
#     done
# done

# i=0.1
# CUDA_VISIBLE_DEVICES=${GPU} \
# python -u main_pruning.py \
# --dataset cora \
# --embedding-dim 1433 16 7 \
# --lr 0.008 \
# --weight-decay 8e-5 \
# --pruning_percent ${i} \
# --total_epoch 300


# for i in {0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
# do
#     echo ---------
#     echo syd ${i}
#     CUDA_VISIBLE_DEVICES=${GPU} \
#     python -u main_pruning.py \
#     --dataset cora \
#     --embedding-dim 1433 16 7 \
#     --lr 0.008 \
#     --weight-decay 8e-5 \
#     --pruning_percent ${i}
# done