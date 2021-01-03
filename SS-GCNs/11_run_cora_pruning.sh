GPU=$1

i=0.1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning.py \
--dataset cora \
--embedding-dim 1433 16 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent ${i}

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