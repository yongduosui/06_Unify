GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning.py \
--dataset pubmed \
--embedding-dim 500 16 3 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent 0.1 \
--total_epoch 300 \
--s1 1e-5 \
--s2 1e-5

# for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
# do
#     echo ---------
#     echo syd ${i}
#     CUDA_VISIBLE_DEVICES=${GPU} \
#     python -u main_pruning.py \
#     --dataset citeseer \
#     --embedding-dim 3703 16 6 \
#     --lr 0.01 \
#     --weight-decay 5e-4 \
#     --pruning_percent ${i}
# done

