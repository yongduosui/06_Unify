GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_acc.py \
--dataset pubmed \
--embedding-dim 500 16 3 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent 0.1 \
--total_epoch 1000 \
--s1 1e-5 \
--s2 1e-5