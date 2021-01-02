GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning.py \
--dataset cora \
--embedding-dim 1433 16 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent 0.1