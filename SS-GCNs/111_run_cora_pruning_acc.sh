GPU=$1
i=0.1
echo ---------
echo syd ${i}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_acc.py \
--dataset cora \
--embedding-dim 1433 16 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent ${i} \
--total_epoch 500 \
--s1 1e-3 \
--s2 1e-4