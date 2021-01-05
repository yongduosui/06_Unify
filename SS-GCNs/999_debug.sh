GPU=$1
s1=1e-5
s2=1e-5
echo syd ------------------------------------------------------
echo syd s1: $s1 s2: $s2
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning.py \
--dataset cora \
--embedding-dim 1433 16 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent 0.1 \
--total_epoch 300 \
--s1 $s1 \
--s2 $s2
echo syd ------------------------------------------------------