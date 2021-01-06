GPU=$1
i=0.1
s1=1e-5
s2=5e-4
echo syd ------------------------------------------------------
echo syd s1: $s1 s2: $s2 sp: $i
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp.py \
--dataset citeseer \
--embedding-dim 3703 16 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent $i \
--total_epoch 500 \
--s1 $s1 \
--s2 $s2
echo syd ------------------------------------------------------
