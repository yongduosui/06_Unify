GPU=$1
s1=1e-5
s2=5e-4
i=0.1

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_omp_find_seed.py \
--dataset citeseer \
--embedding-dim 3703 16 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent $i \
--total_epoch 400 \
--s1 $s1 \
--s2 $s2