GPU=$1
DIM=16
adj=0.05
wei=0.1
s1=1e-5
s2=5e-4
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_omp.py \
--dataset citeseer \
--embedding-dim 3703 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${wei} \
--pruning_percent_adj ${adj} \
--total_epoch 500 \
--s1 $s1 \
--s2 $s2
