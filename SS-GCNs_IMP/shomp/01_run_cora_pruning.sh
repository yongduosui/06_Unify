GPU=$1
DIM=16
adj=0.05
wei=0.1
s1=1e-6
s2=5e-5
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_omp.py \
--dataset cora \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent_wei ${wei} \
--pruning_percent_adj ${adj} \
--total_epoch 500 \
--s1 $s1 \
--s2 $s2