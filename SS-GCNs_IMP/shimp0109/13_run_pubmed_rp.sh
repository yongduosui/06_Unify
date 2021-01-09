GPU=$1
DIM=512
ADJ=0.05
WEI=0.1
S1=1e-5
S2=5e-4
EPOCH=500
echo syd ------------------------------------------------------
echo syd s1: $s1 s2: $s2 adj: ${ADJ} wei: ${WEI}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_random.py \
--dataset pubmed \
--embedding-dim 500 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--total_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2
echo syd ------------------------------------------------------