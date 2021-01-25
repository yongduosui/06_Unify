GPU=$1
DIM=512
ADJ=0.05
WEI=0.2
EPOCH=200
echo syd ------------------------------------------------------
echo syd adj: ${ADJ} wei: ${WEI}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_random.py \
--dataset citeseer \
--embedding-dim 3703 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--total_epoch ${EPOCH} 
echo syd ------------------------------------------------------