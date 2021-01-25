GPU=$1
DIM=512
ADJ=0.05
WEI=0.2
EPOCH=200
echo syd ------------------------------------------------------
echo syd adj: ${ADJ} wei: ${WEI}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_random.py \
--dataset cora \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--total_epoch ${EPOCH}
echo syd ------------------------------------------------------