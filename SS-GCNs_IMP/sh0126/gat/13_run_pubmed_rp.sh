GPU=$1
DIM=512
NET=gat
ADJ=0.05
WEI=0.2
FIXEPOCH=200
echo syd ------------------------------------------------------
echo syd adj: ${ADJ} wei: ${WEI}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_gingat_rp.py \
--dataset pubmed \
--net ${NET} \
--embedding-dim 500 ${DIM} 3 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--fix_epoch ${FIXEPOCH} \
--seed 53
echo syd ------------------------------------------------------