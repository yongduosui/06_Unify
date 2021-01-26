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
--dataset cora \
--net ${NET} \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--fix_epoch ${FIXEPOCH}
echo syd ------------------------------------------------------