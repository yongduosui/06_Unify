GPU=$1
DIM=512
NET=gat
ADJ=0.05
WEI=0.2
MASKEPOCH=10
FIXEPOCH=10
S1=1e-2
S2=1e-2
echo syd ------------------------------------------------------
echo syd s1: $S1 s2: $S2 adj: ${ADJ} wei: ${WEI}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_gingat_imp.py \
--dataset cora \
--net ${NET} \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epoch ${MASKEPOCH} \
--fix_epoch ${FIXEPOCH} \
--s1 $S1 \
--s2 $S2
echo syd ------------------------------------------------------