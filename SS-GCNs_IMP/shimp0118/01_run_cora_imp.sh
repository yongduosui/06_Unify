GPU=$1
DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-5
S2=1e-5
EPOCH=500
echo syd ------------------------------------------------------
echo syd s1: $s1 s2: $s2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp.py \
--dataset cora \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--total_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--rewind_soft_mask
echo syd ------------------------------------------------------

DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-6
S2=1e-6
EPOCH=500
echo syd ------------------------------------------------------
echo syd s1: $s1 s2: $s2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp.py \
--dataset cora \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--total_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--rewind_soft_mask
echo syd ------------------------------------------------------

DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-7
S2=1e-7
EPOCH=500
echo syd ------------------------------------------------------
echo syd s1: $s1 s2: $s2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp.py \
--dataset cora \
--embedding-dim 1433 ${DIM} 7 \
--lr 0.008 \
--weight-decay 8e-5 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--total_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--rewind_soft_mask
echo syd ------------------------------------------------------