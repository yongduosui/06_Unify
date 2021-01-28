GPU=$1
DIM=512
INIT=all_one
ADJ=0.05
WEI=0.4
EPOCH=200
S1=1e-2
S2=1e-2
echo syd ------------------------------------------------------
echo syd s1: $S1 s2: $S2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_pretrain.py \
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
--weight_dir ../GraphCL/cora_double_dgi.pkl
echo syd ------------------------------------------------------