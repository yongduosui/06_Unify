GPU=$1
DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-5
S2=1e-2
EPOCH=200
echo syd ------------------------------------------------------
echo syd s1: $S1 S1: $S2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_pretrain.py \
--dataset citeseer \
--embedding-dim 3703 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--weight_dir ../GraphCL/cite_double_dgi.pkl
echo syd ------------------------------------------------------


DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-4
S2=1e-2
EPOCH=200
echo syd ------------------------------------------------------
echo syd s1: $S1 S1: $S2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_pretrain.py \
--dataset citeseer \
--embedding-dim 3703 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--weight_dir ../GraphCL/cite_double_dgi.pkl
echo syd ------------------------------------------------------



DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-4
S2=1e-3
EPOCH=200
echo syd ------------------------------------------------------
echo syd s1: $S1 S1: $S2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_pretrain.py \
--dataset citeseer \
--embedding-dim 3703 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--weight_dir ../GraphCL/cite_double_dgi.pkl
echo syd ------------------------------------------------------



DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-5
S2=1e-3
EPOCH=200
echo syd ------------------------------------------------------
echo syd s1: $S1 S1: $S2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_pretrain.py \
--dataset citeseer \
--embedding-dim 3703 ${DIM} 6 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epoch ${EPOCH} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--weight_dir ../GraphCL/cite_double_dgi.pkl
echo syd ------------------------------------------------------