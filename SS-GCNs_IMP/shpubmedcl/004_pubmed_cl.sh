GPU=$1
DIM=512
INIT=all_one
ADJ=0.05
WEI=0.2
S1=1e-7
S2=1e-6
echo syd ------------------------------------------------------
echo syd s1: $s1 S1: $S2 adj: ${ADJ} wei: ${WEI} init: ${INIT}
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_pruning_imp_pretrain.py \
--dataset pubmed \
--embedding-dim 500 ${DIM} 3 \
--lr 0.01 \
--weight-decay 5e-4 \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--s1 $S1 \
--s2 $S2 \
--init_soft_mask_type ${INIT} \
--weight_dir ../GraphCL/pubmed_double_dgi.pkl
echo syd ------------------------------------------------------