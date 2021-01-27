GPU=$1
DIM=512
NET=gat
ADJ=0.05
WEI=0.2
MASKEPOCH=200
FIXEPOCH=200

S1=1e-3
S2=1e-3
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
--s2 $S2 \
--seed 51
echo syd ------------------------------------------------------

# for i in 1e-2 1e-3 1e-4 1e-5 1e-6
# do
#     S1=$i
#     S2=$i
#     echo syd ------------------------------------------------------
#     echo syd s1: $S1 s2: $S2 adj: ${ADJ} wei: ${WEI}
#     CUDA_VISIBLE_DEVICES=${GPU} \
#     python -u main_gingat_imp.py \
#     --dataset cora \
#     --net ${NET} \
#     --embedding-dim 1433 ${DIM} 7 \
#     --lr 0.008 \
#     --weight-decay 8e-5 \
#     --pruning_percent_wei ${WEI} \
#     --pruning_percent_adj ${ADJ} \
#     --mask_epoch ${MASKEPOCH} \
#     --fix_epoch ${FIXEPOCH} \
#     --s1 $S1 \
#     --s2 $S2
#     echo syd ------------------------------------------------------
# done


# for i in 1e-4 1e-5 1e-6 1e-7
# do
#     S1=$i
#     S2=1e-3
#     echo syd ------------------------------------------------------
#     echo syd s1: $S1 s2: $S2 adj: ${ADJ} wei: ${WEI}
#     CUDA_VISIBLE_DEVICES=${GPU} \
#     python -u main_gingat_imp.py \
#     --dataset cora \
#     --net ${NET} \
#     --embedding-dim 1433 ${DIM} 7 \
#     --lr 0.008 \
#     --weight-decay 8e-5 \
#     --pruning_percent_wei ${WEI} \
#     --pruning_percent_adj ${ADJ} \
#     --mask_epoch ${MASKEPOCH} \
#     --fix_epoch ${FIXEPOCH} \
#     --s1 $S1 \
#     --s2 $S2
#     echo syd ------------------------------------------------------
# done