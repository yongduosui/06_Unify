GPU=$1
EXP=debug_all_fixed
S1=1e-4
S2=1e-4
FIX=no_fixed

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--save ${EXP} \
--s1 ${S1} \
--s2 ${S2} \
--fixed ${FIX}