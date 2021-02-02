GPU=$1
EXP=debug_no_fixed44
S1=1e-4
S2=1e-4
FIX=no_fixed
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--use_gpu \
--self_loop \
--num_layers 3 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--save ${EXP} \
--s1 ${S1} \
--s2 ${S2} \
--fixed ${FIX} \
--epochs 100

# EXP=debug_no_fixed55
# S1=1e-5
# S2=1e-5
# FIX=no_fixed
# CUDA_VISIBLE_DEVICES=${GPU} \
# python -u main.py \
# --use_gpu \
# --self_loop \
# --num_layers 28 \
# --block res+ \
# --gcn_aggr softmax_sg \
# --t 0.1 \
# --save ${EXP} \
# --s1 ${S1} \
# --s2 ${S2} \
# --fixed ${FIX} \
# --epochs 100

# EXP=debug_no_fixed66
# S1=1e-6
# S2=1e-6
# FIX=no_fixed
# CUDA_VISIBLE_DEVICES=${GPU} \
# python -u main.py \
# --use_gpu \
# --self_loop \
# --num_layers 28 \
# --block res+ \
# --gcn_aggr softmax_sg \
# --t 0.1 \
# --save ${EXP} \
# --s1 ${S1} \
# --s2 ${S2} \
# --fixed ${FIX} \
# --epochs 100