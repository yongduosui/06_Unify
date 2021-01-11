GPU=$1
EXP=debug_ckpt
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--save debug_ckpt