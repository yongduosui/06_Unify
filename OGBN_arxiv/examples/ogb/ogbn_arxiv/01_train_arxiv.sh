GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1