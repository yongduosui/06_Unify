GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--use_gpu \
--self_loop \
--num_layers 7 \
--block res+ \
--gcn_aggr softmax \
--t 1.0
