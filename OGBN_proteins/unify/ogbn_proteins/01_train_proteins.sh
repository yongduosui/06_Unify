GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python -u main.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax \
--t 1.0 \
--learn_t \
--dropout 0.1 