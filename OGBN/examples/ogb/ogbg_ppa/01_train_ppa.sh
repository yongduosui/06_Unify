GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
--use_gpu \
--conv_encode_edge \
--num_layers 28 \
--gcn_aggr softmax_sg \
--t 0.01