python main.py \
--use_gpu \
--conv_encode_edge \
--num_layers 7 \
--dataset ogbg-molhiv \
--block res+ \
--gcn_aggr softmax \
--t 1.0 \
--learn_t \
--dropout 0.2