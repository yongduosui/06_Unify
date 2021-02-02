GPU=$1
SAVE=RP

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_rp.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--epochs 500 \
--model_save_path ${SAVE}