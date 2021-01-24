IMPNUM=$1
GPU=$2
EPOCH=100
ITER=10
SAVE=RP

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_rp.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--epochs ${EPOCH} \
--iteration ${ITER} \
--model_save_path ${SAVE} \
--imp_num ${IMPNUM}