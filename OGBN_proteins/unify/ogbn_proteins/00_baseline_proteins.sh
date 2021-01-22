GPU=$1
EPOCH=100
ITER=10
SAVE=Baseline

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_baseline.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--mlp_layers 2 \
--epochs ${EPOCH} \
--iteration ${ITER} \
--model_save_path ${SAVE}