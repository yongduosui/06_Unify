GPU=$1
LAYER=1

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_baseline.py \
--use_gpu \
--learn_t \
--num_layers ${LAYER} \
--block res+ \
--mask_epochs 500