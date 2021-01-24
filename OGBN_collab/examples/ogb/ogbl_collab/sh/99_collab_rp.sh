RP=$1
GPU=$2
SAVE=RP
LAYER=1


CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_rp.py \
--use_gpu \
--learn_t \
--num_layers ${LAYER} \
--block res+ \
--fix_epochs 500 \
--imp_num ${RP} \
--model_save_path ${SAVE}