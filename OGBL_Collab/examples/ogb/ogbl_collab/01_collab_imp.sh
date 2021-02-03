IMP=$1
GPU=$2
S1=1e-6
S2=1e-5
SAVE=IMP_New_setting1
LAYER=28

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_imp.py \
--use_gpu \
--learn_t \
--num_layers ${LAYER} \
--block res+ \
--s1 ${S1} \
--s2 ${S2} \
--mask_epochs 500 \
--fix_epochs 500 \
--model_save_path ${SAVE} \
--imp_num ${IMP}