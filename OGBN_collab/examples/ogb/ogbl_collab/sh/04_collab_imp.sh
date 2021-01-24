GPU=$1
WEI=0.2
ADJ=0.05
S1=1e-5
S2=1e-4
SAVE=IMP_setting1
LAYER=28

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_imp.py \
--use_gpu \
--learn_t \
--num_layers ${LAYER} \
--block res+ \
--s1 ${S1} \
--s2 ${S2} \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--mask_epochs 500 \
--fix_epochs 500 \
--model_save_path ${SAVE}