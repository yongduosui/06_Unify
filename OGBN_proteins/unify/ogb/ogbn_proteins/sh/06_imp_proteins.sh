GPU=$1
EPOCH=100
ITER=10
WEI=0.7379
ADJ=0.2649
S1=0.1
S2=1e-3
SAVE=IMP
IMPNUM=6

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_imp.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--s1 ${S1} \
--s2 ${S2} \
--pruning_percent_wei ${WEI} \
--pruning_percent_adj ${ADJ} \
--epochs ${EPOCH} \
--iteration ${ITER} \
--model_save_path ${SAVE} \
--imp_num ${IMPNUM}