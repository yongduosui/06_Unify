IMPNUM=$1
GPU=$2
EPOCH=100
ITER=10
S1=0.1
S2=1e-3
SAVE=IMP

CUDA_VISIBLE_DEVICES=${GPU} \
python -u main_imp.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--s1 ${S1} \
--s2 ${S2} \
--epochs ${EPOCH} \
--iteration ${ITER} \
--model_save_path ${SAVE} \
--imp_num ${IMPNUM}
# --resume_dir CKPT/IMP/IMP1_train_ckpt.pth