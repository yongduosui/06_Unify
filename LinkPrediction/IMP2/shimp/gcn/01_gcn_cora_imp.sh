DATASET=cora
S1=1e-2
S2=1e-2
MASK_EPOCH=200
FIX_EPOCH=200
CUDA_VISIBLE_DEVICES=$1 \
python -u main_gcn_imp.py \
--dataset ${DATASET} \
--mask_epoch ${MASK_EPOCH} \
--fix_epoch ${FIX_EPOCH} \
--s1 ${S1} \
--s2 ${S2}