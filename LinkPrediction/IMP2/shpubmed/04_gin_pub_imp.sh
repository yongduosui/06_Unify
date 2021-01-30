DATASET=pubmed
S1=1e-4
S2=1e-4
MASK_EPOCH=100
FIX_EPOCH=200
CUDA_VISIBLE_DEVICES=$1 \
python -u main_gingat_imp.py \
--net gin \
--dataset ${DATASET} \
--mask_epoch ${MASK_EPOCH} \
--fix_epoch ${FIX_EPOCH} \
--s1 ${S1} \
--s2 ${S2}