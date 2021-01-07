GPU=$1
PER=0.1
SEED=31
python -u main_single.py \
--dataset citeseer \
--aug_type node \
--drop_percent ${PER} \
--seed ${SEED} \
--save_name cite_single_dgi.pkl \
--gpu ${GPU}