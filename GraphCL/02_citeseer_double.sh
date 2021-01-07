GPU=$1
PER=0.05
SEED=31
python -u main_double.py \
--dataset citeseer \
--aug_type node \
--drop_percent ${PER} \
--seed ${SEED} \
--save_name cite_double_dgi.pkl \
--gpu ${GPU}