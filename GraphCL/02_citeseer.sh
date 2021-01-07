GPU=$1
PER=0.1
SEED=31
python -u execute.py \
--dataset citeseer \
--aug_type node \
--drop_percent ${PER} \
--seed ${SEED} \
--save_name cite_best_dgi.pkl \
--gpu ${GPU}