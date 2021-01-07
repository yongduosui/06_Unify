GPU=$1
PER=0.5
SEED=39
python -u main_single.py \
--dataset cora \
--aug_type subgraph \
--drop_percent ${PER} \
--seed ${SEED} \
--save_name cora_single_dgi.pkl \
--gpu ${GPU}
