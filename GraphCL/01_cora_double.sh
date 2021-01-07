GPU=$1
PER=0.1
SEED=39
python -u main_double.py \
--dataset cora \
--aug_type subgraph \
--drop_percent ${PER} \
--seed ${SEED} \
--save_name cora_double_dgi.pkl \
--gpu ${GPU}
