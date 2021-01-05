# Unify Pruning On GCN
## 1. Related Repo

https://github.com/Shen-Lab/SS-GCNs

## 2. Experiments

### 2.1 Baseline

| Task | Dataset | Node Num | Edge Num | Baseline (20 seeds) | Avg Epoch |
| :---:| :---: | :---: | :---: | :---: |:---: |
| Node-level | cora    | 2708 |  13264  | 81.03±0.64 | 236.10 |
| Node-level | citeseet| 3327 |  12431  | 70.94±0.77 | 236.95 |
| Node-level | pubmed  | 3327 |  12431  | 70.94±0.77 | 236.95 |

### 2.2 Unify Pruning


| Pruning % | cora | citeseet | pubmed |
| :---:| :---: | :---: | :---: | 
| 0  | 81.03±0.64 | 70.94±0.77  | - |
| 10 | 80.45±0.67 | 42.80±15.05 | - |