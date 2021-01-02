# Unify Pruning On GCN
## 1. Related Repo

https://github.com/Shen-Lab/SS-GCNs

## 2. Experiments

### 2.1 Baseline

| Task | Dataset | Node Num | Edge Num | Baseline (20 seeds) | Avg Epoch |
| :---:| :---: | :---: | :---: | :---: |:---: |
| Node-level | cora    | 2708 |  13264  | 81.03±0.64 | 236.10 |
| Node-level | citeseet| 3327 |  12431  | 70.94±0.77 | 236.95 |

### 2.2 Pruning

| Pruning % | cora | citeseet |
| :---:| :---: | :---: | 
| 10 | 79.19±0.63 | - |
| 20 |  | -  |
| 30 |  | -  |
| 40 |  | -  |
| 50 |  | -  |
| 60 |  | -  |
| 70 |  | -  |
| 80 |  | -  |
| 90 |  | -  |