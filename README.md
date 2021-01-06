# Unify Pruning On GCN
## 1. Related Repo

https://github.com/Shen-Lab/SS-GCNs

## 2. Experiments

### 2.1 Baseline

| Task | Dataset | Node Num | Edge Num | Dim | Classes | Baseline (20 seeds) | Avg Epoch |
| :---:| :---: | :---: | :---: | :---: |:---: |:---: |:---: |
| Node-level | cora    | 2708 |  13264  | 1433 | 7 | 81.03±0.64 | 236.10 |
| Node-level | citeseer| 3327 |  4732   | 3703 | 6 | 70.94±0.77 | 236.95 |
| Node-level | pubmed  |19717 | 108365  | 500  | 3 | 79.16±0.19 | 152.15 |

### 2.2 Unify Pruning


| Pruning % | cora | citeseer | pubmed |
| :---:| :---: | :---: | :---: | 
| 0  | 81.03±0.64 | 70.94±0.77  | 79.16±0.19 |
| 10 | 80.72±0.72 | 70.84±0.37  | 79.30±0.23 |