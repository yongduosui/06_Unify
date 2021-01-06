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

## 1. IMP
| Sparsity (0.9^n) | cora (seed:307) | citeseer (seed:118) | pubmed |
| :---:| :---: | :---: | :---: | 
| 0  | 81.03±0.64 | 70.94±0.77  | 79.16±0.19 |
| 1  | 82.1 | 71.5 | - |
| 2  | 79.3 | 70.9 | - |
| 3  | 75.8 | 68.2 | - |
| 4  | 72.2 | 66.4 | - |
| 5  | 68.7 | 61.9 | - |
| 6  | 62.6 | 58.5 | - |
| 7  | 56.3 | 53.4 | - |
| 8  | 48.7 | 50.3 | - |
| 9  | 41.6 | 43.7 | - |
| 10 | 34.3 | 39.4 | - |

## 2. OMP