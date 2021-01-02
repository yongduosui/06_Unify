# Unify Pruning On GCN
## 1. Related Repo

https://github.com/Shen-Lab/SS-GCNs

## 2. Experiments

### 2.1 Baseline

| Task | Dataset | Node Num | Edge Num | Baseline (20 seeds) | Avg Epoch |
| :---:| :---: | :---: | :---: | :---: |:---: |
| Node-level | cora    | 2708 |  13264  | 81.03±0.64 | 236.10 |
| Node-level | citeseet| 3327 |  12431  | 70.94±0.77 | 236.95 |

### 2.2 OMP Pruning

| Pruning % | cora | citeseet |
| :---:| :---: | :---: | 
| 0  | 81.03±0.64 | 70.94±0.77  | 
| 10 | 79.19±0.63 | 42.80±15.05 |
| 20 | 77.18±0.87 | 43.12±15.11 |
| 30 | 73.30±1.01 | 43.54±14.55 |
| 40 | 67.24±1.24 | 42.02±13.74 |
| 50 | 57.25±1.31 | 41.70±9.50  |
| 60 | 41.37±0.93 | 35.44±9.20  |
| 70 | 26.28±0.12 | 30.04±4.99  |
| 80 | 25.31±0.22 | 20.93±2.04  |
| 90 | 20.51±0.32 | 11.70±0.75  |

### 2.3 IMP Pruning

| Pruning 0.9^n | cora | citeseet |
| :---:| :---: | :---: | 
| 0 | 81.03±0.64 | - |
| 1 | 79.19±0.63 | - |
| 2 | ± | -  |
| 3 | ± | -  |
| 4 | ± | -  |
| 5 | ± | -  |
| 6 | ± | -  |
| 7 | ± | -  |
| 8 | ± | -  |
| 9 | ± | -  |