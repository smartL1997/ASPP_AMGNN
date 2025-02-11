
### Dataset Download
The `PTB-XL` dataset can be downloaded from the [Physionet website](https://physionet.org/content/ptb-xl/1.0.1/).

### Implement

win10

1. Download the `PTB_XL` dataset and extract it to `data -- > ptb -- > row`

2. run `ASPP_AMGCN_train.py`

### What each file does

- `ASPP_AMGCN_train.py` training and testing
- `gnn` graph neural network module
- `models` contains scripts for each model
- `utils` contains utilities for `ecg_data`,  and `metrics`

### Logs and checkpoints
- The logs are saved in `logs/` directory.
- The model checkpoints are saved in `checkpoints/` directory.

### Package version
1. python: 3.7
2. cuda: 10.2
3. cudnn: 7605
4. torch: 1.10.2+cu102
5. torch-geometric: 2.0.4
6. torch-scatter: 2.0.9
7. torch-geometric: 2.0.4
8. torch-sparse: 0.6.13
9. torch-spline-conv: 1.2.1
10. torch-cluster: 1.6.0

