## Overview
Official code for the AISTATS 2026 paper: **"Generalization Bounds for Spectral GNNs via Fourier Domain Analysis"**.


## Regulariser
To run the main node classification experiments on the benchmark datasets (Cora, Citeseer, Chameleon, Squirrel) with the energy-weighted regularizer:

```bash
python run_spectral_gnn.py --dataset cora --base chebyshev --use-reg 1
```

To run without regularizer:

```bash
python run_spectral_gnn.py --dataset cora --base chebyshev --use-reg 0
```

## Bound computation

To compute the non-linear FTGC bounds and stability bounds on the large-scale ogbn-arxiv dataset:

```bash
cd theoretical_validation
python run_arxiv_validation.py --dataset ogbn-arxiv --base chebyshev --out_dir ./results
```

