
import os, json, csv
from typing import Dict, Any, Tuple
import random
import math
import numpy as np
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def append_csv(path: str, row: Dict[str, Any], header_order=None):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_order or list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_features(X: torch.Tensor, method="row_standardize"):
    if method == "row_standardize":
        mu = X.mean(dim=1, keepdim=True)
        std = X.std(dim=1, keepdim=True) + 1e-6
        return (X - mu) / std
    elif method == "row_l2":
        nrm = X.norm(p=2, dim=1, keepdim=True) + 1e-6
        return X / nrm
    else:
        return X

def get_split(y, nclass, seed=0, val_prc=0.35, percls_trn=10, train_prc=None):
    y = y.cpu()
    g = torch.Generator().manual_seed(seed)
    if percls_trn is None or percls_trn == "None":
        percls_trn = int(round(train_prc * len(y) / nclass))
    indices = []
    for i in range(nclass):
        index = (y == i).nonzero(as_tuple=True)[0]
        perm = torch.randperm(index.size(0), generator=g)
        indices.append(index[perm])
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    val_index = torch.cat([i[percls_trn:percls_trn+int(round(val_prc * (i.size(0)-percls_trn)))] for i in indices], dim=0)
    test_index = torch.cat([i[percls_trn+int(round(val_prc * (i.size(0)-percls_trn))):] for i in indices], dim=0)
    return train_index, val_index, test_index


def load_eigendecomp(data_dir: str, device=None):
    U = np.load(os.path.join(data_dir, "U.npy"))
    lambdas = np.load(os.path.join(data_dir, "lambdas.npy"))
    U = torch.from_numpy(U).float()
    lambdas = torch.from_numpy(lambdas).float()
    if device is not None:
        U = U.to(device)
        lambdas = lambdas.to(device)
    return U, lambdas

def load_graph_data(data_dir: str, device=None):
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    train_idx = torch.from_numpy(train_idx).long()
    val_idx = torch.from_numpy(val_idx).long()
    test_idx = torch.from_numpy(test_idx).long()
    if device is not None:
        X = X.to(device); y = y.to(device)
        train_idx = train_idx.to(device); val_idx = val_idx.to(device); test_idx = test_idx.to(device)
    return X, y, train_idx, val_idx, test_idx

def build_vandermonde(lambdas: torch.Tensor, basis: str, K: int) -> torch.Tensor:
    """
    Returns V_P (n x (K+1)) for the chosen basis evaluated at eigenvalues 'lambdas' (shape: n,).
    Basis options: 'monomial', 'chebyshev', 'legendre','bernstein'.
    """
    n = lambdas.shape[0]
    V = torch.empty((n, K+1), dtype=lambdas.dtype, device=lambdas.device)
    if basis == "monomial":
        # V[i,k] = lambda_i^k
        V[:, 0] = 1.0
        if K >= 1:
            V[:, 1] = lambdas
        for k in range(2, K+1):
            V[:, k] = V[:, k-1] * lambdas
    elif basis == "chebyshev":
        # Chebyshev T_0=1, T_1=lambda, T_{k+1}=2*lambda*T_k - T_{k-1}
        V[:, 0] = 1.0
        if K >= 1:
            V[:, 1] = lambdas
        for k in range(1, K):
            V[:, k+1] = 2.0 * lambdas * V[:, k] - V[:, k-1]
    elif basis == "legendre":
        V[:, 0] = 1.0
        if K >= 1:
            V[:, 1] = lambdas
        for k in range(1, K):
            V[:, k + 1] = ((2 * k + 1) * lambdas * V[:, k] - k * V[:, k - 1]) / (k + 1)
    elif basis == "bernstein":
        # Map lambda in [-1,1] to t in [0,1]: t=(lambda+1)/2
        t = (lambdas + 1.0) / 2.0
        # V[i,k] = binom(K,k) t^k (1-t)^{K-k}
        # Compute all powers efficiently
        t_pows = [torch.ones_like(t)]
        for k in range(1, K+1):
            t_pows.append(t_pows[-1] * t)
        one_minus_t = 1.0 - t
        omt_pows = [torch.ones_like(t)]
        for j in range(1, K+1):
            omt_pows.append(omt_pows[-1] * one_minus_t)
        # Precompute binomial coefficients
        binoms = [math.comb(K, k) for k in range(K+1)]
        for k in range(K+1):
            V[:, k] = binoms[k] * t_pows[k] * omt_pows[K-k]
    else:
        raise ValueError(f"Unknown basis: {basis}")
    return V