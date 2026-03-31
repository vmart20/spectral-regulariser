import os, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
from ogb.nodeproppred import NodePropPredDataset
from utils import set_global_seed, ensure_dir, save_json, normalize_features, build_vandermonde
from layers import SpatialPolynomialLayer
import torchmetrics
import math

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ogbn-arxiv")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--base", type=str, default="chebyshev", choices=["monomial","chebyshev","legendre","bernstein"])
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--samples_per_class", type=int, default=10, help="Number of training samples per class for random split")
    p.add_argument("--eval_freq", type=int, default=10)
    p.add_argument("--patience", type=int, default=100)
    return p.parse_args()

def get_split(y, nclass, percls_trn=10, val_prc=0.35):
    y = y.cpu()
    percls_val = int(round(val_prc * len(y) / nclass))
    indices = []
    for i in range(nclass):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0), device=index.device)]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    valid_index = torch.cat([i[percls_trn:percls_trn + percls_val] for i in indices], dim=0)
    test_index = torch.cat([i[percls_trn + percls_val:] for i in indices], dim=0)
    return train_index, valid_index, test_index

def get_adj_normalized(edge_index, num_nodes):
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Add self loops
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    
    # Compute degree
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(edge_index.size(1), device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
    return adj

class SpatialGNN(nn.Module):
    def __init__(self, dims, adj, basis, K, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(
                SpatialPolynomialLayer(
                    dims[i], dims[i+1], adj, basis=basis, K=K, 
                    activation=nn.ReLU() if i < len(dims)-2 else nn.Identity(),
                    dropout=dropout
                )
            )
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def get_basis_max_norm(basis, K, num_points=1000):
    # Approximate ||V||_{2,inf} by sampling in [-1, 1]
    lambdas = torch.linspace(-1, 1, num_points)
    V = build_vandermonde(lambdas, basis, K) # (num_points, K+1)
    row_norms = torch.norm(V, p=2, dim=1)
    return row_norms.max().item()

def compute_jacobian_norm(model, X: torch.Tensor, num_power_iterations: int = 100) -> float:
    """
    Estimates the spectral norm of the Jacobian ||J||_2 using power iteration.
    J = d(vec(H_L)) / d(vec(H_0))
    """
    was_training = model.training
    model.eval()
    
    # Input shape: (n, d_in)
    n, d_in = X.shape
    
    def forward_fn(x_input):
        return model(x_input)

    # Random vector v of same shape as input (n, d_in)
    v = torch.randn_like(X)
    v = v / (v.norm() + 1e-12)
    
    for _ in range(num_power_iterations):
        # u = J v
        _, u = torch.autograd.functional.jvp(forward_fn, X, v)
        
        # v_new = J^T u
        _, v_new = torch.autograd.functional.vjp(forward_fn, X, u)
        
        # Normalize
        v_norm = v_new.norm()
        if v_norm > 1e-12:
            v = v_new / v_norm
        else:
            break
            
    _, u_final = torch.autograd.functional.jvp(forward_fn, X, v)
    sigma = u_final.norm().item()
    
    model.train(was_training)
    return sigma

@torch.no_grad()
def compute_nonlinear_bound_spatial(model, X0, m_labeled):
    # Xi = sqrt( sum_k || P_k(A) X0 ||_F^2 )
    
    n = X0.shape[0]
    L = len(model.layers)
    
    # 1. Compute Xi using the first layer's basis projections
    first_layer = model.layers[0]
    projections = first_layer.compute_basis_projections(X0)
    
    Xi_sq = 0.0
    for proj in projections:
        Xi_sq += torch.linalg.norm(proj, ord='fro').item()**2
    Xi = math.sqrt(Xi_sq)
    
    # 2. Compute V_2inf and product of norms
    # Assuming homogeneous layers for V_2inf
    basis = first_layer.basis
    K = first_layer.K
    V_2inf = get_basis_max_norm(basis, K)
    
    prod_norms = 1.0
    for layer in model.layers:
        W_norm = torch.linalg.norm(layer.W, ord=2).item()
        theta_norm = torch.linalg.norm(layer.theta, ord=2).item()
        prod_norms *= W_norm * theta_norm
    stability_bound = (V_2inf**(L)) * prod_norms
        
    # Bound formula for FTGC
    bound = (math.sqrt(n)*n/m_labeled/(n-m_labeled)) * (V_2inf**(L-1)) * prod_norms * Xi
    return bound, stability_bound

def load_data(args):
    device = torch.device(args.device)
    dataset = NodePropPredDataset(name=args.dataset, root='../data')
    graph, label = dataset[0]
    
    X = torch.from_numpy(graph['node_feat']).to(device)
    y = torch.from_numpy(label).to(device).squeeze()
    edge_index = torch.from_numpy(graph['edge_index']).to(device)
    
    num_nodes = X.shape[0]
    num_classes = dataset.num_classes
    
    # Normalize features
    X = (X - X.mean(0)) / X.std(0)
    
    # Get sparse adjacency
    adj = get_adj_normalized(edge_index, num_nodes)
        
    return X, y, adj, num_classes

def run_once(args, seed, data):
    set_global_seed(seed)
    device = torch.device(args.device)
    X, y, adj, num_classes = data
    
    
    # Use random split consistent with theory
    train_idx, val_idx, test_idx = get_split(y, num_classes, percls_trn=args.samples_per_class)
    train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)
    
    dims = [X.shape[1]] + [args.width]*(args.depth-1) + [num_classes]
    
    model = SpatialGNN(dims, adj, args.base, args.K, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val = 0
    best_test = 0
    best_state = None
    patience_left = args.patience
    
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        
        if epoch % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                out = model(X)
                train_acc = (out[train_idx].argmax(1) == y[train_idx]).float().mean().item()
                val_acc = (out[val_idx].argmax(1) == y[val_idx]).float().mean().item()
                test_acc = (out[test_idx].argmax(1) == y[test_idx]).float().mean().item()
                
                if val_acc > best_val - 1e-8:
                    best_val = val_acc
                    best_test = test_acc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    patience_left = args.patience
                else:
                    patience_left -= args.eval_freq
                    
                print(f"Epoch {epoch}: Train {train_acc:.4f}, Val {val_acc:.4f}, Test {test_acc:.4f}")
                
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute bound
    model.eval()
    bound, stability_bound = compute_nonlinear_bound_spatial(model, X, len(train_idx))
    jacobian_norm = compute_jacobian_norm(model, X)
    
    # Generalization gap (Train Loss - Test Loss)
    with torch.no_grad():
        out = model(X)
        train_loss = criterion(out[train_idx], y[train_idx]).item()
        test_loss = criterion(out[test_idx], y[test_idx]).item()
        gap = abs(train_loss - test_loss)

    results = {
        "seed": seed,
        "val_acc": best_val,
        "test_acc": best_test,
        "nonlinear_bound": bound,
        "gen_gap": gap,
        "stability_bound": stability_bound,
        "jacobian_norm": jacobian_norm
    }
    return results

def main():
    args = parse_args()
    print(args.device)
    ensure_dir(args.out_dir)
    
    data = load_data(args)
    
    for seed in args.seeds:
        res = run_once(args, seed, data)
        print(json.dumps(res, indent=2))
        save_json(os.path.join(args.out_dir, f"results_seed{seed}.json"), res)

if __name__ == "__main__":
    main()
