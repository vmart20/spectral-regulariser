import torch

def get_base_matrix(a, k, base):
    if base == "chebyshev":
        return create_chebyshev_matrix(a, k)
    elif base == "legendre":
        return create_legendre_matrix(a, k)
    elif base == "monomial":
        return create_monomial_matrix(a, k)
    elif base == "bernstein":
        return create_bernstein_matrix(a, k)
    else:
        raise ValueError(f"Unknown base type: {base}")

def create_bernstein_matrix(a, k):
    n = a.size(0)
    V = torch.zeros(n, k, dtype=a.dtype, device=a.device)

    # Map [-1, 1] -> [0, 1]
    t = (a + 1) * 0.5

    j = torch.arange(k, device=a.device, dtype=a.dtype)
    n_deg = k - 1


    lgamma = torch.lgamma
    n_plus_1 = torch.tensor(float(n_deg + 1), device=a.device, dtype=a.dtype)
    binom = torch.exp(
        lgamma(n_plus_1)
        - lgamma(j + 1.0)
        - lgamma(n_deg - j + 1.0)
    )
    t_col = t.unsqueeze(1)                 
    one_minus_t_col = (1.0 - t).unsqueeze(1)

    V = binom * (t_col ** j) * (one_minus_t_col ** (n_deg - j))

    V = V / torch.max(torch.norm(V, dim=1, p=2))

    return V

def create_chebyshev_matrix(a, k):
    n = a.size(0)
    V_cheb = torch.zeros(n, k, dtype=a.dtype, device=a.device)
    V_cheb[:, 0] = 1.0
    if k > 1:
        V_cheb[:, 1] = a
    for j in range(2, k):
        V_cheb[:, j] = 2 * a * V_cheb[:, j - 1] - V_cheb[:, j - 2]
    
    V_cheb = V_cheb / torch.max(torch.norm(V_cheb, dim=1, p=2))

    return V_cheb

def create_legendre_matrix(a: torch.Tensor, k: int) -> torch.Tensor:
    n_rows = a.size(0)
    V = torch.zeros(n_rows, k, dtype=a.dtype, device=a.device)
    V[:, 0] = 1.0
    if k > 1:
        V[:, 1] = a
    for j in range(1, k - 1):
        V[:, j + 1] = ((2 * j + 1) * a * V[:, j] - j * V[:, j - 1]) / (j + 1)

    V = V /  torch.max(torch.norm(V, dim=1, p=2))

    return V

def create_monomial_matrix(a, k):
    n = a.size(0)
    V_mono = torch.zeros(n, k, dtype=a.dtype, device=a.device)
    V_mono[:, 0] = 1.0
    if k > 1:
        V_mono[:, 1] = a
    for j in range(2, k):
        V_mono[:, j] = a * V_mono[:, j - 1]
    
    V_mono = V_mono / torch.max(torch.norm(V_mono, dim=1, p=2))

    return V_mono

