import math, torch, torch.nn as nn
from typing import Optional

class SpatialPolynomialLayer(nn.Module):
    """
    Spatial implementation of polynomial spectral filters.
    H_{l+1} = sigma( (sum theta_k P_k(A)) H_l W_l )
    Uses sparse matrix multiplication to avoid eigendecomposition.
    """
    def __init__(self, d_in: int, d_out: int, adj: torch.Tensor,
                 basis: str = "chebyshev", K: int = 3, activation: Optional[nn.Module] = None,
                 dropout: float = 0.0):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.basis, self.K = basis, K
        self.activation = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        self.register_buffer("adj", adj) # Sparse adjacency
        
        self.theta = nn.Parameter(torch.zeros(K+1))
        nn.init.normal_(self.theta, std=0.01)
        self.W = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def _mm(self, A, B):
        if A.is_sparse:
            return torch.sparse.mm(A, B)
        else:
            return torch.mm(A, B)

    def compute_basis_projections(self, H: torch.Tensor) -> list[torch.Tensor]:
        """Returns [P_0(A)H, P_1(A)H, ..., P_K(A)H]"""
        projections = []
        
        if self.basis == "chebyshev":
            T0 = H
            projections.append(T0)
            if self.K >= 1:
                T1 = self._mm(self.adj, H)
                projections.append(T1)
                for k in range(2, self.K + 1):
                    Tk = 2 * self._mm(self.adj, T1) - T0
                    projections.append(Tk)
                    T0, T1 = T1, Tk
                    
        elif self.basis == "monomial":
             Pk = H
             projections.append(Pk)
             for k in range(1, self.K + 1):
                 Pk = self._mm(self.adj, Pk)
                 projections.append(Pk)
                 
        elif self.basis == "legendre":
            P0 = H
            projections.append(P0)
            if self.K >= 1:
                P1 = self._mm(self.adj, H)
                projections.append(P1)
                for k in range(1, self.K):
                    term1 = (2*k + 1) / (k + 1)
                    term2 = k / (k + 1)
                    Pk_plus_1 = term1 * self._mm(self.adj, P1) - term2 * P0
                    projections.append(Pk_plus_1)
                    P0, P1 = P1, Pk_plus_1
                    
        elif self.basis == "bernstein":
            T = [H]
            curr = H
            for _ in range(self.K):
                # t * curr = 0.5 * (A @ curr + curr)
                curr = 0.5 * (self._mm(self.adj, curr) + curr)
                T.append(curr)
            
            for k in range(self.K + 1):
                base_coeff = math.comb(self.K, k)
                vec = base_coeff * T[k]
                
                for j in range(1, self.K - k + 1):
                    coeff = base_coeff * math.comb(self.K - k, j) * ((-1)**j)
                    vec = vec + coeff * T[k+j]
                
                projections.append(vec)
                
        else:
            raise NotImplementedError(f"Basis {self.basis} not implemented")
            
        return projections

    def apply_filter(self, H: torch.Tensor) -> torch.Tensor:
        
        if self.basis == "chebyshev":
            
            T0 = H
            T1 = self._mm(self.adj, H)
            
            out = self.theta[0] * T0 + self.theta[1] * T1
            
            for k in range(2, self.K + 1):
                # T_k = 2 A T_{k-1} - T_{k-2}
                Tk = 2 * self._mm(self.adj, T1) - T0
                out = out + self.theta[k] * Tk
                T0, T1 = T1, Tk
                
            return out
        elif self.basis == "monomial":
             
             Pk = H
             out = self.theta[0] * Pk
             for k in range(1, self.K + 1):
                 Pk = self._mm(self.adj, Pk)
                 out = out + self.theta[k] * Pk
             return out
        elif self.basis == "legendre":
            # P_0 = 1, P_1 = A
            # P_{k+1} = ((2k+1) A P_k - k P_{k-1}) / (k+1)
            
            P0 = H
            if self.K == 0:
                return self.theta[0] * P0
            
            P1 = self._mm(self.adj, H)
            out = self.theta[0] * P0 + self.theta[1] * P1
            
            for k in range(1, self.K):
                # Compute P_{k+1}
                term1 = (2*k + 1) / (k + 1)
                term2 = k / (k + 1)
                Pk_plus_1 = term1 * self._mm(self.adj, P1) - term2 * P0
                out = out + self.theta[k+1] * Pk_plus_1
                P0, P1 = P1, Pk_plus_1
            return out
        elif self.basis == "bernstein":
            w = []
            for p in range(self.K + 1):
                coeff = 0.0
                for k in range(p + 1):
                    term = ((-1)**(p-k)) * math.comb(p, k) * self.theta[k]
                    coeff += term
                w.append(coeff * math.comb(self.K, p))
            
            
            out = w[self.K] * H
            for p in range(self.K - 1, -1, -1):
                # t * out = 0.5 * (A @ out + out)
                t_out = 0.5 * (self._mm(self.adj, out) + out)
                out = w[p] * H + t_out
            
            return out
        else:
            raise NotImplementedError(f"Basis {self.basis} not implemented for spatial layer")
    
    def forward(self, H: torch.Tensor) -> torch.Tensor:           
        H = H @ self.W
        # H = self.dropout(H)
        Z = self.apply_filter(H)
        Z = self.activation(Z)
        return Z

