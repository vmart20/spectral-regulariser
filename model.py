import torch.nn as nn


class SpectralModel(nn.Module):
    def __init__(self, nclass, nfeat, hidden_dim=128, feat_dropout=0.0, dropout1=0.0, power=10, V_orth=None):
        super().__init__()
        self.nfeat = nfeat
        self.hidden_dim = hidden_dim
        self.power = power
        self.V_orth = V_orth
        self.nclass = nclass
        self.reg = 0

        self.feature = nn.Sequential(
            nn.Dropout(feat_dropout),
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(feat_dropout),
            nn.Linear(hidden_dim, nclass),
        )
        self.emb_V = nn.Linear(power + 1, hidden_dim, bias=False)

    def forward(self, e, u, ut, x):
        h = self.feature(x)

        utx = ut @ h
        V = self.emb_V(self.V_orth)

        utx_det = utx.detach()
        energy_term = (V + 1).pow(2) * utx_det.pow(2)
        self.reg = energy_term.sum() / utx_det.pow(2).sum().clamp_min(1e-12)

        res = u @ (V * utx + utx)
        res = self.classifier(res)
        return res

    def regularisation(self):
        return self.reg
