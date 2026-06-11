"""
Dual-head PINN for the PSA surrogate: shared tanh backbone, a K=3 committee
of deterministic heads, an evidential (NIG) head, and a frozen PCA bottleneck
over the initial-condition block (cols 2..99 -> k=8 scores -> reconstruction,
so autograd still flows through the IC features).

Two unit-handling details matter if you want to reproduce the paper numbers:

* PDE residuals are evaluated in PHYSICAL units. The network is trained on
  min-max normalised outputs, so residuals_ads() denormalises the prediction
  (through the y_lb/y_ub buffers) before any physics formula. Skipping this
  inflates the q2 residual by roughly 5e9 and the PDE loss then actively
  corrupts the data fit.

* The time column is normalised by the true t_max of the data set (124.1 s
  for adsorption), not by the [0,1] clamp used for the IC columns. With the
  clamp, normalised t reaches ~247 and the tanh backbone saturates.
  build_model(t_max=...) applies the override.

float64 throughout.
"""

import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from sklearn.decomposition import PCA

torch.set_default_dtype(torch.float64)


# --- Physical constants -------------------------------------------------------
_PHYS = dict(
    rho_s=937.748616, b01=9.3880523013e-08, b02=2.549e-07,
    deltaUb1=-31135.10206, deltaUb2=-11890.66206,
    d01=5.22911980202e-07, d02=2.549e-07,
    deltaUd1=-31135.10206, deltaUd2=-11890.66206,
    qsat1=3.298453336, qsat2=1.891959934,
    epsilon=0.37, epsilon_p=0.35,
    Dp=1.6e-5/3.0, rp=0.0015/2.0, DL=3.862e-4,
    v0=0.5, L=1.0, R=8.314472, T=298.15,
    qs0=3093.12005077458, P0=101300.0,
    Rp=7.5e-4, mu=1.72e-5,
)


# --- Architecture sizing ------------------------------------------------------
def get_arch(budget: int) -> tuple:
    if budget <= 12:    return 96,  3
    if budget <= 30:    return 128, 3
    if budget <= 100:   return 192, 4
    return 256, 4


# --- Evidential layer (NIG head) ----------------------------------------------
class EvidentialLayer(nn.Module):
    def __init__(self, in_features: int, n_outputs: int = 4):
        super().__init__()
        self.gamma_net = nn.Linear(in_features, n_outputs)
        self.nu_net    = nn.Linear(in_features, n_outputs)
        self.alpha_net = nn.Linear(in_features, n_outputs)
        self.beta_net  = nn.Linear(in_features, n_outputs)
        for net in [self.gamma_net, self.nu_net, self.alpha_net, self.beta_net]:
            nn.init.xavier_normal_(net.weight, gain=0.1)
            nn.init.constant_(net.bias, 0.01)

    def forward(self, x):
        gamma = self.gamma_net(x)
        nu    = torch.nn.functional.softplus(self.nu_net(x))    + 1.0
        alpha = torch.nn.functional.softplus(self.alpha_net(x)) + 1.0
        beta  = torch.nn.functional.softplus(self.beta_net(x))  + 0.1
        return gamma, nu, alpha, beta


def evidential_loss(gamma, nu, alpha, beta, targets, lam: float = 0.01):
    nu    = torch.clamp(nu,    min=1.0,  max=1e4)
    alpha = torch.clamp(alpha, min=1.0,  max=1e4)
    beta  = torch.clamp(beta,  min=0.1,  max=1e6)
    error   = (targets - gamma) ** 2
    log_2pi = np.log(2.0 * np.pi)
    nll = (0.5 * log_2pi
           - 0.5 * torch.log(nu)
           + (alpha + 0.5) * torch.log(beta + nu * error / 2.0)
           - alpha * torch.log(beta))
    reg = lam * (2.0 * nu + alpha)
    return torch.mean(nll + reg)


# --- Building blocks ----------------------------------------------------------
def _make_backbone(n_in: int, hidden_dim: int, n_layers: int):
    layers = []; in_dim = n_in
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        in_dim = hidden_dim
    return nn.Sequential(*layers)


def _xavier_init(module, gain: float = 1.0):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)


class _DetTower(nn.Module):
    def __init__(self, hidden_dim: int, n_out: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_out),
        )
    def forward(self, x): return self.net(x)


# --- PCA bottleneck -----------------------------------------------------------
class _PCABottleneck(nn.Module):
    """Differentiable IC bottleneck: cols 2..99 -> PCA scores -> reconstructed."""
    def __init__(self, components: np.ndarray, mean: np.ndarray):
        super().__init__()
        self.register_buffer('components', torch.tensor(components, dtype=torch.float64))
        self.register_buffer('ic_mean',    torch.tensor(mean,       dtype=torch.float64))
        self.k = components.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ic   = x[:, 2:100] - self.ic_mean
        proj = ic @ self.components.T
        rec  = proj @ self.components + self.ic_mean
        return torch.cat([x[:, :2], rec, x[:, 100:]], dim=1)


# --- PSADualHeadPINN ----------------------------------------------------
class PSADualHeadPINN(nn.Module):
    """Dual-head PSA PINN: PCA bottleneck + K committee heads + NIG head.

    y_lb / y_ub are registered as buffers so they follow the model across
    devices; residuals_ads() uses them to denormalise before the physics.
    """
    def __init__(self, hidden_dim: int, n_layers: int,
                 low_bound: torch.Tensor, up_bound: torch.Tensor,
                 pca_components: np.ndarray = None,
                 pca_mean:        np.ndarray = None,
                 n_committee:     int = 3,
                 init_seed:       int = 0,
                 y_lb:            torch.Tensor = None,
                 y_ub:            torch.Tensor = None):
        super().__init__()
        self.low_bound = low_bound.double()
        self.up_bound  = up_bound.double()
        self.n_in      = 102
        self.n_out     = 4
        self.hidden_dim = hidden_dim
        self.n_committee = n_committee

        # y bounds ride along as buffers (auto .to(device)); PDE denorm needs them
        # Defaults: identity transform (no-op) so old code that doesn't pass
        # y_lb/y_ub still works correctly (u_phys = u * 1 + 0 = u).
        _y_lb = y_lb.double().flatten() if y_lb is not None \
                else torch.zeros(4, dtype=torch.float64)
        _y_ub = y_ub.double().flatten() if y_ub is not None \
                else torch.ones(4, dtype=torch.float64)
        self.register_buffer('y_lb_pde', _y_lb)
        self.register_buffer('y_ub_pde', _y_ub)

        # -- PCA bottleneck ---------------------------------------------------
        if pca_components is not None:
            self.pca = _PCABottleneck(pca_components, pca_mean)
            self.use_pca = True
        else:
            self.pca = None
            self.use_pca = False

        # -- Backbone + heads -------------------------------------------------
        self.backbone = _make_backbone(self.n_in, hidden_dim, n_layers)

        self.committee_heads = nn.ModuleList()
        for k in range(n_committee):
            tower = _DetTower(hidden_dim, self.n_out)
            torch.manual_seed(init_seed * 1000 + k * 17 + 7)
            _xavier_init(tower, gain=1.0)
            self.committee_heads.append(tower)
        self.det_head = self.committee_heads[0]

        self.evid_head = EvidentialLayer(hidden_dim, self.n_out)

        torch.manual_seed(init_seed)
        _xavier_init(self.backbone, gain=1.0)

        # Physical constants
        for k, v in _PHYS.items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float64))

        self.register_buffer('residual_scales', torch.ones(4, dtype=torch.float64))

    # -- Internal helpers ------------------------------------------------------
    def _bottleneck(self, x): return self.pca(x) if self.use_pca else x

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x.double() - self.low_bound) / (self.up_bound - self.low_bound) - 1.0

    def features(self, x): return self.backbone(self._normalise(self._bottleneck(x)))

    # -- Forward ---------------------------------------------------------------
    def forward_deterministic(self, x): return self.committee_heads[0](self.features(x))
    def forward(self, x):               return self.forward_deterministic(x)

    def forward_committee(self, x):
        h = self.features(x)
        return torch.stack([head(h) for head in self.committee_heads], dim=0)

    def committee_disagreement(self, x):
        was_train = self.training; self.eval()
        with torch.no_grad():
            preds = self.forward_committee(x)
            score = preds.std(dim=0, unbiased=False).sum(dim=1)
        if was_train: self.train()
        return score.detach()

    def forward_evidential(self, x): return self.evid_head(self.features(x))

    def epistemic_uncertainty(self, x):
        self.eval()
        with torch.no_grad():
            _, nu, alpha, beta = self.forward_evidential(x)
        return (beta / (alpha * nu + 1e-8)).sum(dim=1)

    # -- Losses ----------------------------------------------------------------
    def loss_evidential(self, x, y, lam_evid: float = 0.01):
        gamma, nu, alpha, beta = self.forward_evidential(x)
        # y is already normalised - correct, evidential head trains on norm space
        return evidential_loss(gamma, nu, alpha, beta, y.double(), lam=lam_evid)

    def loss_deterministic_mse(self, x, y):
        h = self.features(x)
        loss = 0.0
        for head in self.committee_heads:
            loss = loss + nn.functional.mse_loss(head(h), y.double())
        return loss / max(1, len(self.committee_heads))

    # --- PDE residuals ---------------------------------------------------
    def residuals_ads(self, x_pde: torch.Tensor):
        """PDE residuals of the four governing equations, in physical units.

        The normalised output u is converted through y_lb_pde / y_ub_pde
        before any physics formula. The derivatives are then taken on
        u_phys; the denorm is a linear map, so autograd handles the chain
        rule. Columns 100/101 of the input are ignored (q2_IC tail values,
        see module docstring) and the nominal eps / qsat1 are used instead.
        """
        g = x_pde.double().clone().requires_grad_(True)
        u = self.forward_deterministic(g)                     # (N,4) normalised

        # denormalise to physical units
        y_range  = (self.y_ub_pde - self.y_lb_pde)           # (4,)
        u_phys   = u * y_range + self.y_lb_pde               # (N,4) physical

        # -- Physical constants (eps/qsat override for corrupt cols 100,101) --
        eps_col  = torch.full((g.shape[0], 1), 0.37,
                              dtype=g.dtype, device=g.device)
        qsat_col = torch.full((g.shape[0], 1), 3.298453336,
                              dtype=g.dtype, device=g.device)
        coeff0 = ((4.0/150.0) * self.Rp**2 * eps_col**2 * self.P0
                  / ((1.0 - eps_col)**2 * self.mu * self.v0 * self.L))
        coeff3 = ((self.R * self.T * self.qs0 / self.P0)
                  * (1.0 - eps_col) / eps_col)

        # -- Autograd derivatives - on u_phys so units are correct ------------
        ones = torch.ones(g.shape[0], 1, dtype=torch.float64, device=g.device)
        dy1 = autograd.grad(u_phys[:, 0:1], g, ones, retain_graph=True, create_graph=True)[0]
        dP  = autograd.grad(u_phys[:, 1:2], g, ones, retain_graph=True, create_graph=True)[0]
        dq1 = autograd.grad(u_phys[:, 2:3], g, ones, retain_graph=True, create_graph=True)[0]
        dq2 = autograd.grad(u_phys[:, 3:4], g, ones, retain_graph=True, create_graph=True)[0]

        y1_z = dy1[:, 0:1]; y1_t = dy1[:, 1:2]
        P_z  = dP[:,  0:1]; P_t  = dP[:,  1:2]
        q1_t = dq1[:, 1:2]; q2_t = dq2[:, 1:2]

        ve = -coeff0 * P_z

        # Second-order terms - also on u_phys
        Py1z_z  = autograd.grad(u_phys[:, 1:2] * y1_z, g, ones,
                                retain_graph=True, create_graph=True)[0][:, 0:1]
        y1Pve_z = autograd.grad(u_phys[:, 0:1] * u_phys[:, 1:2] * ve, g, ones,
                                retain_graph=True, create_graph=True)[0][:, 0:1]
        Pve_z   = autograd.grad(u_phys[:, 1:2] * ve, g, ones,
                                retain_graph=True, create_graph=True)[0][:, 0:1]

        b1 = self.b01 * torch.exp(-self.deltaUb1 / (self.R * self.T))
        b2 = self.b02 * torch.exp(-self.deltaUb2 / (self.R * self.T))
        d1 = self.d01 * torch.exp(-self.deltaUd1 / (self.R * self.T))
        d2 = self.d02 * torch.exp(-self.deltaUd2 / (self.R * self.T))

        # Use physical P and y1 for isotherm / LDF computation
        P_bar = u_phys[:, 1:2].clamp(min=1e-3)               # physical P (dimensionless P/P0)
        y1    = u_phys[:, 0:1].clamp(0.0, 1.5)               # physical y1 (mole fraction)

        conc  = P_bar * self.P0 / (self.R * self.T)
        beta1 = conc * b1; beta2 = conc * b2
        deta1 = conc * d1; deta2 = conc * d2
        denom_b = 1.0 + beta1 * y1 + beta2 * (1.0 - y1)
        denom_d = 1.0 + deta1 * y1 + deta2 * (1.0 - y1)

        q1s = ((qsat_col * self.rho_s / self.qs0) * beta1 * y1 / denom_b
               + (self.qsat2 * self.rho_s / self.qs0) * deta1 * y1 / denom_d)
        q2s = ((qsat_col * self.rho_s / self.qs0) * beta2 * (1.0 - y1) / denom_b
               + (self.qsat2 * self.rho_s / self.qs0) * deta2 * (1.0 - y1) / denom_d)

        QC1 = ((qsat_col * self.rho_s * b1 / denom_b)
               + (self.qsat2 * self.rho_s * d1 / denom_d))
        QC2 = ((qsat_col * self.rho_s * b2 / denom_b)
               + (self.qsat2 * self.rho_s * d2 / denom_d))

        scale = 15.0 * self.epsilon_p * self.Dp / self.rp**2 * (self.L / self.v0)
        k1 = scale / (QC1 * P_bar)
        k2 = scale / (QC2 * P_bar)

        # Physical q1, q2 from model
        q1_phys = u_phys[:, 2:3]
        q2_phys = u_phys[:, 3:4]

        f_y  = (y1_t
                - (self.DL / (self.v0 * self.L)) * (Py1z_z / P_bar)
                + (y1Pve_z / P_bar)
                + coeff3 * q1_t
                + y1 * P_t / P_bar)
        f_P  = P_t + Pve_z + coeff3 * q1_t + coeff3 * q2_t
        f_q1 = q1_t - k1 * (q1s - q1_phys)
        f_q2 = q2_t - k2 * (q2s - q2_phys)
        return f_y, f_P, f_q1, f_q2

    # -- Residual scale update + losses (unchanged) ----------------------------
    @torch.no_grad()
    def _residual_rms(self, x_pde):
        was_train = self.training; self.train()
        with torch.enable_grad():
            f_y, f_P, f_q1, f_q2 = self.residuals_ads(x_pde)
            rms = torch.tensor([
                float((f_y **2).mean().sqrt().item()),
                float((f_P **2).mean().sqrt().item()),
                float((f_q1**2).mean().sqrt().item()),
                float((f_q2**2).mean().sqrt().item()),
            ], dtype=torch.float64, device=x_pde.device)
        if not was_train: self.eval()
        return rms.clamp_min(1e-6)

    def update_residual_scales(self, x_pde, momentum=0.9):
        new = self._residual_rms(x_pde)
        if momentum == 0.0:
            self.residual_scales = new.clone()
        else:
            self.residual_scales = (momentum * self.residual_scales
                                    + (1.0 - momentum) * new)
        self.residual_scales = self.residual_scales.clamp_min(1e-6)

    def loss_pde(self, x_pde):
        f_y, f_P, f_q1, f_q2 = self.residuals_ads(x_pde)
        s = self.residual_scales
        z = torch.zeros_like(f_y)
        mse = nn.functional.mse_loss
        return (mse(f_y/s[0], z) + mse(f_P/s[1], z)
              + mse(f_q1/s[2], z) + mse(f_q2/s[3], z))

    def residual_per_point(self, x_pde):
        was_train = self.training; self.train()
        with torch.enable_grad():
            f_y, f_P, f_q1, f_q2 = self.residuals_ads(x_pde)
            s = self.residual_scales
            score = ((f_y/s[0]).abs() + (f_P/s[1]).abs()
                   + (f_q1/s[2]).abs() + (f_q2/s[3]).abs()).mean(dim=1)
        if not was_train: self.eval()
        return score.detach()

    def last_layer_embedding(self, x):
        self.eval()
        with torch.no_grad():
            h    = self.features(x)
            mid  = torch.tanh(self.committee_heads[0].net[0](h))
            pred = self.committee_heads[0].net[2](mid)
            pseudo = pred - pred.mean(dim=0, keepdim=True)
            emb = pseudo.unsqueeze(2) * mid.unsqueeze(1)
            emb = emb.reshape(emb.shape[0], -1)
        return emb.detach()


# --- Vanilla ANN (unchanged except y_lb/y_ub stubs for interface compat) ------
class PSAVanillaANN(nn.Module):
    def __init__(self, hidden_dim, n_layers, low_bound, up_bound,
                 pca_components=None, pca_mean=None, init_seed=0,
                 y_lb=None, y_ub=None):   # y_lb/y_ub accepted but not used (no PDE)
        super().__init__()
        self.low_bound = low_bound.double()
        self.up_bound  = up_bound.double()
        self.n_in = 102; self.n_out = 4; self.hidden_dim = hidden_dim

        if pca_components is not None:
            self.pca = _PCABottleneck(pca_components, pca_mean); self.use_pca = True
        else:
            self.pca = None; self.use_pca = False

        self.backbone = _make_backbone(self.n_in, hidden_dim, n_layers)
        self.det_head = _DetTower(hidden_dim, self.n_out)
        torch.manual_seed(init_seed)
        _xavier_init(self.backbone, gain=1.0)
        _xavier_init(self.det_head, gain=1.0)

    def _bottleneck(self, x): return self.pca(x) if self.use_pca else x
    def _normalise(self, x):
        return 2.0*(x.double()-self.low_bound)/(self.up_bound-self.low_bound)-1.0
    def features(self, x): return self.backbone(self._normalise(self._bottleneck(x)))
    def forward_deterministic(self, x): return self.det_head(self.features(x))
    def forward(self, x): return self.forward_deterministic(x)
    def loss_mse(self, x, y):
        return nn.functional.mse_loss(self.forward_deterministic(x), y.double())
    def epistemic_uncertainty(self, x):
        return torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    def committee_disagreement(self, x):
        return torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    def last_layer_embedding(self, x):
        self.eval()
        with torch.no_grad():
            h = self.features(x)
            mid = torch.tanh(self.det_head.net[0](h))
            pred = self.det_head.net[2](mid)
            pseudo = pred - pred.mean(dim=0, keepdim=True)
            emb = pseudo.unsqueeze(2) * mid.unsqueeze(1)
            emb = emb.reshape(emb.shape[0], -1)
        return emb.detach()


# --- PCA fitting helper -------------------------------------------------------
def fit_pca_for_bottleneck(x_all, all_scenarios, k=8):
    x_np = x_all.double().numpy()
    ic_vecs = np.stack([x_np[int(s['indices'][0]), 2:100] for s in all_scenarios])
    pca = PCA(n_components=k)
    pca.fit(ic_vecs)
    print(f'  PCA bottleneck k={k}: explains {100*pca.explained_variance_ratio_.sum():.1f}% IC variance')
    return pca.components_.astype(np.float64), pca.mean_.astype(np.float64)


# --- model factory -----------------------------------------------------
def build_model(budget, low_bound_np, up_bound_np, device='cpu',
                model_type='pinn', pca_components=None, pca_mean=None,
                n_committee=3, init_seed=0,
                y_lb=None, y_ub=None,
                t_max=None) -> nn.Module:
    """Build a PSA model.

    y_lb, y_ub : (1,4) or (4,) output normalisation bounds from
                 compute_y_bounds(y_all); used inside residuals_ads() to
                 convert the normalised prediction to physical units.
    t_max      : maximum physical t in the data set (seconds). Overrides
                 up_bound[:,1] so the t column normalises correctly.
    """
    h, L = get_arch(budget)
    lb = torch.tensor(low_bound_np, dtype=torch.float64).to(device)
    ub = torch.tensor(up_bound_np,  dtype=torch.float64).to(device)

    # override the t upper bound with the actual data max
    if t_max is not None:
        ub = ub.clone()
        ub[:, 1] = float(t_max)
        print(f'  t_max override: ub[:,1] = {t_max:.2f}  '
              f'(was {float(torch.tensor(up_bound_np)[:, 1].squeeze()):.2f})')

    bn_str = f'  +PCA bottleneck k={pca_components.shape[0]}' \
             if pca_components is not None else '  no bottleneck'

    if model_type == 'pinn':
        m = PSADualHeadPINN(h, L, lb, ub,
                            pca_components=pca_components, pca_mean=pca_mean,
                            n_committee=n_committee, init_seed=init_seed,
                            y_lb=y_lb, y_ub=y_ub).to(device)
        print(f'  Model: PINN  h={h}  L={L}  K={n_committee}{bn_str}')
    elif model_type == 'ann':
        m = PSAVanillaANN(h, L, lb, ub,
                          pca_components=pca_components, pca_mean=pca_mean,
                          init_seed=init_seed).to(device)
        print(f'  Model: ANN   h={h}  L={L}{bn_str}')
    else:
        raise ValueError(f'Unknown model_type {model_type!r}')
    return m


# --- Smoke test ---------------------------------------------------------------
if __name__ == '__main__':
    print('smoke test: psa_pinn_model_v3')
    rng = np.random.default_rng(0)
    comps = rng.standard_normal((8, 98)) * 0.1
    mean  = rng.standard_normal(98) * 0.5
    lb = np.zeros((1, 102)); ub = np.ones((1, 102))

    # Simulate proper y bounds (as returned by compute_y_bounds for ads step)
    y_lb_t = torch.tensor([[0.0, 1.0, 0.0, 0.001]], dtype=torch.float64)
    y_ub_t = torch.tensor([[0.2, 1.02, 0.5, 0.002]], dtype=torch.float64)

    m = build_model(20, lb, ub, 'cpu', model_type='pinn',
                    pca_components=comps, pca_mean=mean,
                    n_committee=3, init_seed=42,
                    y_lb=y_lb_t, y_ub=y_ub_t, t_max=125.0)
    a = build_model(20, lb, ub, 'cpu', model_type='ann',
                    pca_components=comps, pca_mean=mean, init_seed=42)

    x = torch.rand(16, 102, dtype=torch.float64)
    x[:, 1] = x[:, 1] * 125.0   # t in physical seconds
    x[:, -2] = 0.37; x[:, -1] = 3.3
    y = torch.rand(16, 4, dtype=torch.float64)

    print('  PINN forward:        ', m.forward_deterministic(x).shape)
    print('  PINN committee:      ', m.forward_committee(x).shape)
    print('  PINN evidential[0]:  ', m.forward_evidential(x)[0].shape)
    print('  PINN disagreement:   ', m.committee_disagreement(x).shape)
    m.update_residual_scales(x, momentum=0.0)
    print('  residual_scales:     ', m.residual_scales.tolist())
    print('  PINN loss_pde:       ', m.loss_pde(x).item())
    print('  PINN loss_evid:      ', m.loss_evidential(x, y).item())
    print('  PINN loss_det_mse:   ', m.loss_deterministic_mse(x, y).item())
    print('  PINN BADGE emb:      ', m.last_layer_embedding(x).shape)
    print('  ANN forward:         ', a.forward_deterministic(x).shape)
    print('  ANN loss_mse:        ', a.loss_mse(x, y).item())

    # Verify y_lb_pde buffer moved correctly
    assert m.y_lb_pde.shape == torch.Size([4])
    print('  y_lb_pde buffer:     ', m.y_lb_pde.tolist())
    print('  y_ub_pde buffer:     ', m.y_ub_pde.tolist())
    print('SMOKE TEST PASSED.')
