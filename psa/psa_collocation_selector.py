"""
Manifold-aware collocation generation for the PSA PINN.

Plain LHS over all 102 input dimensions treats the 98 IC columns as
independent uniforms and produces jagged, non-physical IC profiles: on the
adsorption data the variance of consecutive z-differences is ~700x larger
for LHS samples than for real profiles. PDE residuals evaluated on such
profiles are large for reasons unrelated to model error, and the physics
loss ends up pulling the model toward inputs it will never see at test
time. So collocation candidates are sampled from the data manifold instead:

  1. base IC = convex combination of n_mix labeled IC profiles (convex
     combinations of smooth physical profiles stay smooth and physical),
  2. (z, t) sampled uniformly on [0,1]^2, independent of the data grid, so
     the PDE loss extends supervision beyond the observed time/space points,
  3. small Gaussian noise on the IC (sigma = 0.5% of per-column std).

Candidates are then scored by physics residual + epistemic uncertainty +
input-gradient norm, with the same weights as the CSTR collocation selector.
"""

import torch
import numpy as np
from scipy.stats import qmc


_N_FEATURES = 102
_IDX_Z      = 0
_IDX_T      = 1
_IC_SLICE   = slice(2, 100)   # 98 IC columns
_IDX_P_END_1 = 100
_IDX_P_END_2 = 101


# --- Manifold-aware candidate generator ---------------------------------------

def manifold_collocation_candidates(
        n_candidates: int,
        labeled_x:    torch.Tensor,
        rng_seed:     int = 42,
        n_mix:        int = 3,
        ic_noise_frac: float = 0.005,
) -> torch.Tensor:
    """
    Generate `n_candidates` collocation points whose IC profiles are convex
    combinations of `n_mix` randomly chosen labeled IC profiles.

    Parameters
    ----------
    n_candidates : how many collocation points to return
    labeled_x    : (N, 102) labeled rows.  ANY n_mix of them combined gives a
                   physically-valid IC profile (smooth, mass-balanced).
    rng_seed     : for reproducibility
    n_mix        : how many labeled profiles to mix per candidate (3 is a
                   good balance - too few = no novelty, too many = bias to
                   the average profile).
    ic_noise_frac: small additive Gaussian noise on IC, sigma = frac x profile std.
                   Keeps the residual gradient sensitive but doesn't break
                   physical realism.

    Returns
    -------
    (n_candidates, 102) float64 tensor:
        col 0       : z   ~ U[0, 1]   (any spatial point, not just grid)
        col 1       : t   ~ U[0, 1]   (any time, not just data times)
        cols 2..99  : convex combination of n_mix labeled IC profiles + noise
        col 100     : eps     ~ labeled-set range
        col 101     : qsat1   ~ labeled-set range
    """
    rng = np.random.default_rng(rng_seed)
    x_np = labeled_x.detach().cpu().double().numpy()
    N_lab = x_np.shape[0]
    if N_lab < n_mix:
        n_mix = max(1, N_lab)

    # Identify unique IC profiles in the labeled set (one per scenario,
    # not per row - many rows share the same IC).
    # Round to dedupe; cols 2..12 are a near-unique fingerprint.
    fp = np.round(x_np[:, 2:12], 5)
    _, unique_idx = np.unique(fp, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    unique_ics = x_np[unique_idx]                    # (N_unique, 102) - first row per scenario
    N_uniq = unique_ics.shape[0]
    n_mix  = min(n_mix, N_uniq)

    # IC noise std per column (from labeled distribution)
    ic_block = x_np[:, 2:100]
    ic_std   = ic_block.std(axis=0) + 1e-8

    # cols 100/101 of the training data hold q2_IC tail values, not eps/qsat
    # (see model module docstring); write the nominal constants the residual
    # computation expects
    EPS_NOMINAL  = 0.37
    QSAT_NOMINAL = 3.298453336

    cands = np.empty((n_candidates, _N_FEATURES), dtype=np.float64)

    # Choose n_mix scenarios per candidate, with random Dirichlet weights
    # Dirichlet(alpha=1) -> uniform on simplex -> unbiased convex combinations
    for i in range(n_candidates):
        idx_pick = rng.choice(N_uniq, size=n_mix, replace=False)
        weights  = rng.dirichlet(np.ones(n_mix))     # (n_mix,) sums to 1
        ic_mix   = (unique_ics[idx_pick, 2:100] * weights[:, None]).sum(axis=0)
        # Add small Gaussian noise (per-column)
        ic_mix  += rng.standard_normal(98) * ic_std * ic_noise_frac
        # Clip y1 columns (cols 2..27 of original = cols 0..25 of ic block) to [0,1]
        ic_mix[0:25]  = np.clip(ic_mix[0:25], 0.0, 1.0)
        # Clip P columns (cols 77..102 of original = cols 75..98 of ic block) to [0,1]
        # Only first 23 P columns live in cols 2..99 (last 2 are at 100,101)
        ic_mix[75:98] = np.clip(ic_mix[75:98], 0.0, 1.0)

        cands[i, 0]      = rng.random()                                  # z
        cands[i, 1]      = rng.random()                                  # t
        cands[i, 2:100]  = ic_mix
        cands[i, 100]    = EPS_NOMINAL                                   # was junk q2_IC
        cands[i, 101]    = QSAT_NOMINAL                                  # was junk q2_IC

    return torch.tensor(cands, dtype=torch.float64)


# --- Real-data collocation (rounds 0-1 only, before model has signal) ---------

def real_data_collocation(labeled_x: torch.Tensor, n_max: int = 2000) -> torch.Tensor:
    """For the very first round when the model is essentially random, use the
    LABELED rows themselves as collocation.  This forces physics enforcement
    on the points the model is being trained on - same data, two losses (data
    fit + physics consistency).  No risk of out-of-distribution synthetic data.
    """
    N = labeled_x.shape[0]
    if N <= n_max:
        return labeled_x.detach().clone()
    idx = torch.randperm(N)[:n_max]
    return labeled_x[idx].detach().clone()


# --- Scoring (same as v1/v2 but on physically valid candidates) ---------------

def _normalise(v: torch.Tensor) -> torch.Tensor:
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-14:
        return torch.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def score_physics_residual(model, candidates: torch.Tensor, device: str) -> torch.Tensor:
    return model.residual_per_point(candidates.to(device)).cpu()


def score_epistemic_uncertainty(model, candidates: torch.Tensor, device: str) -> torch.Tensor:
    return model.epistemic_uncertainty(candidates.to(device)).cpu()


def score_gradient_of_residual(model, candidates: torch.Tensor, device: str,
                                batch_size: int = 128) -> torch.Tensor:
    """||d(sum_j u_j(x))/dx||2 per candidate.  Original-space gradient."""
    model.train()
    grads = []
    n = candidates.shape[0]
    for start in range(0, n, batch_size):
        x_b = candidates[start:start+batch_size].to(device).double().detach().clone()
        x_b.requires_grad_(True)
        with torch.enable_grad():
            u = model.forward_deterministic(x_b)
            u.sum().backward()
        if x_b.grad is not None:
            grads.append(x_b.grad.detach().norm(dim=1).cpu())
        else:
            grads.append(torch.zeros(x_b.shape[0], dtype=torch.float64))
        x_b.grad = None
    model.eval()
    return torch.cat(grads, dim=0)


# --- Main selector ------------------------------------------------------------

class PSACollocationSelectorV3:
    """Collocation selection with manifold-aware candidates.

    Round 0 returns real labeled rows (the model has no PDE signal yet, so
    synthetic candidates would only mislead). Later rounds generate manifold
    candidates, mix in real rows 50/50, and keep the top-K by the combined
    physics/uncertainty/gradient score (weights 0.50/0.20/0.30, as in the
    CSTR collocation selector).
    """
    def __init__(self,
                 w_physics:       float = 0.50,
                 w_uncertainty:   float = 0.20,
                 w_gradient:      float = 0.30,
                 n_candidates:    int   = 2000,
                 coll_multiplier: int   = 4,
                 device:          str   = 'cpu',
                 use_manifold:    bool  = True,
                 n_mix:           int   = 3,
                 ic_noise_frac:   float = 0.005):
        self.w_physics      = w_physics
        self.w_uncertainty  = w_uncertainty
        self.w_gradient     = w_gradient
        self.n_candidates   = n_candidates
        self.coll_multiplier = coll_multiplier
        self.device         = device
        self.use_manifold   = use_manifold
        self.n_mix          = n_mix
        self.ic_noise_frac  = ic_noise_frac

    def select(self, model, labeled_x: torch.Tensor,
                rng_seed: int = 42, round_idx: int = 0) -> torch.Tensor:
        """
        Returns (K, 102) float64 collocation tensor.

        Round 0:  use real labeled rows directly (no synthetic) - model has no
                  PDE signal yet, so synthetic data risks misleading gradients.
        Round 1+: use manifold candidates + scoring.
        """
        K = min(self.coll_multiplier * len(labeled_x), self.n_candidates)
        K = max(K, 32)

        # -- Round 0: use real labeled rows directly --------------------------
        if round_idx == 0:
            return real_data_collocation(labeled_x, n_max=K).to(self.device)

        # round 1+: manifold-based generation + scoring. use_manifold=True is
        # the only supported mode (the old LHS fallback is gone).
        cands = manifold_collocation_candidates(
            self.n_candidates, labeled_x,
            rng_seed=rng_seed, n_mix=self.n_mix,
            ic_noise_frac=self.ic_noise_frac).to(self.device)

        # Mix in some real data for robustness (50/50 manifold/real)
        # This guarantees we always have at least some clean residuals.
        n_real = min(self.n_candidates // 2, len(labeled_x))
        if n_real > 0:
            real_pts = real_data_collocation(labeled_x, n_max=n_real).to(self.device)
            # Replace last n_real candidates with real data
            cands = torch.cat([cands[:-n_real], real_pts], dim=0)

        s_phys = _normalise(score_physics_residual    (model, cands, self.device))
        s_unc  = _normalise(score_epistemic_uncertainty(model, cands, self.device))
        s_grad = _normalise(score_gradient_of_residual(model, cands, self.device))

        score = (self.w_physics      * s_phys
               + self.w_uncertainty  * s_unc
               + self.w_gradient     * s_grad)

        top_idx = score.topk(min(K, len(cands))).indices
        return cands[top_idx]


# --- Smoke test ---------------------------------------------------------------
if __name__ == '__main__':
    print('smoke test: manifold collocation generator')
    rng = np.random.default_rng(42)

    # Simulate 4 scenarios x 75 rows (smooth IC profiles)
    scen_ics = []
    for s in range(4):
        z = np.linspace(0, 1, 25)
        # Smooth profile + offset
        y1 = 0.3 + 0.4 * np.exp(-((z - 0.3 - s*0.1)**2)/0.05)
        q1 = 0.2 + 0.5 * y1 / (1 + 2*y1)
        q2 = 0.1 + 0.3 * (1-y1) / (1 + 0.5*(1-y1))
        P  = 1.0 - 0.05 * z + s*0.01
        ic_full = np.concatenate([y1, q1, q2, P[:23]])     # 98-dim IC
        for ti in range(3):
            for zi in range(25):
                row = np.zeros(102)
                row[0] = z[zi]; row[1] = ti / 2.0
                row[2:100] = ic_full
                row[100]   = 0.37; row[101] = 3.3
                scen_ics.append(row)

    labeled_x = torch.tensor(np.array(scen_ics), dtype=torch.float64)
    print(f'  Labeled rows: {labeled_x.shape[0]}')

    cands = manifold_collocation_candidates(500, labeled_x, rng_seed=0, n_mix=3)
    print(f'  Candidates shape: {cands.shape}')

    # Smoothness check
    real_diff_var = float(torch.var(torch.diff(labeled_x[0, 2:27])))
    cand_diff_var = float(torch.var(torch.diff(cands[0, 2:27])))
    print(f'  Real IC profile diff variance:    {real_diff_var:.4e}')
    print(f'  Manifold cand diff variance:      {cand_diff_var:.4e}')
    print(f'  Ratio: {cand_diff_var / max(real_diff_var, 1e-12):.2f}x (should be ~1, not ~700x)')

    # Check (z,t) span
    print(f'  z range: [{float(cands[:,0].min()):.3f}, {float(cands[:,0].max()):.3f}]  (expect ~[0,1])')
    print(f'  t range: [{float(cands[:,1].min()):.3f}, {float(cands[:,1].max()):.3f}]  (expect ~[0,1])')

    print('SMOKE TEST PASSED.')
