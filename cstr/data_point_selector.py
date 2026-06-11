"""
DATA POINT ACTIVE LEARNING SELECTOR  (v3 - physics-guided gradient scoring)
=============================================================================
Scoring strategy redesigned for smooth steady-state (SS) regression:
 
  NEW: physics-guided gradient + uncertainty hybrid
    1. Output gradient magnitude  |grad _x gamma(x)| - find where CSTR response
       changes fastest w.r.t. operating conditions.  Works at small N
       because gradients are directional even for an undertrained model.
       
    2. Physics residual at candidate point - find where model's SS prediction
       violates the algebraic CSTR constraints.  These points need labeling
       to enforce physical consistency. This is the part that differs
       from common practice: physics guides the DATA selection here,
       not just the collocation points.
       
    3. Space-filling diversity - min-distance to existing labeled points,
       vectorized GPU computation.

  Epistemic uncertainty is REMOVED from data AL scoring.  For smooth SS
  data it is an unreliable proxy that hurts small-budget performance.
  Uncertainty is still used in COLLOCATION AL (where it belongs).

Score = w_grad * |grad _x gamma(x)| + w_physics * |f_SS(x, gamma(x))| + w_div * diversity

Interface compatible with v2 - drop-in replacement.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class DataPointSelector:
    """
    Physics-guided active learning selector for labeled data points.
    Designed for smooth steady-state regression problems.
    """

    def __init__(self, model, device,
                 w_gradient=0.4, w_physics=0.4, w_diversity=0.2,
                 y_norm_bounds=None):
        """
        Args:
            model          : DualHeadPhysicsPINN
            device         : torch device
            w_gradient     : weight for output gradient magnitude score
            w_physics      : weight for SS physics residual score
            w_diversity    : weight for space-filling diversity
            y_norm_bounds  : needed for physics residual (passed from trainer)
        """
        self.model         = model
        self.device        = device
        self.w_gradient    = w_gradient
        self.w_physics     = w_physics
        self.w_diversity   = w_diversity
        self.y_norm_bounds = y_norm_bounds
        self.labeled_points = None
        self._call_count   = 0

        print(f"[DataPointSelector v3]  "
              f"grad={w_gradient:.2f}  phys={w_physics:.2f}  div={w_diversity:.2f}")

    def update_labeled_points(self, x_labeled):
        self.labeled_points = x_labeled.clone().detach()
        self._call_count += 1

    # -- Score 1: output gradient magnitude ------------------------------------
    def compute_gradient_score(self, x_candidates, batch_size=256):
        """
        |grad _x gamma(x)| averaged over the 4 outputs.
        Points where the predicted mean changes fastest -> most informative
        for interpolation in a smooth function.
        Requires grad on x but no physics knowledge - works at small N.
        """
        self.model.eval()
        scores = []
        for i in range(0, len(x_candidates), batch_size):
            batch = x_candidates[i:i+batch_size].clone().detach().requires_grad_(True)
            gamma, nu, alpha, beta = self.model.forward_evidential(batch)
            # Sum all outputs to get scalar; grad gives d(sum_j gamma_j)/dx_i per point
            total = gamma.sum()
            grads = torch.autograd.grad(total, batch,
                                        create_graph=False, retain_graph=False)[0]
            # L2 norm over input dims -> per-point score
            scores.append(torch.norm(grads, dim=1).detach())
        return torch.cat(scores, dim=0)

    # -- Score 2: SS physics residual at candidate points ----------------------
    def compute_physics_residual_score(self, x_candidates, batch_size=512):
        """
        For each unlabeled candidate x, evaluate the SS CSTR residuals using
        the model's current prediction gamma(x). 
        
        |f_SS(x, gamma(x))| large -> model violates physics here -> label this point
        to give the model a physically-correct anchor in this operating region.
        
        Physics guiding the data selection (not just collocation) is the
        point of this selector.
        
        Returns zero scores if y_norm_bounds not set (graceful degradation).
        """
        if self.y_norm_bounds is None:
            return torch.zeros(len(x_candidates), device=self.device, dtype=torch.float64)

        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(x_candidates), batch_size):
                batch = x_candidates[i:i+batch_size]
                residuals = self.model.compute_pde_residuals(batch, self.y_norm_bounds)
                res_score = torch.zeros(batch.shape[0], device=self.device, dtype=torch.float64)
                valid = 0
                for r in residuals:
                    if r is not None:
                        res_score = res_score + r.detach() ** 2
                        valid += 1
                if valid > 0:
                    res_score = res_score / valid
                scores.append(res_score)
        return torch.cat(scores, dim=0)

    # -- Score 3: space-filling diversity --------------------------------------
    def compute_diversity_score(self, x_candidates):
        """
        Vectorised min-distance to existing labeled points.
        ||a-b||^2 = ||a||^2 + ||b||^2 - 2a*b  (no Python for-loop).
        """
        if self.labeled_points is None or len(self.labeled_points) == 0:
            return torch.ones(len(x_candidates), device=self.device, dtype=torch.float64)

        cand = x_candidates.to(dtype=torch.float32, device=self.device)
        lab  = self.labeled_points.to(dtype=torch.float32, device=self.device)

        cand_sq = (cand**2).sum(1, keepdim=True)
        lab_sq  = (lab **2).sum(1, keepdim=True)
        cross   = cand @ lab.T
        dist2   = torch.clamp(cand_sq + lab_sq.T - 2*cross, min=0.0)
        return torch.sqrt(dist2.min(1)[0]).to(dtype=torch.float64)

    # -- Combine ---------------------------------------------------------------
    def _normalize(self, s):
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-10:
            return torch.ones_like(s) * 0.5
        return (s - lo) / (hi - lo)

    def combine_scores(self, grad_s, phys_s, div_s):
        return (self.w_gradient * self._normalize(grad_s) +
                self.w_physics  * self._normalize(phys_s) +
                self.w_diversity* self._normalize(div_s))

    # -- Cluster-then-select ---------------------------------------------------
    def _find_k(self, features, k_range=(3, 10)):
        n = min(len(features), 500)
        if n < k_range[0] * 4:
            return max(2, k_range[0])
        idx = np.random.choice(len(features), n, replace=False)
        fs  = features[idx]
        best_k, best_s = k_range[0], -1.0
        for k in range(k_range[0], min(k_range[1]+1, n//2)):
            try:
                lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(fs)
                if len(np.unique(lbl)) > 1:
                    s = silhouette_score(fs, lbl, sample_size=min(300, n))
                    if s > best_s:
                        best_s, best_k = s, k
            except Exception:
                continue
        return best_k

    def _cluster_select(self, x_candidates, scores, n_select):
        if len(x_candidates) <= n_select:
            return x_candidates, torch.arange(len(x_candidates), device=self.device)

        k = self._find_k(
            torch.cat([x_candidates, scores.unsqueeze(1)], 1).cpu().numpy())
        k = max(2, min(k, len(x_candidates)//2, n_select))

        features = torch.cat([x_candidates, scores.unsqueeze(1)], 1).cpu().numpy()
        labels   = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features)
        scores_np = scores.cpu().numpy()

        selected = []
        ppc = n_select // k
        rem = n_select % k
        for cid in range(k):
            idxs  = np.where(labels == cid)[0]
            n_pick = min(ppc + (1 if cid < rem else 0), len(idxs))
            if n_pick > 0:
                selected.extend(idxs[np.argsort(scores_np[idxs])[-n_pick:]])

        sel = np.array(selected)
        return x_candidates[sel], torch.tensor(sel, dtype=torch.long, device=self.device)

    # -- Main entry point ------------------------------------------------------
    def select_data_points(self, x_unlabeled, y_unlabeled, n_select,
                           adaptive_k=True, verbose=True, **kwargs):
        if verbose:
            print(f"\n[DataAL v3] Pool={len(x_unlabeled)}, select={n_select}, "
                  f"call#{self._call_count}")

        grad_s  = self.compute_gradient_score(x_unlabeled)
        phys_s  = self.compute_physics_residual_score(x_unlabeled)
        div_s   = self.compute_diversity_score(x_unlabeled)
        combined = self.combine_scores(grad_s, phys_s, div_s)

        sel_x, sel_idx = self._cluster_select(x_unlabeled, combined, n_select)
        sel_y = y_unlabeled[sel_idx]

        score_info = {
            'gradient_scores':  grad_s.cpu().numpy(),
            'physics_scores':   phys_s.cpu().numpy(),
            'diversity_scores': div_s.cpu().numpy(),
            'combined_scores':  combined.cpu().numpy(),
            # Legacy keys for visualization compatibility
            'epistemic_scores': grad_s.cpu().numpy(),
            'aleatoric_scores': phys_s.cpu().numpy(),
            'epistemic_norm':   self._normalize(grad_s).cpu().numpy(),
            'aleatoric_norm':   self._normalize(phys_s).cpu().numpy(),
            'diversity_norm':   self._normalize(div_s).cpu().numpy(),
            'selected_indices': sel_idx.cpu().numpy(),
            'all_candidates_x': x_unlabeled.cpu().numpy(),
            'all_candidates_y': y_unlabeled.cpu().numpy(),
            'used_uncertainty': False,
        }

        if verbose:
            si = sel_idx.cpu().numpy()
            print(f"   grad_mean={grad_s[si].mean():.3e}  "
                  f"phys_mean={phys_s[si].mean():.3e}  "
                  f"div_mean={div_s[si].mean():.3e}")

        return sel_x, sel_y, sel_idx, score_info


class RandomDataPointSelector:
    """Random baseline selector - same interface"""

    def __init__(self, device):
        self.device = device

    def update_labeled_points(self, x_labeled):
        pass

    def select_data_points(self, x_unlabeled, y_unlabeled, n_select,
                           verbose=True, **kwargs):
        if verbose:
            print(f"\n[RandomDataAL] Pool={len(x_unlabeled)}, select={n_select}")
        n   = min(n_select, len(x_unlabeled))
        idx = torch.randperm(len(x_unlabeled), device=self.device)[:n]
        dummy = np.zeros(len(x_unlabeled))
        score_info = {
            'epistemic_scores': dummy, 'aleatoric_scores': dummy,
            'diversity_scores': dummy, 'combined_scores':  dummy,
            'epistemic_norm':   dummy, 'aleatoric_norm':   dummy,
            'diversity_norm':   dummy, 'gradient_scores':  dummy,
            'physics_scores':   dummy,
            'selected_indices': idx.cpu().numpy(),
            'all_candidates_x': x_unlabeled.cpu().numpy(),
            'all_candidates_y': y_unlabeled.cpu().numpy(),
            'used_uncertainty': False,
        }
        return x_unlabeled[idx], y_unlabeled[idx], idx, score_info
