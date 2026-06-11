"""
Multi-signal AL selector for PSA scenarios (the "v4 mixture" of the paper's
signal ablation), plus the per-scenario scoring helpers shared with the v5
single-signal selectors.

The mixture combines four per-scenario scores by rank percentile with
round-dependent weights: PDE residual, committee disagreement and input-
gradient norm (each aggregated per scenario by top-k mean over its rows),
and BADGE farthest-first diversity. Early rounds lean almost entirely on
diversity, because the model-aware signals are noise before the model has
fit anything; later rounds rebalance toward physics + committee.

An Expected-Error-Reduction term (cosine alignment between a scenario's
BADGE embedding and the mean validation embedding) is implemented below but
weighted zero: it steers selection toward the validation distribution,
which hurt generalisation whenever the evaluation distribution differed
from validation. Kept for reproducibility of the older experiments.
"""

import numpy as np
import torch


def _rank_pct(v: np.ndarray) -> np.ndarray:
    n = len(v)
    if n <= 1:
        return np.full(n, 0.5, dtype=np.float64)
    order = np.argsort(v)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j)
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks / max(1, n - 1)


def _topk_mean(values: np.ndarray, k: int) -> float:
    if len(values) <= k:
        return float(values.mean())
    top = np.partition(values, -k)[-k:]
    return float(top.mean())


# --- Per-scenario score functions (top-K aggregation) ------------------------

def _physics_score_topk(model, pool_scenarios, x_pool_orig, device, k_top=15):
    if not hasattr(model, 'residual_per_point'):
        return np.zeros(len(pool_scenarios), dtype=np.float64)
    scores = np.zeros(len(pool_scenarios))
    for i, s in enumerate(pool_scenarios):
        x_sub = x_pool_orig[s['indices'].tolist()].double().to(device)
        try:
            res = model.residual_per_point(x_sub).cpu().numpy()
            scores[i] = _topk_mean(res, k_top)
        except Exception:
            scores[i] = 0.0
    return scores


def _qbc_score_topk(model, pool_scenarios, x_pool_orig, device, k_top=15):
    if not hasattr(model, 'committee_disagreement'):
        return np.zeros(len(pool_scenarios), dtype=np.float64)
    scores = np.zeros(len(pool_scenarios))
    for i, s in enumerate(pool_scenarios):
        x_sub = x_pool_orig[s['indices'].tolist()].double().to(device)
        d = model.committee_disagreement(x_sub).cpu().numpy()
        scores[i] = _topk_mean(d, k_top) if len(d) else 0.0
    return scores


def _gradient_score_topk(model, pool_scenarios, x_pool_orig, device, k_top=15):
    was_train = getattr(model, 'training', True)
    model.train()
    scores = np.zeros(len(pool_scenarios))
    for i, s in enumerate(pool_scenarios):
        x_b = x_pool_orig[s['indices'].tolist()].double().to(device).detach().clone()
        x_b.requires_grad_(True)
        with torch.enable_grad():
            model.forward_deterministic(x_b).sum().backward()
        if x_b.grad is not None:
            g = x_b.grad.detach().norm(dim=1).cpu().numpy()
            scores[i] = _topk_mean(g, k_top)
        x_b.grad = None
    if not was_train:
        model.eval()
    return scores


def _badge_emb_per_scenario(model, scenarios, x_pool_orig, device) -> np.ndarray:
    if not hasattr(model, 'last_layer_embedding'):
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(scenarios), 16))
    embs = []
    for s in scenarios:
        x_sub = x_pool_orig[s['indices'].tolist()].double().to(device)
        e = model.last_layer_embedding(x_sub).mean(dim=0).cpu().numpy()
        embs.append(e)
    return np.stack(embs, axis=0)


def _farthest_first(pool_emb, lab_emb):
    if lab_emb.shape[0] == 0:
        return np.ones(pool_emb.shape[0], dtype=np.float64)
    pn = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12)
    ln = lab_emb  / (np.linalg.norm(lab_emb,  axis=1, keepdims=True) + 1e-12)
    diff = pn[:, None, :] - ln[None, :, :]
    return np.linalg.norm(diff, axis=2).min(axis=1)


# --- EER signal: alignment with validation gradient direction ----------------

def _eer_score(pool_emb: np.ndarray, val_emb_mean: np.ndarray) -> np.ndarray:
    """EER proxy: |cosine| between a scenario embedding and the mean
    validation embedding. For a quadratic loss the most useful training
    point is the one whose gradient best aligns with the validation
    gradient; without sign information the absolute cosine is the usable
    stand-in. Higher = better candidate.
    """
    pn = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12)
    vn = val_emb_mean / (np.linalg.norm(val_emb_mean) + 1e-12)
    # Absolute cosine similarity
    return np.abs(pn @ vn)


# --- Main selector v4 ---------------------------------------------------------

class PSADataSelectorV4:
    """Rank-percentile mixture of physics + qbc + gradient + BADGE diversity.

    Weights interpolate (cosine in round_idx / max_rounds) from W_INIT,
    which is diversity-dominated, to W_FINAL, which is balanced. The EER
    weight stays at zero; see the module docstring for why.
    """

    W_INIT  = dict(physics=0.15, qbc=0.15, gradient=0.10, badge=0.60, eer=0.00)
    W_FINAL = dict(physics=0.30, qbc=0.30, gradient=0.15, badge=0.25, eer=0.00)

    def __init__(self,
                 k_select: int = 2,
                 device:   str = 'cpu',
                 k_top:    int = 15,
                 # Final-round weight overrides
                 w_physics:  float = None,
                 w_qbc:      float = None,
                 w_gradient: float = None,
                 w_badge:    float = None,
                 w_eer:      float = None,
                 **kwargs):
        self.k = k_select
        self.device = device
        self.k_top = k_top
        if w_physics  is not None: self.W_FINAL['physics']  = w_physics
        if w_qbc      is not None: self.W_FINAL['qbc']      = w_qbc
        if w_gradient is not None: self.W_FINAL['gradient'] = w_gradient
        if w_badge    is not None: self.W_FINAL['badge']    = w_badge
        if w_eer      is not None: self.W_FINAL['eer']      = w_eer

    def _weights(self, round_idx: int, max_rounds: int) -> dict:
        r = min(1.0, round_idx / max(1, max_rounds))
        a = 0.5 * (1 - np.cos(np.pi * r))
        return {k: (1 - a) * self.W_INIT[k] + a * self.W_FINAL[k]
                for k in self.W_FINAL}

    def select(self, model, pool_scenarios, labeled_scenarios,
               x_pool_orig, x_val=None,
               rng_seed: int = 42,
               round_idx: int = 0, max_rounds: int = 1,
               **kwargs) -> list:
        n = len(pool_scenarios)
        if n == 0:
            return []
        k = min(self.k, n)

        # -- 1. Compute scores ------------------------------------------------
        s_phys = _physics_score_topk (model, pool_scenarios, x_pool_orig,
                                       self.device, k_top=self.k_top)
        s_qbc  = _qbc_score_topk     (model, pool_scenarios, x_pool_orig,
                                       self.device, k_top=self.k_top)
        s_grad = _gradient_score_topk(model, pool_scenarios, x_pool_orig,
                                       self.device, k_top=self.k_top)
        pool_emb = _badge_emb_per_scenario(model, pool_scenarios, x_pool_orig, self.device)
        if labeled_scenarios:
            lab_emb = _badge_emb_per_scenario(model, labeled_scenarios, x_pool_orig, self.device)
        else:
            lab_emb = np.zeros((0, pool_emb.shape[1]), dtype=np.float64)
        s_div = _farthest_first(pool_emb, lab_emb)

        # -- EER: requires validation embedding (compute once, mean it) ------
        if x_val is not None and hasattr(model, 'last_layer_embedding'):
            with torch.no_grad():
                val_emb = model.last_layer_embedding(
                    x_val.double().to(self.device)).cpu().numpy()
            val_emb_mean = val_emb.mean(axis=0)
            s_eer = _eer_score(pool_emb, val_emb_mean)
        else:
            s_eer = np.zeros(n)

        # -- 2. Rank-percentile each (constant 0.5 if degenerate) -------------
        def _safe_rank(v):
            return _rank_pct(v) if v.std() > 1e-12 else np.full(n, 0.5)
        r_phys = _safe_rank(s_phys); r_qbc  = _safe_rank(s_qbc)
        r_grad = _safe_rank(s_grad); r_div  = _safe_rank(s_div)
        r_eer  = _safe_rank(s_eer)

        # -- 3. Adaptive weights ----------------------------------------------
        w = self._weights(round_idx, max_rounds)

        # -- 4. Greedy top-k with farthest-first reweighting ------------------
        selected, selected_idx = [], []
        cur_lab_emb = lab_emb.copy()
        for _ in range(k):
            combined = (w['physics']  * r_phys
                      + w['qbc']      * r_qbc
                      + w['gradient'] * r_grad
                      + w['badge']    * r_div
                      + w['eer']      * r_eer)
            for si in selected_idx:
                combined[si] = -np.inf
            best = int(np.argmax(combined))
            selected.append(pool_scenarios[best])
            selected_idx.append(best)
            cur_lab_emb = np.vstack([cur_lab_emb, pool_emb[best:best+1]])
            new_div = _farthest_first(pool_emb, cur_lab_emb)
            r_div = _safe_rank(new_div)

        return selected


class RandomPSADataSelector:
    def __init__(self, k_select=2, device='cpu', **kwargs):
        self.k = k_select

    def select(self, model, pool_scenarios, labeled_scenarios,
               x_pool_orig, x_val=None, rng_seed=42, **kwargs):
        n = len(pool_scenarios)
        if n == 0:
            return []
        idx = np.random.default_rng(rng_seed).choice(n, min(self.k, n), replace=False)
        return [pool_scenarios[i] for i in idx]
