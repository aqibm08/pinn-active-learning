"""
Single-signal AL selectors at the PSA scenario level - one signal at a time,
so the ablation can tell which signal is actually doing the work.

Modes:
  badge       farthest-first on last-layer pseudo-gradient embeddings;
              model-aware diversity (this is what the paper's PINN+AL uses)
  qbc         committee disagreement, per-scenario top-k mean
  physics     PDE residual magnitude, per-scenario top-k mean; only useful
              if the residual is a decent proxy for "where the model is
              wrong", which fails when the PDE carries intrinsic residual
              on the real data
  gradient    ||d gamma / dx||, per-scenario top-k mean
  ic_coreset  model-free farthest-first on each scenario's [y1, P] IC
              summary; pure data-side coverage
  random      baseline

All modes use greedy top-k or farthest-first. No rank mixing, no adaptive
weights, no access to the validation set.
"""

import numpy as np
import torch

from psa_data_selector_v4 import (
    _physics_score_topk, _qbc_score_topk, _gradient_score_topk,
    _badge_emb_per_scenario, _farthest_first,
)


class PSADataSelectorV5:
    VALID_MODES = ('badge', 'qbc', 'physics', 'gradient', 'ic_coreset', 'random')

    def __init__(self, mode: str = 'badge', k_select: int = 2,
                 device: str = 'cpu', k_top: int = 15, **kwargs):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.k = k_select
        self.device = device
        self.k_top = k_top

    # Same call signature as v4 so trainer's try-except chain picks it up
    def select(self, model, pool_scenarios, labeled_scenarios,
               x_pool_orig, x_val=None,
               rng_seed: int = 42, round_idx: int = 0, max_rounds: int = 1,
               **kwargs):
        n = len(pool_scenarios)
        if n == 0:
            return []
        k = min(self.k, n)

        if self.mode == 'random':
            rng = np.random.default_rng(rng_seed)
            idx = rng.choice(n, k, replace=False)
            return [pool_scenarios[int(i)] for i in idx]

        if self.mode == 'ic_coreset':
            return self._ic_coreset(pool_scenarios, labeled_scenarios, k)

        if self.mode == 'badge':
            return self._badge(model, pool_scenarios, labeled_scenarios, x_pool_orig, k)

        # Scalar-score modes
        if self.mode == 'physics':
            scores = _physics_score_topk(model, pool_scenarios, x_pool_orig,
                                          self.device, k_top=self.k_top)
        elif self.mode == 'qbc':
            scores = _qbc_score_topk(model, pool_scenarios, x_pool_orig,
                                      self.device, k_top=self.k_top)
        elif self.mode == 'gradient':
            scores = _gradient_score_topk(model, pool_scenarios, x_pool_orig,
                                           self.device, k_top=self.k_top)
        else:
            raise AssertionError(f'unreachable mode: {self.mode}')

        # Degenerate-signal fallback: if std ~ 0 the signal is uninformative
        # at this round; pick random to avoid biasing toward arbitrary first
        # scenario in argsort.
        if float(np.std(scores)) < 1e-12:
            rng = np.random.default_rng(rng_seed + round_idx)
            idx = rng.choice(n, k, replace=False)
        else:
            # Greedy top-k by raw score (descending).  Note: no diversity
            # term - this mode tests the SCORE alone, with all the
            # consequences (top-K can cluster on similar scenarios).
            idx = np.argsort(-scores)[:k]

        return [pool_scenarios[int(i)] for i in idx]

    # -- Mode implementations --------------------------------------------------

    def _badge(self, model, pool_scenarios, labeled_scenarios, x_pool_orig, k):
        pool_emb = _badge_emb_per_scenario(model, pool_scenarios, x_pool_orig,
                                            self.device)
        if labeled_scenarios:
            lab_emb = _badge_emb_per_scenario(model, labeled_scenarios,
                                               x_pool_orig, self.device)
        else:
            lab_emb = np.zeros((0, pool_emb.shape[1]), dtype=np.float64)

        selected_idx = []
        cur_lab = lab_emb.copy()
        for _ in range(k):
            if cur_lab.shape[0] > 0:
                div = _farthest_first(pool_emb, cur_lab)
            else:
                div = np.linalg.norm(pool_emb, axis=1)
            for si in selected_idx:
                div[si] = -np.inf
            best = int(np.argmax(div))
            selected_idx.append(best)
            cur_lab = np.vstack([cur_lab, pool_emb[best:best+1]])

        return [pool_scenarios[i] for i in selected_idx]

    def _ic_coreset(self, pool_scenarios, labeled_scenarios, k):
        """Farthest-first on normalised [mean y1_IC, mean P_IC] per scenario.

        Cols 2 and 3 of ic_summary are mean of corrupt q2_IC tail (cols 100, 101)
        and would just add noise.  We use the 2 physically-meaningful dims.
        """
        pool_ic = np.stack([s['ic_summary'][:2] for s in pool_scenarios])
        if labeled_scenarios:
            lab_ic = np.stack([s['ic_summary'][:2] for s in labeled_scenarios])
        else:
            lab_ic = np.zeros((0, 2))

        all_ic = np.vstack([pool_ic, lab_ic]) if lab_ic.shape[0] > 0 else pool_ic
        lo, hi = all_ic.min(axis=0), all_ic.max(axis=0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        pool_n = (pool_ic - lo) / span
        lab_n = (lab_ic - lo) / span if lab_ic.shape[0] > 0 else lab_ic

        selected_idx = []
        cur_lab = lab_n.copy()
        for _ in range(k):
            if cur_lab.shape[0] == 0:
                # First pick: scenario closest to pool median (representative)
                med = np.median(pool_n, axis=0)
                d = np.linalg.norm(pool_n - med, axis=1)
                best = int(np.argmin(d))
            else:
                diff = pool_n[:, None, :] - cur_lab[None, :, :]
                d = np.linalg.norm(diff, axis=2).min(axis=1)
                for si in selected_idx:
                    d[si] = -np.inf
                best = int(np.argmax(d))
            selected_idx.append(best)
            cur_lab = np.vstack([cur_lab, pool_n[best:best+1]])

        return [pool_scenarios[i] for i in selected_idx]


# Smoke test
if __name__ == '__main__':
    print('smoke test: PSADataSelectorV5 modes')

    # Fake scenarios with synthetic ic_summary
    rng = np.random.default_rng(0)
    n_pool = 50
    scenarios = []
    for i in range(n_pool):
        scenarios.append({
            'indices': np.array([i * 10 + j for j in range(10)], dtype=np.int64),
            'ic_summary': rng.standard_normal(4),
            'n_rows': 10,
        })

    id2idx = {id(s): i for i, s in enumerate(scenarios)}
    for mode in ('random', 'ic_coreset'):
        sel = PSADataSelectorV5(mode=mode, k_select=3)
        out = sel.select(None, scenarios, [], None)
        assert len(out) == 3
        picked = [id2idx[id(s)] for s in out]
        print(f'  mode={mode}: picked {len(out)} scenarios (indices: {picked})')

    print('SMOKE TEST PASSED.')
