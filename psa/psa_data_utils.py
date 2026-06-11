"""
psa_data_utils.py - scenario-level data handling for the PSA case study.

Loads the merged simulator output (.mat v7.3 via h5py, with a scipy fallback
for older formats), groups rows into scenarios by their IC fingerprint,
splits at scenario level so no initial condition leaks between pool, val and
test, and builds the operating-condition-clustered pool used in the paper.


Main entry points:
  load_step_data / load_lhs_test_data    .mat loaders (102-col x, 4-col y)
  identify_scenarios / split_scenarios   scenario grouping, stratified split
  build_pool_tensors / get_rows          row gathering
  make_biased_scenario_pool              clustered-pool construction
  compute_bounds / compute_y_bounds      normalisation helpers
  evaluate_model                         per-output MSE/MAE on a test set
"""

import numpy as np
import torch
import scipy.io as io
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Optional
import h5py


N_Z_POINTS = 25     # fixed spatial grid size
N_STEPS    = 25     # time steps extracted per scenario -> 25x25 = 625 rows each
                    # Bumped from 3 (legacy) for the 403-scenario pool.
                    # Each scenario in the merged data has ~1,375 timesteps,
                    # so 25 is conservative and still equal across scenarios.


# -- Format-agnostic .mat loader (handles both v7.3/HDF5 and v5/v7) -----------

def _load_mat_arrays(mat_path: str, *keys) -> Tuple[np.ndarray, ...]:
    """
    Load named arrays from a .mat file, auto-detecting the format.

    MATLAB has two on-disk formats:
      - v7.3 / HDF5: requires h5py, arrays are stored TRANSPOSED (need .T)
      - v5 / v7:    requires scipy.io.loadmat, arrays are stored as-is

    Try h5py first; on signature failure, fall back to scipy.io.

    Returns a tuple of np.float64 arrays in the order requested.
    """
    # Try v7.3 / HDF5
    try:
        with h5py.File(mat_path, 'r') as d:
            return tuple(np.real(np.array(d[k])).T.astype(np.float64) for k in keys)
    except (OSError, IOError):
        pass
    # Fall back to v5 / v7
    d = io.loadmat(mat_path)
    return tuple(np.real(d[k]).astype(np.float64) for k in keys)


# -- Load (auto-detects v7.3 / HDF5 vs v5/v7 format) --------------------------

def load_step_data(mat_path: str, step: str = 'ads') -> Tuple[torch.Tensor, torch.Tensor]:
    """Load training .mat file, auto-detecting v7.3 / v5 / v7 format.
    Returns x(N,102) and y(N,4) float64."""
    x, y = _load_mat_arrays(mat_path, f'{step}_x', f'{step}_y')
    return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


def load_lhs_test_data(mat_path: str, step: str = 'ads') -> Tuple[torch.Tensor, torch.Tensor]:
    """Load LHS test .mat (test_<step>_LHS_withbv.mat), auto-detecting format.

    Test files have 103 columns (extra bv = bed voidage at end).
    We strip the trailing bv column to match training data's 102-col layout.

    Returns x(N,102) and y(N,4) float64.
    """
    x, y = _load_mat_arrays(mat_path, f'{step}_x', f'{step}_y')
    if x.shape[1] == 103:
        x = x[:, :102]                                # drop bv column
    elif x.shape[1] != 102:
        raise ValueError(f'Unexpected col count in test file: {x.shape[1]}')
    return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


# -- Bounds --------------------------------------------------------------------

def compute_bounds(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-feature [0,1] / data-driven bounds for PINN [-1,1] normalisation.
    Cols 0..51 are clamped to [0,1] (z, t and the y1/P IC profiles).
    Always computed from the FULL dataset for stable, consistent bounds."""
    lb, _ = x.min(dim=0); ub, _ = x.max(dim=0)
    lb = lb.unsqueeze(0).double(); ub = ub.unsqueeze(0).double()
    zero = (ub - lb) < 1e-12; ub[zero] = lb[zero] + 1.0
    lb[:, :52] = 0.0; ub[:, :52] = 1.0
    return lb, ub


# -- Output normalisation ------------------------------------------------------

def compute_y_bounds(y: torch.Tensor):
    """Per-output [0,1] bounds from full dataset."""
    y_lb, _ = y.min(dim=0); y_ub, _ = y.max(dim=0)
    y_lb = y_lb.unsqueeze(0).double(); y_ub = y_ub.unsqueeze(0).double()
    zero = (y_ub - y_lb) < 1e-10; y_ub[zero] = y_lb[zero] + 1.0
    return y_lb, y_ub

def normalise_y(y, y_lb, y_ub):
    return (y.double() - y_lb) / (y_ub - y_lb)

def denormalise_y(y_norm, y_lb, y_ub):
    return y_norm.double() * (y_ub - y_lb) + y_lb


# -- Scenario identification --------------------------------------------------

def identify_scenarios(x: torch.Tensor) -> List[dict]:
    """Group rows by unique IC fingerprint -> list of scenario dicts."""
    x_np = x.double().numpy()
    fp   = np.round(x_np[:, 2:12], 3)

    scen_map: Dict[bytes, list] = {}
    for i in range(len(x_np)):
        key = fp[i].tobytes()
        if key not in scen_map:
            scen_map[key] = []
        scen_map[key].append(i)

    scenarios = []
    for idx_list in scen_map.values():
        idx = np.array(idx_list, dtype=np.int64)
        ic  = np.array([
            float(x_np[idx, 2:27].mean()),
            float(x_np[idx, 27:52].mean()),
            float(x_np[idx, 100].mean()),
            float(x_np[idx, 101].mean()),
        ], dtype=np.float64)
        scenarios.append({'indices': idx, 'ic_summary': ic, 'n_rows': len(idx)})
    return scenarios


# -- Equal-time-step row extraction -------------------------------------------

def extract_scenario_rows(
        scenario: dict,
        x:        torch.Tensor,
        n_steps:  int = N_STEPS,
) -> np.ndarray:
    """Return exactly n_steps x N_Z_POINTS row indices from this scenario.

    Strategy: find all unique t-values in the scenario, select n_steps
    of them at evenly-spaced quantiles (including first and last), then
    return all 25 z-rows at each chosen t.
    """
    idx   = scenario['indices']
    t_all = x[idx.tolist(), 1].double().numpy()
    t_uniq = np.unique(np.round(t_all, 6))

    if len(t_uniq) <= n_steps:
        chosen_t = t_uniq
    else:
        pos      = np.linspace(0, len(t_uniq) - 1, n_steps, dtype=int)
        chosen_t = t_uniq[pos]

    selected = []
    for ct in chosen_t:
        mask = np.abs(t_all - ct) < 1e-5
        selected.extend(idx[mask].tolist())

    selected = list(dict.fromkeys(selected))
    if len(selected) < n_steps * N_Z_POINTS:
        all_set = set(selected)
        for i in idx.tolist():
            if len(selected) >= n_steps * N_Z_POINTS: break
            if i not in all_set: selected.append(i); all_set.add(i)
    return np.array(selected[:n_steps * N_Z_POINTS], dtype=np.int64)


# -- Stratified scenario split ------------------------------------------------

def split_scenarios(
        all_scenarios: List[dict],
        n_val_scen:    int  = 30,
        n_test_scen:   int  = 40,
        seed:          int  = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split into pool / val / test at scenario level (no IC leakage).
    Stratified by mean_y1_IC bands.

    Defaults updated for 403-scenario pool: 30 val + 40 test = 70 reserved,
    leaving ~333 for the AL pool.  At budget=120 (current max), that's ~36%
    of pool - plenty of headroom for AL to differentiate from random.
    """
    N = len(all_scenarios)
    assert N >= n_val_scen + n_test_scen + 1, \
        f'Too few scenarios ({N}) for val={n_val_scen} + test={n_test_scen}'

    rng = np.random.default_rng(seed)
    y1  = np.array([s['ic_summary'][0] for s in all_scenarios])
    ord_= np.argsort(y1).tolist()
    bs  = N // 3
    bands = [ord_[:bs], ord_[bs:2*bs], ord_[2*bs:]]

    val_idx, test_idx = [], []
    nv = max(1, n_val_scen  // 3)
    nt = max(1, n_test_scen // 3)

    for band in bands:
        b = list(band); rng.shuffle(b)
        test_idx.extend(b[:nt]); val_idx.extend(b[nt:nt+nv])

    used   = set(val_idx) | set(test_idx)
    unused = [i for i in range(N) if i not in used]; rng.shuffle(unused)
    while len(test_idx) < n_test_scen and unused: test_idx.append(unused.pop(0))
    while len(val_idx)  < n_val_scen  and unused: val_idx.append(unused.pop(0))

    test_set = set(test_idx[:n_test_scen]); val_set = set(val_idx[:n_val_scen])
    pool_idx = [i for i in range(N) if i not in test_set | val_set]
    return ([all_scenarios[i] for i in pool_idx],
            [all_scenarios[i] for i in val_set],
            [all_scenarios[i] for i in test_set])


# -- Row gathering ------------------------------------------------------------

def get_rows(
        scenario_list: List[dict],
        x:             torch.Tensor,
        y:             torch.Tensor,
        n_steps:       int = N_STEPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not scenario_list:
        return (torch.empty(0, x.shape[1], dtype=torch.float64),
                torch.empty(0, y.shape[1], dtype=torch.float64))
    all_idx = np.concatenate([
        extract_scenario_rows(s, x, n_steps) for s in scenario_list
    ])
    return x[all_idx.tolist()].double(), y[all_idx.tolist()].double()


def build_pool_tensors(
        pool_scenarios: List[dict],
        x_all:          torch.Tensor,
        y_all:          torch.Tensor,
        n_steps:        int = N_STEPS,
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """Build pool row tensors with scenario indices remapped to LOCAL positions."""
    all_idx = np.concatenate([
        extract_scenario_rows(s, x_all, n_steps) for s in pool_scenarios
    ])
    x_pool = x_all[all_idx.tolist()].double()
    y_pool = y_all[all_idx.tolist()].double()

    lookup = {int(g): int(l) for l, g in enumerate(all_idx)}
    remapped = []
    for s in pool_scenarios:
        local_idx = extract_scenario_rows(s, x_all, n_steps)
        local     = np.array([lookup[int(i)] for i in local_idx], dtype=np.int64)
        remapped.append({'indices': local,
                         'ic_summary': s['ic_summary'].copy(),
                         'n_rows': len(local)})
    return x_pool, y_pool, remapped


# -- Biased pool --------------------------------------------------------------

def make_biased_scenario_pool(
        pool_scenarios: List[dict],
        bias_fraction:  float = 0.70,
        seed:           int   = 42,
) -> List[dict]:
    """Operating-condition-clustered pool: replicate scenarios near 3 IC
    nominals so random sampling keeps landing in the dense regions while the
    spread stays thin. The test set is left uniform.

    Mirrors the CSTR pool construction. On a uniform LHS pool random
    selection is already close to information-optimal and AL has little room
    to improve on it; clustering gives AL under-covered regions to find.

    Note: an earlier version only reordered the pool, which turned out to be
    a no-op because the trainer re-sorts scenarios for stratified seed
    selection. The pool has to be an actual multiset of replicas.

    Pool size stays close to N. Each replica keeps the same `indices` array,
    so every copy maps back to the same training rows.
    """
    rng = np.random.default_rng(seed)
    N   = len(pool_scenarios)
    if N == 0:
        return pool_scenarios

    S   = np.stack([s['ic_summary'] for s in pool_scenarios])
    span = np.where(S.max(0) - S.min(0) > 1e-8, S.max(0) - S.min(0), 1.0)
    Sn  = (S - S.min(0)) / span

    # 3 nominals at quantiles 1/6, 1/2, 5/6 on the first IC dim (y1)
    srt   = np.argsort(S[:, 0])
    nom_i = [int(srt[int(N * q)]) for q in (1/6, 1/2, 5/6)]

    # For each scenario, distance to nearest of the 3 nominals
    d_to_noms = np.stack([np.linalg.norm(Sn - Sn[ni], axis=1) for ni in nom_i])
    min_d = d_to_noms.min(axis=0)            # (N,)

    # Define a SMALL "true core" of scenarios near nominals (15% of pool).
    # These get heavily replicated.  The rest are "spread" and kept singly.
    # Final pool stays close to N entries, but biased_fraction of those are
    # duplicates of the core - so random sampling hits the same dense regions
    # repeatedly while AL avoids them.
    n_core = max(3, int(N * 0.15))            # ~50 scenarios at N=333
    biased_idx = np.argsort(min_d)[:n_core].tolist()
    spread_idx = np.argsort(min_d)[n_core:].tolist()

    # Target: bias_fraction of final pool comes from core (with replication),
    # (1-bias_fraction) comes from spread (singly).  Subsample spread if
    # needed; replicate core to fill.
    target_spread = max(1, int(round(N * (1 - bias_fraction))))
    if target_spread < len(spread_idx):
        # Subsample spread to keep pool size close to N
        pick = rng.choice(len(spread_idx), target_spread, replace=False)
        spread_idx = [spread_idx[i] for i in pick]

    n_spread = len(spread_idx)
    target_biased = N - n_spread
    k_b = max(1, int(round(target_biased / max(1, n_core))))

    # each replica must be a fresh dict: the trainer tracks labeled scenarios
    # by id(), so reusing one object would drop every copy from the pool the
    # moment one of them gets picked
    def _replicate(src, rep_idx):
        return {
            'indices'   : src['indices'].copy(),
            'ic_summary': src['ic_summary'].copy(),
            'n_rows'    : src['n_rows'],
            '_replica_of': src.get('_replica_of', id(src)),
            '_replica_id': rep_idx,
        }

    out = []
    for i in biased_idx:
        for r in range(k_b):
            out.append(_replicate(pool_scenarios[i], r))
    for i in spread_idx:
        out.append(_replicate(pool_scenarios[i], 0))

    # Shuffle so biased replicas are mixed with spread (random selection sees
    # uniform pool indices but biased multiplicity)
    perm = rng.permutation(len(out))
    out_shuffled = [out[i] for i in perm]

    n_unique_biased = len(biased_idx)
    n_unique_spread = len(spread_idx)
    pool_size = len(out_shuffled)
    print(f'  [biased_pool] unique={n_unique_biased + n_unique_spread} '
          f'({n_unique_biased} biased x{k_b} + {n_unique_spread} spread) '
          f'-> pool size {pool_size}, '
          f'effective bias = {k_b * n_unique_biased / pool_size:.0%}')
    return out_shuffled


# -- Evaluation ----------------------------------------------------------------

def evaluate_model(
        model,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        device:  str   = "cpu",
        y_lb:    torch.Tensor = None,
        y_ub:    torch.Tensor = None,
) -> Dict[str, float]:
    """MSE + MAE per output, in original scale (denormalised if bounds provided).
    y_lb/y_ub are the output normalisation bounds from compute_y_bounds().
    """
    model.eval()
    with torch.no_grad():
        yh = model.forward_deterministic(x_test.double().to(device)).cpu().double()
    if y_lb is not None and y_ub is not None:
        yh = denormalise_y(yh, y_lb.cpu(), y_ub.cpu())
    y_test = y_test.double().cpu()
    diff = yh - y_test.double().cpu()
    mse  = (diff**2).mean(dim=0); mae = diff.abs().mean(dim=0)
    out  = {"mse_total": float(mse.mean()), "mae_total": float(mae.mean())}
    for i, n in enumerate(["y1", "P", "q1", "q2"]):
        out[f"mse_{n}"] = float(mse[i]); out[f"mae_{n}"] = float(mae[i])
    return out


# -- IC PCA compression (legacy, kept for backwards compat) -------------------

N_IC_COLS = 98
N_PCA_COMPONENTS = 8

def fit_ic_pca(
        all_scenarios: List[dict],
        x_all:         torch.Tensor,
        n_components:  int = N_PCA_COMPONENTS,
):
    """Legacy PCA fit on IC blocks.  v3 model uses fit_pca_for_bottleneck()
    in psa_pinn_model_v3.py instead."""
    ic_vecs = np.stack([
        x_all[int(s['indices'][0]), 2:100].double().numpy()
        for s in all_scenarios])
    pca = PCA(n_components=n_components)
    pca.fit(ic_vecs)
    return pca


def compress_x(x, pca):
    """Legacy compression: 102-dim -> (k+4)-dim.
    Not used by v3 model (which has internal PCA bottleneck instead)."""
    x_np = x.double().cpu().numpy()
    ic_block = x_np[:, 2:100]
    scores = pca.transform(ic_block)
    return torch.tensor(np.concatenate(
        [x_np[:, :2], scores, x_np[:, 100:]], axis=1
    ), dtype=torch.float64)


def compressed_bounds(x_compressed):
    lb, _ = x_compressed.min(dim=0); ub, _ = x_compressed.max(dim=0)
    lb = lb.unsqueeze(0).double(); ub = ub.unsqueeze(0).double()
    zero = (ub - lb) < 1e-12; ub[zero] = lb[zero] + 1.0
    return lb, ub
