"""
CSTR budget sweep: sequential active learning vs random baselines on the
operating-condition-clustered pool.


Methods:
  sequential_al      sequential queries, AL scoring each round
  sequential_random  same protocol, random queries (controls for the
                     effect of sequential retraining alone)
  random_random      one-shot random (standard baseline)
"""

import argparse
import json
import copy
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from main_dual_al_experiment import (
    DualHeadPhysicsPINN, load_and_split_data, device
)
from active_learning_core import ActiveLearningSelector
from data_point_selector import DataPointSelector, RandomDataPointSelector
from dual_al_trainer import DualActiveLearningTrainer, evidential_loss
from visualization_budget_sweep import generate_sweep_visualizations

torch.set_default_dtype(torch.float64)


# =======================================================================
# CONFIGURATION
# =======================================================================
SWEEP_CONFIG = {
    'budgets':  [50, 100, 150, 200, 300, 400, 500],
    'seeds':    [42, 123, 456, 789, 1337],
    'methods':  ['sequential_al', 'random_random', 'sequential_random'],

    'n_val':    150,
    'n_test':   150,

    # Sequential AL
    'n_initial':       10,
    'k_per_round':     10,
    'epochs_per_round': 300,
    'final_epochs':    1000,
    'lr':              1e-3,
    'weight_decay':    1e-5,
    'lambda_physics':  1.0,
    'patience':        150,

    # Collocation
    'coll_multiplier':  4,
    'n_coll_candidates': 2000,

    # Biased pool
    'biased_pool':   True,
    'bias_fraction': 0.70,
    'bias_noise':    0.08,

    # AL weights - data
    'w_gradient_data':  0.35,
    'w_physics_data':   0.35,
    'w_diversity_data': 0.30,
    # AL weights - collocation
    'w_physics_coll':   0.50,
    'w_uncertainty':    0.20,
    'w_gradient_coll':  0.30,

    # Architecture
    'input_dim':  6,
    'output_dim': 4,
    'activation': 'tanh',

    'results_dir': 'sweep_results',
}


# =======================================================================
# ADAPTIVE ARCHITECTURE
# =======================================================================
def get_arch(budget):
    if budget <= 50:
        return dict(hidden_dim=64,  n_backbone_layers=2)
    elif budget <= 150:
        return dict(hidden_dim=128, n_backbone_layers=3)
    elif budget <= 300:
        return dict(hidden_dim=192, n_backbone_layers=3)
    else:
        return dict(hidden_dim=256, n_backbone_layers=4)


def make_model(budget, cfg):
    arch = get_arch(budget)
    m = DualHeadPhysicsPINN(
        input_dim        =cfg['input_dim'],
        hidden_dim       =arch['hidden_dim'],
        output_dim       =cfg['output_dim'],
        n_backbone_layers=arch['n_backbone_layers'],
        activation       =cfg['activation'],
    ).to(device)
    n = sum(p.numel() for p in m.parameters())
    print(f"   Model: h={arch['hidden_dim']} L={arch['n_backbone_layers']} params={n:,}")
    return m


# =======================================================================
# BIASED POOL CONSTRUCTION
# =======================================================================
def make_biased_pool(x_pool_orig, y_pool_orig, cfg, rng):
    """
    Cluster 70% of pool near 3 nominal operating points.
    Test set is uniform - AL must detect and query the sparse regions.
    """
    if not cfg.get('biased_pool', False):
        return x_pool_orig.clone(), y_pool_orig.clone()

    N = len(x_pool_orig)
    n_biased = int(N * cfg['bias_fraction'])
    n_random = N - n_biased
    noise_std = cfg['bias_noise']

    x_np = x_pool_orig.cpu().numpy()
    # Pick 3 nominal points spread across the input space (first dim proxy)
    sorted_i = np.argsort(x_np[:, 0])
    nominal_idx = [sorted_i[N//6], sorted_i[N//2], sorted_i[5*N//6]]

    biased_x, biased_y = [], []
    per_nominal = n_biased // 3

    for ni in nominal_idx:
        noise = torch.tensor(
            rng.normal(0, noise_std, (per_nominal, x_pool_orig.shape[1])),
            dtype=torch.float64, device=device)
        xi = torch.clamp(x_pool_orig[ni].unsqueeze(0) + noise, -1.0, 1.0)
        # Assign y via nearest neighbour in original pool
        dists = torch.cdist(xi, x_pool_orig)
        nn_idx = dists.argmin(dim=1)
        biased_x.append(xi)
        biased_y.append(y_pool_orig[nn_idx])

    # Remainder: random draws from original pool
    rand_idx = rng.choice(N, n_random, replace=False)
    biased_x.append(x_pool_orig[rand_idx])
    biased_y.append(y_pool_orig[rand_idx])

    xb = torch.cat(biased_x, dim=0)
    yb = torch.cat(biased_y, dim=0)
    perm = torch.randperm(len(xb), device=device)

    print(f"   Biased pool: {len(xb)} pts  "
          f"({n_biased} clustered + {n_random} spread)")
    return xb[perm], yb[perm]


# =======================================================================
# ONE TRAINING STEP
# =======================================================================
def train_step(model, x_lab, y_lab, x_val, y_val, y_norm_bounds,
               cfg, epochs, patience, optimizer=None):
    """
    Train for `epochs` epochs. Pass optimizer to continue sequential training.
    Returns (best_val_loss, optimizer).
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg['lr'],
                               weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=max(10, patience//4), min_lr=1e-6)

    n_coll = cfg['coll_multiplier'] * len(x_lab)
    from scipy.stats.qmc import LatinHypercube
    lhs_samp = LatinHypercube(d=x_lab.shape[1])
    x_coll = torch.tensor(
        lhs_samp.random(n_coll) * 2 - 1,
        dtype=torch.float64, device=device)

    best_val   = float('inf')
    best_state = None
    pat_ctr    = 0

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()

        g, nu, al, be = model.forward_evidential(x_lab)
        d_loss = evidential_loss(g, nu, al, be, y_lab)

        residuals = model.compute_pde_residuals(
            x_coll.clone().detach(), y_norm_bounds)
        p_loss = torch.mean(torch.stack([
            torch.mean(r**2) for r in residuals if r is not None]))
        loss = d_loss + cfg['lambda_physics'] * p_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            gv, nv, av, bv = model.forward_evidential(x_val)
            val_loss = evidential_loss(gv, nv, av, bv, y_val).item()
        model.train()

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            pat_ctr    = 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, optimizer


# =======================================================================
# EVALUATE
# =======================================================================
def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        g, nu, al, be = model.forward_evidential(x_test)
        tl  = evidential_loss(g, nu, al, be, y_test).item()
        tm  = torch.mean((g - y_test)**2).item()
        tma = torch.mean(torch.abs(g - y_test)).item()
        per_mse = [(g[:,i]-y_test[:,i]).pow(2).mean().item()
                   for i in range(y_test.shape[1])]
        per_mae = [(g[:,i]-y_test[:,i]).abs().mean().item()
                   for i in range(y_test.shape[1])]
        eps = (be / (al * nu + 1e-8)).cpu().numpy()
    return tl, tm, tma, per_mse, per_mae, g.cpu().numpy(), eps


# =======================================================================
# SEQUENTIAL RUN (AL or random queries)
# =======================================================================
def run_sequential(budget, seed, use_al, cfg,
                   x_val, y_val, x_test, y_test,
                   y_norm_bounds, x_pool, y_pool):

    n0       = cfg['n_initial']
    k        = cfg['k_per_round']
    n_rounds = max(0, (budget - n0) // k)
    leftover = max(0, budget - n0 - n_rounds * k)

    label = 'Sequential AL' if use_al else 'Sequential Random'
    print(f"\n  {label}: n0={n0}, k={k}, rounds={n_rounds} (+{leftover})")

    # Seed labeled set
    perm   = torch.randperm(len(x_pool), device=device)
    x_lab  = x_pool[perm[:n0]].clone()
    y_lab  = y_pool[perm[:n0]].clone()
    x_unlab= x_pool[perm[n0:]].clone()
    y_unlab= y_pool[perm[n0:]].clone()

    model = make_model(budget, cfg)

    if use_al:
        selector = DataPointSelector(
            model, device,
            w_gradient  =cfg['w_gradient_data'],
            w_physics   =cfg['w_physics_data'],
            w_diversity =cfg['w_diversity_data'],
            y_norm_bounds=y_norm_bounds,
        )
    else:
        selector = RandomDataPointSelector(device)

    optimizer = None
    round_mse  = []
    n_hist     = []
    t0 = time.time()

    for rnd in range(n_rounds + 1):
        is_last = (rnd == n_rounds)
        n_add   = leftover if is_last else k
        if n_add == 0 and is_last:
            break

        # Query new points (skip round 0 - use seed)
        if rnd > 0 and n_add > 0 and len(x_unlab) > 0:
            selector.update_labeled_points(x_lab)
            n_add = min(n_add, len(x_unlab))
            sel_x, sel_y, sel_idx, _ = selector.select_data_points(
                x_unlab, y_unlab, n_select=n_add, verbose=False)
            x_lab  = torch.cat([x_lab,  sel_x], dim=0)
            y_lab  = torch.cat([y_lab,  sel_y], dim=0)
            mask   = torch.ones(len(x_unlab), dtype=torch.bool, device=device)
            si     = sel_idx if isinstance(sel_idx, torch.Tensor) else \
                     torch.tensor(sel_idx, dtype=torch.long, device=device)
            mask[si.long()] = False
            x_unlab = x_unlab[mask]
            y_unlab = y_unlab[mask]

        # Train
        epochs  = cfg['final_epochs'] if is_last else cfg['epochs_per_round']
        patience= cfg['patience'] * 2   if is_last else cfg['patience']

        best_val, optimizer = train_step(
            model, x_lab, y_lab, x_val, y_val, y_norm_bounds,
            cfg, epochs=epochs, patience=patience, optimizer=optimizer)

        _, tm, _, _, _, _, _ = evaluate(model, x_test, y_test)
        round_mse.append(tm)
        n_hist.append(len(x_lab))
        print(f"    Rnd {rnd:2d}/{n_rounds}: N={len(x_lab):3d}  "
              f"val={best_val:.4e}  test_mse={tm:.4e}")

    wall = time.time() - t0
    tl, tm, tma, per_mse, per_mae, preds, eps = evaluate(model, x_test, y_test)

    return {
        'method':             label,
        'method_key':         'sequential_al' if use_al else 'sequential_random',
        'budget':             budget, 'seed': seed,
        'test_loss':          tl, 'test_mse': tm, 'test_mae': tma,
        'wall_time':          wall, 'warmup_time': 0.0,
        'best_val_loss':      min(round_mse) if round_mse else float('inf'),
        'epochs_run':         n_rounds * cfg['epochs_per_round'] + cfg['final_epochs'],
        'output_mse':         per_mse, 'output_mae': per_mae,
        'val_mse_history':    round_mse,
        'total_loss_history': round_mse,
        'uncertainty_history': [],
        'test_predictions':   preds.tolist(),
        'test_targets':       y_test.cpu().numpy().tolist(),
        'test_epistemic':     eps.tolist(),
        'round_mse_history':  round_mse,
        'n_total_history':    n_hist,
    }


# =======================================================================
# ONE-SHOT RANDOM BASELINE
# =======================================================================
def run_random_oneshot(budget, seed, cfg,
                       x_val, y_val, x_test, y_test,
                       y_norm_bounds, x_pool, y_pool):
    perm  = torch.randperm(len(x_pool), device=device)
    x_lab = x_pool[perm[:budget]].clone()
    y_lab = y_pool[perm[:budget]].clone()
    model = make_model(budget, cfg)

    # Give RR the same total compute budget as sequential (approx)
    n_rounds = (budget - cfg['n_initial']) // cfg['k_per_round']
    total_ep = n_rounds * cfg['epochs_per_round'] + cfg['final_epochs']

    t0 = time.time()
    best_val, _ = train_step(
        model, x_lab, y_lab, x_val, y_val, y_norm_bounds,
        cfg, epochs=total_ep, patience=cfg['patience'] * 3)
    wall = time.time() - t0

    tl, tm, tma, per_mse, per_mae, preds, eps = evaluate(model, x_test, y_test)
    return {
        'method':             'Random-Random',
        'method_key':         'random_random',
        'budget':             budget, 'seed': seed,
        'test_loss':          tl, 'test_mse': tm, 'test_mae': tma,
        'wall_time':          wall, 'warmup_time': 0.0,
        'best_val_loss':      best_val,
        'epochs_run':         total_ep,
        'output_mse':         per_mse, 'output_mae': per_mae,
        'val_mse_history':    [],
        'total_loss_history': [],
        'uncertainty_history': [],
        'test_predictions':   preds.tolist(),
        'test_targets':       y_test.cpu().numpy().tolist(),
        'test_epistemic':     eps.tolist(),
    }


# =======================================================================
# DISPATCHER
# =======================================================================
def run_one(budget, seed, method, cfg,
            x_val, y_val, x_test, y_test, y_norm_bounds,
            x_pool_orig, y_pool_orig):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    x_pool, y_pool = make_biased_pool(x_pool_orig, y_pool_orig, cfg, rng)

    if method == 'sequential_al':
        return run_sequential(budget, seed, use_al=True, cfg=cfg,
                              x_val=x_val, y_val=y_val,
                              x_test=x_test, y_test=y_test,
                              y_norm_bounds=y_norm_bounds,
                              x_pool=x_pool, y_pool=y_pool)
    elif method == 'sequential_random':
        return run_sequential(budget, seed, use_al=False, cfg=cfg,
                              x_val=x_val, y_val=y_val,
                              x_test=x_test, y_test=y_test,
                              y_norm_bounds=y_norm_bounds,
                              x_pool=x_pool, y_pool=y_pool)
    elif method == 'random_random':
        return run_random_oneshot(budget, seed, cfg,
                                  x_val=x_val, y_val=y_val,
                                  x_test=x_test, y_test=y_test,
                                  y_norm_bounds=y_norm_bounds,
                                  x_pool=x_pool, y_pool=y_pool)
    else:
        raise ValueError(f"Unknown method: {method}")


# =======================================================================
# SWEEP LOOP
# =======================================================================
def run_sweep(cfg):
    results_dir = Path(cfg['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading data ...")
    (_, _,
     x_pool_orig, y_pool_orig,
     x_val, y_val, x_test, y_test,
     y_norm_bounds, *_) = load_and_split_data(
        n_initial_labeled=10,
        n_val=cfg['n_val'],
        n_test=cfg['n_test'],
        seed=42,
    )
    print(f"   Pool:{len(x_pool_orig)}  Val:{len(x_val)}  Test:{len(x_test)}")

    results = {}
    total   = len(cfg['budgets']) * len(cfg['methods']) * len(cfg['seeds'])
    idx     = 0
    t0      = time.time()

    for budget in cfg['budgets']:
        results[budget] = {m: [] for m in cfg['methods']}
        for method in cfg['methods']:
            for seed in cfg['seeds']:
                idx += 1
                elapsed = time.time() - t0
                eta = elapsed / idx * (total - idx) if idx > 1 else 0
                print(f"\n{'='*80}")
                print(f"  [{idx}/{total}] budget={budget} method={method} "
                      f"seed={seed}  ETA~{eta/60:.1f}m")
                print(f"{'='*80}")
                try:
                    res = run_one(budget, seed, method, cfg,
                                  x_val, y_val, x_test, y_test,
                                  y_norm_bounds, x_pool_orig, y_pool_orig)
                    results[budget][method].append(res)
                    print(f"  [ok] MSE={res['test_mse']:.4e}  "
                          f"t={res['wall_time']:.0f}s")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results[budget][method].append({
                        'method': method, 'budget': budget, 'seed': seed,
                        'test_mse': float('nan'), 'test_mae': float('nan'),
                        'test_loss': float('nan'), 'error': str(e),
                    })
            _save_results(results, results_dir / 'sweep_results_partial.json')

    _save_results(results, results_dir / 'sweep_results_final.json')
    print(f"\nSweep complete in {(time.time()-t0)/60:.1f} min")
    return results


# =======================================================================
# PRINT + AGGREGATE
# =======================================================================
def print_sweep_table(results):
    print(f"\n{'='*110}")
    print("BUDGET SWEEP SUMMARY")
    print(f"{'='*110}")
    print(f"{'Budget':>8} {'Method':<26} {'MSE mean':>14} "
          f"{'MSE std':>12} {'MAE mean':>14} {'Epochs':>8}")
    print("-"*90)
    for budget in sorted(results.keys()):
        for method in results[budget]:
            valid = [r for r in results[budget][method]
                     if 'test_mse' in r and not np.isnan(r['test_mse'])]
            if not valid:
                continue
            mses = [r['test_mse'] for r in valid]
            maes = [r['test_mae'] for r in valid]
            eps  = [r.get('epochs_run', 0) for r in valid]
            lbl  = valid[0].get('method', method)
            print(f"{budget:>8}  {lbl:<26} "
                  f"{np.mean(mses):>14.4e} {np.std(mses):>12.4e} "
                  f"{np.mean(maes):>14.4e} {np.mean(eps):>8.0f}")
        print()

    print(f"\n{'='*110}")
    print("IMPROVEMENT OVER RANDOM-RANDOM")
    print(f"{'='*110}")
    for budget in sorted(results.keys()):
        rr = [r['test_mse'] for r in results[budget].get('random_random', [])
              if 'test_mse' in r and not np.isnan(r['test_mse'])]
        if not rr:
            continue
        rr_mu, rr_sd = np.mean(rr), np.std(rr)
        for m in [x for x in results[budget] if x != 'random_random']:
            al = [r['test_mse'] for r in results[budget][m]
                  if 'test_mse' in r and not np.isnan(r['test_mse'])]
            if not al:
                continue
            al_mu, al_sd = np.mean(al), np.std(al)
            imp = (rr_mu - al_mu) / rr_mu * 100
            sig = "yes" if (al_mu + al_sd) < (rr_mu - rr_sd) else "no"
            lbl = results[budget][m][0].get('method', m) \
                  if results[budget][m] else m
            print(f"  {budget:>4}  {lbl:<26} "
                  f"{al_mu:.4e} vs RR {rr_mu:.4e}  {imp:>+7.1f}%  {sig}")
        print()
    print(f"{'='*110}")


def aggregate(results, metric='test_mse'):
    stats = {}
    for method in list(next(iter(results.values())).keys()):
        blist, means, stds, allv = [], [], [], []
        for b in sorted(results.keys()):
            vals = [r[metric] for r in results[b].get(method, [])
                    if metric in r and not np.isnan(float(r.get(metric, float('nan'))))]
            if vals:
                blist.append(b); means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals))); allv.append(vals)
        stats[method] = {'budgets': blist, 'mean': means,
                         'std': stds, 'all': allv}
    return stats


# =======================================================================
# JSON HELPERS
# =======================================================================
def _save_results(results, path):
    def _c(o):
        if isinstance(o, dict):  return {k: _c(v) for k, v in o.items()}
        if isinstance(o, list):  return [_c(i) for i in o]
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(path, 'w') as f:
        json.dump(_c(results), f, indent=2)
    print(f'saved {path}')


def _load_results(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# =======================================================================
# CLI
# =======================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--quick',     action='store_true')
    p.add_argument('--budgets',   nargs='+', type=int, default=None)
    p.add_argument('--seeds',     type=int, default=None)
    p.add_argument('--unbiased',  action='store_true',
                   help='LHS pool - AL should ~ RR (sanity check)')
    p.add_argument('--no-seq-random', action='store_true')
    p.add_argument('--plot-only', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = copy.deepcopy(SWEEP_CONFIG)

    if args.quick:
        cfg['budgets']         = [50, 100, 200]
        cfg['seeds']           = [42]
        cfg['epochs_per_round']= 80
        cfg['final_epochs']    = 200
        cfg['patience']        = 50
        print("Quick: 1 seed x 3 budgets x 80ep/round")

    if args.budgets:       cfg['budgets']  = sorted(args.budgets)
    if args.seeds:         cfg['seeds']    = SWEEP_CONFIG['seeds'][:args.seeds]
    if args.unbiased:      cfg['biased_pool'] = False
    if args.no_seq_random: cfg['methods']  = ['sequential_al', 'random_random']

    n_rounds_max = (max(cfg['budgets']) - cfg['n_initial']) // cfg['k_per_round']
    total = len(cfg['budgets']) * len(cfg['methods']) * len(cfg['seeds'])

    print(f"\n{'='*80}")
    print(f"CSTR sequential AL sweep  "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"  Budgets : {cfg['budgets']}")
    print(f"  Seeds   : {cfg['seeds']}")
    print(f"  Methods : {cfg['methods']}")
    print(f"  Seq     : n0={cfg['n_initial']} k={cfg['k_per_round']} "
          f"max_rounds={n_rounds_max}")
    print(f"  Pool    : {'BIASED 70%' if cfg['biased_pool'] else 'LHS unbiased'}")
    print(f"  Epochs  : {cfg['epochs_per_round']}/round + "
          f"{cfg['final_epochs']} final")
    print(f"  Runs    : {total}")
    print(f"{'='*80}")

    if args.plot_only:
        results = _load_results(args.plot_only)
        print_sweep_table(results)
        generate_sweep_visualizations(results, cfg)
        return

    results = run_sweep(cfg)
    print_sweep_table(results)
    print("\nGenerating visualisations ...")
    generate_sweep_visualizations(results, cfg)
    print(f"\n[ok] Done - {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == '__main__':
    main()
