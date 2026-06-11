"""
True-vs-predicted figures for the PSA adsorption surrogate (b=60, seed 42,
clustered pool): time profiles at a fixed bed position, spatial profiles at
a fixed time, and a full-test-set scatter per output channel.

Training is identical to the headline sweep (it reuses run_sequential and
the sweep config from the psa folder). The trained state is cached in
<repo>/checkpoints, so the figures regenerate in seconds once the
checkpoint exists; delete it to retrain (~20 min on a workstation GPU).
"""
import json, copy
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from _paths import FIGS, PSA_DATA, CKPT_DIR, add_psa_to_path
from plot_style import apply_paper_style, COL_AL, COL_RAND_RANDOM

add_psa_to_path()
from run_psa_budget_sweep_v7 import prepare_data, run_sequential, SWEEP_CONFIG


apply_paper_style()
OUT = FIGS; OUT.mkdir(parents=True, exist_ok=True)

OUTPUT_LABELS = {
    0: (r'CO$_2$ mole fraction  $y_1$',  r'$y_1$'),
    1: (r'Pressure  $P/P_0$',            r'$P$'),
    2: (r'CO$_2$ loading  $q_1$',        r'$q_1$'),
    3: (r'N$_2$ loading  $q_2$',         r'$q_2$'),
}


def train_one_model():
    """Train a fresh PINN+AL model with the headline config."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = copy.deepcopy(SWEEP_CONFIG)
    cfg['step'] = 'ads'
    cfg.update(epochs_per_round=120, final_epochs=300, warmup_epochs=120,
                biased_pool=True)
    print('Loading data and applying clustered-pool construction...')
    splits, lb, ub = prepare_data(cfg, str(PSA_DATA))

    print('\nTraining PINN+AL at b=60, seed=42, ads, clustered pool...')
    res = run_sequential('pinn_al', 60, cfg, splits, lb, ub, seed=42,
                          device=device)
    return res, splits, device


def get_predictions(model, x_test, y_test, y_lb, y_ub, device):
    """Run the trained model on the test set, denormalise outputs."""
    model.eval()
    with torch.no_grad():
        y_hat_norm = model.forward_deterministic(
            x_test.double().to(device)).cpu()
    # Denormalise to physical units
    y_hat = y_hat_norm * (y_ub - y_lb) + y_lb
    return y_hat.numpy(), y_test.numpy()


def find_representative_scenario(x_test, y_test, y_hat, z_target=0.2):
    """Pick a test scenario with visible CO2-front dynamics at the chosen z
    AND roughly-median overall MSE.  Scenarios are scored by (a) y1 variance
    over time at z=z_target, (b) y1 RANGE at z_target.  We then pick a
    representative scenario among the well-dynamics ones by median MSE.
    """
    x = x_test.numpy()
    fp = np.round(x[:, 2:12], 3)
    _, unique_idx, inv = np.unique(fp, axis=0, return_index=True,
                                     return_inverse=True)
    n_scens = len(unique_idx)
    yt_np = y_test.numpy()

    metrics = []
    for k in range(n_scens):
        m = inv == k
        scen_mse = ((y_hat[m] - yt_np[m]) ** 2).mean()
        z_rows = m & (np.abs(x[:, 0] - z_target) < 0.04)
        if z_rows.sum() < 3:
            y1_range = 0.0
        else:
            y1s = yt_np[z_rows, 0]
            y1_range = float(y1s.max() - y1s.min())
        metrics.append((k, scen_mse, y1_range))

    # Require visible front dynamics: y1 swings by at least ~30% of max range
    max_range = max(m[2] for m in metrics)
    threshold = max_range * 0.3
    print(f'  z={z_target}: max y1 range across scenarios = {max_range:.3f}; '
          f'threshold {threshold:.3f}')
    dynamic = [m for m in metrics if m[2] > threshold]
    if not dynamic:
        dynamic = sorted(metrics, key=lambda m: -m[2])[:max(1, n_scens // 4)]
    dynamic.sort(key=lambda m: m[1])
    chosen = dynamic[len(dynamic) // 2]
    k = chosen[0]
    print(f'  Picked scenario {k}/{n_scens}  '
          f'(MSE {chosen[1]:.3e}, y1 range @ z={z_target}: {chosen[2]:.3e})')
    return inv == k


def fig_timeseries(x_test_np, y_test, y_hat, mask, name, z_target=0.2):
    """4-panel time series at fixed z for one scenario."""
    xs = x_test_np[mask]
    ys_true = y_test[mask]
    ys_pred = y_hat[mask]
    z_vals = xs[:, 0]
    z_unique = np.unique(np.round(z_vals, 5))
    z_pick = z_unique[np.argmin(np.abs(z_unique - z_target))]
    z_mask = np.abs(z_vals - z_pick) < 1e-5
    t = xs[z_mask, 1]
    order = np.argsort(t)
    t = t[order]
    yt = ys_true[z_mask][order]
    yp = ys_pred[z_mask][order]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for k, ax in enumerate(axes.flatten()):
        mse_k = ((yp[:, k] - yt[:, k]) ** 2).mean()
        ax.plot(t, yt[:, k], '-', color='#444444', linewidth=3.0,
                 label='Simulator (truth)', alpha=0.85)
        ax.plot(t, yp[:, k], '--', color=COL_AL, linewidth=2.4,
                 label='PINN+AL prediction', dashes=(4, 2))
        ax.set_xlabel(r'Dimensionless time  $\tau$ (s)')
        ax.set_ylabel(OUTPUT_LABELS[k][0])
        ax.text(0.03, 0.96, f'MSE = {mse_k:.2e}',
                 transform=ax.transAxes, fontsize=11, va='top',
                 bbox=dict(facecolor='white', edgecolor='#888',
                            alpha=0.92, pad=4))
        if k == 0:
            ax.legend(loc='lower right')
    #fig.suptitle(f'Time profiles at fixed $\\zeta \\approx {z_pick:.2f}$ '
    #             f'(single test scenario)', y=1.00)
    fig.tight_layout()
    out = OUT / f'{name}.png'
    fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig_spatial(x_test_np, y_test, y_hat, mask, name):
    """4-panel spatial profile at a fixed mid-range t for one scenario."""
    xs = x_test_np[mask]
    ys_true = y_test[mask]
    ys_pred = y_hat[mask]

    t_vals = xs[:, 1]
    t_unique = np.unique(np.round(t_vals, 5))
    # Pick a time partway through the scenario (around the 50% mark)
    t_pick = t_unique[len(t_unique) // 2]
    t_mask = np.abs(t_vals - t_pick) < 1e-5
    z = xs[t_mask, 0]
    order = np.argsort(z)
    z = z[order]
    yt = ys_true[t_mask][order]
    yp = ys_pred[t_mask][order]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for k, ax in enumerate(axes.flatten()):
        mse_k = ((yp[:, k] - yt[:, k]) ** 2).mean()
        ax.plot(z, yt[:, k], '-', color='#444444', linewidth=3.0,
                 label='Simulator (truth)', alpha=0.85)
        ax.plot(z, yp[:, k], '--', color=COL_AL, linewidth=2.4,
                 label='PINN+AL prediction', dashes=(4, 2))
        ax.set_xlabel(r'Dimensionless position  $\zeta$')
        ax.set_ylabel(OUTPUT_LABELS[k][0])
        ax.text(0.03, 0.96, f'MSE = {mse_k:.2e}',
                 transform=ax.transAxes, fontsize=11, va='top',
                 bbox=dict(facecolor='white', edgecolor='#888',
                            alpha=0.92, pad=4))
        if k == 0:
            ax.legend(loc='lower right')
    #fig.suptitle(f'Spatial profiles at fixed $\\tau \\approx {t_pick:.1f}$ s '
    #             f'(single test scenario)', y=1.00)
    fig.tight_layout()
    out = OUT / f'{name}.png'
    fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig_scatter(y_test, y_hat, name):
    """4-panel true-vs-predicted scatter across all test points."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for k, ax in enumerate(axes.flatten()):
        yt = y_test[:, k]; yp = y_hat[:, k]
        mse_k = ((yp - yt) ** 2).mean()
        r2 = 1.0 - ((yp - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        # Scatter with low alpha for density visibility
        ax.scatter(yt, yp, s=8, alpha=0.35, color=COL_AL,
                    edgecolors='none')
        # Identity line
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                 'k-', linewidth=1.5, alpha=0.7)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f'Simulator  {OUTPUT_LABELS[k][1]}  (truth)')
        ax.set_ylabel(f'PINN+AL  {OUTPUT_LABELS[k][1]}  (predicted)')
        ax.set_aspect('equal', adjustable='box')
        ax.text(0.03, 0.96,
                 f'MSE = {mse_k:.2e}\n$R^2$ = {r2:.4f}',
                 transform=ax.transAxes, fontsize=11, va='top',
                 bbox=dict(facecolor='white', edgecolor='#888',
                            alpha=0.92, pad=4))
    #fig.suptitle('True vs predicted on internal-test set (all rows, all scenarios)',
    #              y=1.00)
    fig.tight_layout()
    out = OUT / f'{name}.png'
    fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def main():
    from psa_pinn_model_v3 import build_model
    from psa_data_selector_v5 import PSADataSelectorV5
    from psa_collocation_selector_v3 import PSACollocationSelectorV3
    from psa_al_trainer_v3 import PSAActiveLearningTrainerV3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = copy.deepcopy(SWEEP_CONFIG)
    cfg['step'] = 'ads'
    cfg.update(epochs_per_round=120, final_epochs=300, warmup_epochs=120,
                biased_pool=True)
    print('Loading data and applying clustered-pool construction...')
    splits, lb, ub = prepare_data(cfg, str(PSA_DATA))

    ckpt = CKPT_DIR / 'truepred_model_b60_seed42_ads.pt'
    model = build_model(60, lb.numpy(), ub.numpy(), device,
                         model_type='pinn',
                         pca_components=splits['pca_components'],
                         pca_mean=splits['pca_mean'],
                         n_committee=3, init_seed=42,
                         y_lb=splits['y_lb'], y_ub=splits['y_ub'],
                         t_max=splits['t_max'])

    if ckpt.exists():
        print(f'Loading cached model state from {ckpt}')
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                           weights_only=True))
    else:
        print('\nTraining PINN+AL at b=60, seed=42, ads, clustered pool '
              '(~22 min)...')
        data_sel = PSADataSelectorV5(mode='badge', k_select=4, device=device)
        coll_sel = PSACollocationSelectorV3(n_candidates=1500, coll_multiplier=4,
                                              use_manifold=True, device=device)
        trainer = PSAActiveLearningTrainerV3(
            model=model, data_selector=data_sel, coll_selector=coll_sel,
            n0=4, k_per_round=4, budget=60,
            epochs_per_round=120, n_final_epochs=300, warmup_epochs=120,
            lambda_pde_max=0.002, lam_evid=0.05,
            lr=1e-3, weight_decay=1e-4, device=device, verbose=False,
            train_committee=True)
        trainer.run(splits['pool_scenarios'],
                     x_pool=splits['x_pool'],
                     x_pool_orig=splits['x_pool_orig'],
                     y_pool=splits['y_pool'],
                     x_val=splits['x_val'], y_val=splits['y_val'], seed=42)
        torch.save(model.state_dict(), ckpt)
        print(f'Saved model state to {ckpt}')

    print('\nGenerating true-vs-predicted figures...')
    y_hat, y_test_arr = get_predictions(
        model, splits['x_test'], splits['y_test'],
        splits['y_lb'], splits['y_ub'], device)

    x_test_np = splits['x_test'].numpy()
    mask = find_representative_scenario(splits['x_test'], splits['y_test'], y_hat)

    fig_timeseries(x_test_np, y_test_arr, y_hat, mask, 'truepred_timeseries')
    fig_spatial(x_test_np, y_test_arr, y_hat, mask, 'truepred_spatial')
    fig_scatter(y_test_arr, y_hat, 'truepred_scatter')


if __name__ == '__main__':
    main()
