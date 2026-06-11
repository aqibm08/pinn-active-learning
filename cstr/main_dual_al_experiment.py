"""
MAIN DUAL ACTIVE LEARNING EXPERIMENT
=====================================
Compares 4 approaches:
1. Full AL: AL data + AL collocation
2. Random-Random: Random data + Random collocation
3. AL-Random: AL data + Random collocation
4. Random-AL: Random data + AL collocation

"""

import numpy as np
import torch
import torch.nn as nn
from scipy import io
import sys
import os
import copy

# Import custom modules
from active_learning_core import ActiveLearningSelector
from data_point_selector import DataPointSelector, RandomDataPointSelector
from dual_al_trainer import DualActiveLearningTrainer


# ============================================================================
# SETUP
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float64)

print(f"\n{'='*100}")
print(f"DUAL ACTIVE LEARNING EXPERIMENT (Data + Collocation)")
print(f"{'='*100}")
print(f"  Using device: {device}\n")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class EvidentialLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gamma_net = nn.Linear(input_dim, output_dim)
        self.nu_net = nn.Linear(input_dim, output_dim)
        self.alpha_net = nn.Linear(input_dim, output_dim)
        self.beta_net = nn.Linear(input_dim, output_dim)

        for net in [self.gamma_net, self.nu_net, self.alpha_net, self.beta_net]:
            nn.init.xavier_normal_(net.weight, gain=0.1)
            nn.init.constant_(net.bias, 0.01)

    def forward(self, x):
        gamma = self.gamma_net(x)
        nu = torch.nn.functional.softplus(self.nu_net(x)) + 1.0
        alpha = torch.nn.functional.softplus(self.alpha_net(x)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_net(x)) + 0.1
        return gamma, nu, alpha, beta


def _make_backbone(input_dim, hidden_dim, n_layers, activation='tanh'):
    """Build backbone MLP with configurable depth and activation"""
    act_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
    }
    Act = act_map.get(activation, nn.Tanh)

    layers = [nn.Linear(input_dim, hidden_dim), Act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), Act()]
    return nn.Sequential(*layers)


class DualHeadPhysicsPINN(nn.Module):
    """
    Dual-head PINN with configurable architecture.
    [FIX 6] hidden_dim and n_backbone_layers are now wired from config.
    """

    def __init__(self, input_dim=6, hidden_dim=256, output_dim=4,
                 n_backbone_layers=4, activation='tanh'):
        super().__init__()

        self.backbone = _make_backbone(input_dim, hidden_dim, n_backbone_layers, activation)

        self.deterministic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.evidential_head = EvidentialLayer(hidden_dim, output_dim)

        # Weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x):
        return self.backbone(x)

    def forward_deterministic(self, x):
        features = self.forward_features(x)
        return self.deterministic_head(features)

    def forward_evidential(self, x):
        features = self.forward_features(x)
        gamma, nu, alpha, beta = self.evidential_head(features)
        return gamma, nu, alpha, beta

    def forward(self, x):
        return self.forward_deterministic(x)

    def compute_pde_residuals(self, x_physics, y_norm_bounds, scale_factor=1.0):
        """
        Compute STEADY-STATE CSTR algebraic residuals.

        This data is STEADY-STATE (SS_data_*). Each sample is an independent
        operating point (Q_in, Qc_in, Ca0, T0, Tc0, h0) -> (Ca_ss, T_ss, Tc_ss, h_ss).
        There is NO time axis in x. The previous ODE formulation computed
        d(output)/d(x[:,0]) = d(Ca)/d(Q_in), which is physically meaningless
        and produced noise that dominated training.

        CORRECT formulation: At steady state d/dt = 0, so the CSTR ODEs become
        algebraic constraints f(x, y) = 0 that the model output must satisfy.
        These need NO autograd - just evaluate the formula with predicted outputs.

        Residuals:
          f1 = Q_in*(Caf-Ca)/(A*h) - k*exp(-E/RT)*Ca                     [mol/L/min]
          f2 = -H*rxn/Rho_Cp + Q_in*(Tf-T)/(A*h) + U_Ac*(Tc-T)/(Rho_Cp*A*h)  [K/min]
          f3 = Qc_in*(Tcf-Tc)/Vc + U_Ac*(T-Tc)/(Rhoc_Cpc*Vc)            [K/min]
          f4 = (Q_in - Cv*sqrt(h)) / A                                    [m/min]

        Each residual is normalised by a characteristic scale before squaring
        so all four contribute equally regardless of unit differences.
        """
        try:
            # -- denormalise inputs --------------------------------------------
            x_min_phys = torch.tensor([100.55, 10.02, 0.000410, 385.65, 338.15, 6.319032],
                                      device=x_physics.device, dtype=torch.float64)
            x_max_phys = torch.tensor([138.33, 19.95, 0.011763, 458.22, 380.99, 11.958720],
                                      device=x_physics.device, dtype=torch.float64)
            x_denorm = 0.5 * (x_physics + 1.0) * (x_max_phys - x_min_phys) + x_min_phys
            Q_in  = x_denorm[:, 0].detach()   # [L/min]
            Qc_in = x_denorm[:, 1].detach()   # [L/min]

            # -- get model predictions at collocation points -------------------
            # Use forward_deterministic so gradients flow through model params
            u_pred = self.forward_deterministic(x_physics)
            Ca_norm = u_pred[:, 0]
            T_norm  = u_pred[:, 1]
            Tc_norm = u_pred[:, 2]
            h_norm  = u_pred[:, 3]

            # -- denormalise outputs -------------------------------------------
            Ca_min, Ca_max = y_norm_bounds['Ca']
            T_min,  T_max  = y_norm_bounds['T']
            Tc_min, Tc_max = y_norm_bounds['Tc']
            h_min,  h_max  = y_norm_bounds['h']

            Ca = 0.5 * (Ca_norm + 1.0) * (Ca_max - Ca_min) + Ca_min
            T  = 0.5 * (T_norm  + 1.0) * (T_max  - T_min)  + T_min
            Tc = 0.5 * (Tc_norm + 1.0) * (Tc_max - Tc_min) + Tc_min
            h  = 0.5 * (h_norm  + 1.0) * (h_max  - h_min)  + h_min

            # Safety clamps (prevent NaN in exp / sqrt)
            Ca = torch.clamp(Ca, min=1e-8,  max=0.5)
            T  = torch.clamp(T,  min=280.0, max=600.0)
            Tc = torch.clamp(Tc, min=250.0, max=500.0)
            h  = torch.clamp(h,  min=0.01,  max=20.0)

            # -- CSTR constants ------------------------------------------------
            EbyR     = 8750.0   # E/R  [K]
            k        = 7.2e10   # pre-exponential [1/min]
            Cv       = 40.0     # valve coeff [L^0.5/min]
            A        = 100.0    # tank cross-section [m^2  or dm^2]
            H        = -5e4     # heat of reaction [J/mol]
            Rho_Cp   = 239.0    # liquid densityxCp [J/(L*K)]
            Rhoc_Cpc = 4175.0   # coolant Rho*Cp [J/(L*K)]
            U_Ac     = 5e4      # UA [J/(min*K)]
            Vc       = 250.0    # coolant volume [L]
            Caf      = 1.0      # inlet Ca [mol/L]
            Tf       = 320.0    # inlet T [K]
            Tcf      = 300.0    # coolant inlet T [K]

            # -- reaction rate -------------------------------------------------
            exp_arg = torch.clamp(-EbyR / T, min=-100.0, max=0.0)
            rxn     = k * torch.exp(exp_arg) * Ca          # [mol/(L*min)]
            Q_out   = Cv * torch.sqrt(h)                   # [L/min]

            # -- algebraic steady-state residuals -----------------------------
            # f1: mass balance on Ca  [mol/(L*min)]
            f1 = Q_in * (Caf - Ca) / (A * h) - rxn

            # f2: energy balance on T  [J/(L*min)] -> divide by Rho_Cp for [K/min]
            f2 = ((-H * rxn) / Rho_Cp
                  + Q_in * (Tf - T)   / (A * h)
                  + U_Ac * (Tc - T)   / (Rho_Cp * A * h))

            # f3: energy balance on Tc  [K/min]
            f3 = Qc_in * (Tcf - Tc) / Vc + U_Ac * (T - Tc) / (Rhoc_Cpc * Vc)

            # f4: level balance  [L/(m^2*min)] = [m/min] effectively
            f4 = (Q_in - Q_out) / A

            # -- normalise each residual by its characteristic scale -----------
            # Prevents one equation from dominating due to unit differences.
            # Scales estimated from typical operating ranges.
            f1 = f1 / (1e-3 + 1e-4)          # ~O(1e-3 mol/L/min)
            f2 = f2 / (1e-1 + 1.0)            # ~O(1 K/min)
            f3 = f3 / (1e-1 + 0.5)            # ~O(0.5 K/min)
            f4 = f4 / (1e-1 + 0.1)            # ~O(0.1 m/min)

            return [f1, f2, f3, f4]

        except Exception as e:
            print(f"[warn]  SS physics residual failed: {e}")
            import traceback; traceback.print_exc()
            batch_size = x_physics.shape[0]
            zero = torch.zeros(batch_size, device=x_physics.device, dtype=torch.float64)
            return [zero, zero, zero, zero]


# ============================================================================
# DATA LOADING AND SPLITTING
# ============================================================================
def load_and_split_data(n_initial_labeled, n_val, n_test, seed=42):
    """
    Load data and split into labeled/unlabeled/val/test.

    [FIX 5] Uses a seeded generator for the permutation so that all experiments
    in a run see identical data splits (previously used global state which
    differed per-call depending on call order).

    Args:
        n_initial_labeled: Initial labeled training points
        n_val: Validation set size (fixed)
        n_test: Test set size (fixed)
        seed: Random seed for reproducibility

    Returns:
        All data splits and normalization info
    """
    print("Loading data...")

    try:
        data = io.loadmat("data/SS_data_ndata1000_sampT1_cstr_ForEncDec.mat")
    except FileNotFoundError:
        print("ERROR: data/SS_data_ndata1000_sampT1_cstr_ForEncDec.mat not found")
        sys.exit(1)

    x = torch.tensor(data['x'], dtype=torch.float64).to(device)
    y = torch.tensor(data['y'], dtype=torch.float64).to(device)

    print(f"[ok] Data loaded: X={x.shape}, Y={y.shape}")

    # Normalize
    x_min, x_max = x.min(dim=0)[0], x.max(dim=0)[0]
    y_min, y_max = y.min(dim=0)[0], y.max(dim=0)[0]

    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

    y_norm_bounds = {
        'Ca': (y_min[0].item(), y_max[0].item()),
        'T': (y_min[1].item(), y_max[1].item()),
        'Tc': (y_min[2].item(), y_max[2].item()),
        'h': (y_min[3].item(), y_max[3].item()),
    }

    # [FIX 5] Seeded permutation - same split for all experiments
    n_total = len(x_norm)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)

    # Fixed splits
    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val + n_test]

    # Training pool (labeled + unlabeled)
    train_pool_idx = indices[n_val + n_test:]

    # Initial labeled set
    labeled_idx = train_pool_idx[:n_initial_labeled]
    unlabeled_idx = train_pool_idx[n_initial_labeled:]

    # Create datasets
    x_labeled = x_norm[labeled_idx]
    y_labeled = y_norm[labeled_idx]

    x_unlabeled = x_norm[unlabeled_idx]
    y_unlabeled = y_norm[unlabeled_idx]

    x_val = x_norm[val_idx]
    y_val = y_norm[val_idx]

    x_test = x_norm[test_idx]
    y_test = y_norm[test_idx]

    # Store raw (unnormalized) test y for plotting true vs predicted
    y_test_raw = y[test_idx]

    print(f"\nData Split (seed={seed}):")
    print(f"   Initial Labeled: {len(x_labeled)}")
    print(f"   Unlabeled Pool: {len(x_unlabeled)}")
    print(f"   Validation: {len(x_val)}")
    print(f"   Test: {len(x_test)}")
    print(f"   Total: {len(x_labeled) + len(x_unlabeled) + len(x_val) + len(x_test)}")

    return (x_labeled, y_labeled, x_unlabeled, y_unlabeled,
            x_val, y_val, x_test, y_test, y_norm_bounds,
            y_min.cpu().numpy(), y_max.cpu().numpy(), y_test_raw.cpu().numpy())


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
def run_single_experiment(mode_name, use_data_al, use_collocation_al,
                          x_labeled, y_labeled, x_unlabeled, y_unlabeled,
                          x_val, y_val, x_test, y_test, y_norm_bounds,
                          config):
    """
    Run a single training experiment

    Args:
        mode_name: Name of the experiment
        use_data_al: Use AL for data selection
        use_collocation_al: Use AL for collocation selection
        [data splits]
        config: Configuration dictionary

    Returns:
        Trained model and trainer
    """
    print(f"\n{'='*100}")
    print(f"EXPERIMENT: {mode_name}")
    print(f"{'='*100}\n")

    # [FIX 6] Create model with config-driven architecture
    model = DualHeadPhysicsPINN(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        n_backbone_layers=config['model'].get('n_backbone_layers', 4),
        activation=config['model'].get('activation', 'tanh')
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")

    # Create data selector
    if use_data_al:
        data_selector = DataPointSelector(
            model=model,
            device=device,
            w_epistemic=config['data_al']['w_epistemic'],
            w_aleatoric=config['data_al']['w_aleatoric'],
            w_diversity=config['data_al']['w_diversity']
        )
    else:
        data_selector = RandomDataPointSelector(device=device)

    # Create collocation selector
    if use_collocation_al:
        collocation_selector = ActiveLearningSelector(
            model=model,
            device=device,
            y_norm_bounds=y_norm_bounds,
            w_physics=config['coll_al']['w_physics'],
            w_uncertainty=config['coll_al']['w_uncertainty'],
            w_gradient=config['coll_al']['w_gradient']
        )
    else:
        collocation_selector = None

    # Create trainer
    trainer = DualActiveLearningTrainer(
        model=model,
        x_labeled=x_labeled.clone(),
        y_labeled=y_labeled.clone(),
        x_unlabeled=x_unlabeled.clone(),
        y_unlabeled=y_unlabeled.clone(),
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        y_norm_bounds=y_norm_bounds,
        data_selector=data_selector,
        collocation_selector=collocation_selector,
        device=device,
        use_data_al=use_data_al,
        use_collocation_al=use_collocation_al
    )

    # Train
    trainer.train(
        epochs=config['training']['epochs'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        lambda_physics=config['training']['lambda_physics'],
        data_al_frequency=config['data_al']['frequency'],
        n_add_data=config['data_al']['n_add'],
        target_labeled=config['data_al']['target_size'],
        coll_al_frequency=config['coll_al']['frequency'],
        n_add_coll=config['coll_al']['n_add'],
        n_candidates_coll=config['coll_al']['n_candidates'],
        n_initial_coll=config['coll_al']['n_initial']
    )

    # Test
    test_loss, test_mse, test_mae = trainer.test()

    return model, trainer, test_loss, test_mse, test_mae


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def main(config):
    """
    Main experiment runner - trains all 4 approaches

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with all results
    """
    # Load and split data - use seeded split shared across all experiments
    split_seed = config.get('data_split_seed', 42)
    (x_labeled, y_labeled, x_unlabeled, y_unlabeled,
     x_val, y_val, x_test, y_test, y_norm_bounds,
     y_min_raw, y_max_raw, y_test_raw) = load_and_split_data(
        n_initial_labeled=config['data_al']['n_initial'],
        n_val=config['data_al']['n_val'],
        n_test=config['data_al']['n_test'],
        seed=split_seed
    )

    # Store shared metadata for visualization
    shared_data = {
        'x_test': x_test,
        'y_test': y_test,
        'y_test_raw': y_test_raw,
        'y_min_raw': y_min_raw,
        'y_max_raw': y_max_raw,
        'y_norm_bounds': y_norm_bounds,
        'output_names': ['Ca', 'T', 'Tc', 'h']
    }

    results = {}

    # ========================================================================
    # Experiment 1: Full AL (AL Data + AL Collocation)
    # ========================================================================
    if config['experiment']['run_full_al']:
        model, trainer, test_loss, test_mse, test_mae = run_single_experiment(
            mode_name="Full AL (Data + Collocation)",
            use_data_al=True,
            use_collocation_al=True,
            x_labeled=x_labeled, y_labeled=y_labeled,
            x_unlabeled=x_unlabeled, y_unlabeled=y_unlabeled,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            y_norm_bounds=y_norm_bounds,
            config=config
        )

        results['full_al'] = {
            'model': model,
            'trainer': trainer,
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae
        }

        if config['experiment']['save_models']:
            torch.save({
                'model_state': model.state_dict(),
                'trainer_history': {
                    'data_loss': trainer.data_loss_history,
                    'physics_loss': trainer.physics_loss_history,
                    'total_loss': trainer.total_loss_history,
                    'val_loss': trainer.val_loss_history,
                    'val_mse': trainer.val_mse_history,
                    'train_size': trainer.train_size_history,
                    'collocation_size': trainer.collocation_size_history,
                    'uncertainty': trainer.uncertainty_history,
                },
                'config': config
            }, config['experiment']['path_full_al'])
            print(f"Saved: {config['experiment']['path_full_al']}\n")

    # ========================================================================
    # Experiment 2: Random-Random
    # ========================================================================
    if config['experiment']['run_random_random']:
        model, trainer, test_loss, test_mse, test_mae = run_single_experiment(
            mode_name="Random-Random (Baseline)",
            use_data_al=False,
            use_collocation_al=False,
            x_labeled=x_labeled, y_labeled=y_labeled,
            x_unlabeled=x_unlabeled, y_unlabeled=y_unlabeled,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            y_norm_bounds=y_norm_bounds,
            config=config
        )

        results['random_random'] = {
            'model': model,
            'trainer': trainer,
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae
        }

        if config['experiment']['save_models']:
            torch.save({
                'model_state': model.state_dict(),
                'trainer_history': {
                    'data_loss': trainer.data_loss_history,
                    'physics_loss': trainer.physics_loss_history,
                    'total_loss': trainer.total_loss_history,
                    'val_loss': trainer.val_loss_history,
                    'val_mse': trainer.val_mse_history,
                    'train_size': trainer.train_size_history,
                    'collocation_size': trainer.collocation_size_history,
                    'uncertainty': trainer.uncertainty_history,
                },
                'config': config
            }, config['experiment']['path_random_random'])
            print(f"Saved: {config['experiment']['path_random_random']}\n")

    # ========================================================================
    # Experiment 3: AL-Random (AL Data + Random Collocation)
    # ========================================================================
    if config['experiment']['run_al_random']:
        model, trainer, test_loss, test_mse, test_mae = run_single_experiment(
            mode_name="AL Data + Random Collocation",
            use_data_al=True,
            use_collocation_al=False,
            x_labeled=x_labeled, y_labeled=y_labeled,
            x_unlabeled=x_unlabeled, y_unlabeled=y_unlabeled,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            y_norm_bounds=y_norm_bounds,
            config=config
        )

        results['al_random'] = {
            'model': model,
            'trainer': trainer,
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae
        }

        if config['experiment']['save_models']:
            torch.save({
                'model_state': model.state_dict(),
                'trainer_history': {
                    'data_loss': trainer.data_loss_history,
                    'physics_loss': trainer.physics_loss_history,
                    'total_loss': trainer.total_loss_history,
                    'val_loss': trainer.val_loss_history,
                    'val_mse': trainer.val_mse_history,
                    'train_size': trainer.train_size_history,
                    'collocation_size': trainer.collocation_size_history,
                    'uncertainty': trainer.uncertainty_history,
                },
                'config': config
            }, config['experiment']['path_al_random'])
            print(f"Saved: {config['experiment']['path_al_random']}\n")

    # ========================================================================
    # Experiment 4: Random-AL (Random Data + AL Collocation)
    # ========================================================================
    if config['experiment']['run_random_al']:
        model, trainer, test_loss, test_mse, test_mae = run_single_experiment(
            mode_name="Random Data + AL Collocation",
            use_data_al=False,
            use_collocation_al=True,
            x_labeled=x_labeled, y_labeled=y_labeled,
            x_unlabeled=x_unlabeled, y_unlabeled=y_unlabeled,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            y_norm_bounds=y_norm_bounds,
            config=config
        )

        results['random_al'] = {
            'model': model,
            'trainer': trainer,
            'test_loss': test_loss,
            'test_mse': test_mse,
            'test_mae': test_mae
        }

        if config['experiment']['save_models']:
            torch.save({
                'model_state': model.state_dict(),
                'trainer_history': {
                    'data_loss': trainer.data_loss_history,
                    'physics_loss': trainer.physics_loss_history,
                    'total_loss': trainer.total_loss_history,
                    'val_loss': trainer.val_loss_history,
                    'val_mse': trainer.val_mse_history,
                    'train_size': trainer.train_size_history,
                    'collocation_size': trainer.collocation_size_history,
                    'uncertainty': trainer.uncertainty_history,
                },
                'config': config
            }, config['experiment']['path_random_al'])
            print(f"Saved: {config['experiment']['path_random_al']}\n")

    # ========================================================================
    # Attach shared data to results for downstream visualization
    # ========================================================================
    results['_shared'] = shared_data

    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*100)
    print("FINAL COMPARISON - ALL APPROACHES")
    print("="*100)

    print(f"\n{'Approach':<30} {'Test Loss':>15} {'Test MSE':>15} {'Test MAE':>15}")
    print("-" * 78)

    approach_display = {
        'full_al': 'Full AL',
        'random_random': 'Random-Random',
        'al_random': 'AL Data + Random Coll',
        'random_al': 'Random Data + AL Coll'
    }

    for key in ['full_al', 'random_random', 'al_random', 'random_al']:
        if key in results:
            print(f"{approach_display[key]:<30} {results[key]['test_loss']:>15.6e} "
                  f"{results[key]['test_mse']:>15.6e} {results[key]['test_mae']:>15.6e}")

    # Compute improvements
    if 'full_al' in results and 'random_random' in results:
        loss_imp = ((results['random_random']['test_loss'] - results['full_al']['test_loss']) /
                    results['random_random']['test_loss']) * 100
        mse_imp = ((results['random_random']['test_mse'] - results['full_al']['test_mse']) /
                   results['random_random']['test_mse']) * 100
        mae_imp = ((results['random_random']['test_mae'] - results['full_al']['test_mae']) /
                   results['random_random']['test_mae']) * 100

        print(f"\nFULL AL vs RANDOM-RANDOM IMPROVEMENTS:")
        print(f"   Test Loss: {loss_imp:+.2f}%")
        print(f"   Test MSE:  {mse_imp:+.2f}%")
        print(f"   Test MAE:  {mae_imp:+.2f}%")

    print("\n" + "="*100 + "\n")

    return results


if __name__ == '__main__':
    pass
