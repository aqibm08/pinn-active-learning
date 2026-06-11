"""
DUAL ACTIVE LEARNING TRAINER
=============================
Handles both Data AL and Collocation AL simultaneously.

Supports 4 training modes:
1. Full AL: AL data + AL collocation
2. Random-Random: Random data + Random collocation
3. AL-Random: AL data + Random collocation
4. Random-AL: Random data + AL collocation

"""

import numpy as np
import torch
import torch.optim as optim
import time
import copy
from scipy.stats.qmc import LatinHypercube as LHC


def evidential_loss(gamma, nu, alpha, beta, targets, lam=0.01):
    """NIG loss function"""
    nu = torch.clamp(nu, min=1.0, max=1e6)
    alpha = torch.clamp(alpha, min=1.0, max=1e6)
    beta = torch.clamp(beta, min=0.1, max=1e6)

    error = (targets - gamma) ** 2
    log_2pi = np.log(2 * np.pi)

    nll = (1/2) * log_2pi - (1/2) * torch.log(nu) \
        + (alpha + 1/2) * torch.log(beta + nu * error / 2) \
        - alpha * torch.log(beta)

    reg = lam * (2 * nu + alpha)
    loss = nll + reg

    return torch.mean(loss)


class DualActiveLearningTrainer:
    """
    Trainer with Active Learning for both data and collocation points
    """

    def __init__(self, model,
                 x_labeled, y_labeled, x_unlabeled, y_unlabeled,
                 x_val, y_val, x_test, y_test,
                 y_norm_bounds,
                 data_selector, collocation_selector, device,
                 use_data_al=True, use_collocation_al=True,
                 max_patience=300):
        """
        Args:
            model: DualHeadPhysicsPINN model
            x_labeled, y_labeled: Initial labeled training data
            x_unlabeled, y_unlabeled: Unlabeled pool for data AL
            x_val, y_val: Validation data (fixed)
            x_test, y_test: Test data (fixed)
            y_norm_bounds: Normalization bounds
            data_selector: DataPointSelector or RandomDataPointSelector
            collocation_selector: ActiveLearningSelector or None (for random)
            device: torch device
            use_data_al: Whether to use AL for data selection
            use_collocation_al: Whether to use AL for collocation selection
        """
        self.model = model

        # Training data (will grow)
        self.x_train = x_labeled
        self.y_train = y_labeled

        # Unlabeled pool (will shrink)
        self.x_unlabeled = x_unlabeled
        self.y_unlabeled = y_unlabeled

        # Fixed validation and test
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.y_norm_bounds = y_norm_bounds
        self.data_selector = data_selector
        self.collocation_selector = collocation_selector
        self.device = device

        self.use_data_al = use_data_al
        self.use_collocation_al = use_collocation_al
        self.max_patience = max_patience

        # Collocation points (will grow)
        self.x_physics = None

        # Core loss history
        self.data_loss_history = []
        self.physics_loss_history = []
        self.total_loss_history = []
        self.val_loss_history = []
        self.val_mse_history = []
        self.best_val_loss = float('inf')
        self.best_model_state = None  # [FIX 1] will be deep-copied

        # Per-output metrics history (Ca, T, Tc, h)
        self.output_names = ['Ca', 'T', 'Tc', 'h']
        self.val_mse_per_output_history = {name: [] for name in self.output_names}
        self.val_mae_per_output_history = {name: [] for name in self.output_names}

        # Uncertainty history
        self.uncertainty_history = []

        # AL-specific tracking
        self.data_al_iterations = []
        self.collocation_al_iterations = []
        self.data_score_info_history = []
        self.collocation_score_info_history = []
        self.train_size_history = []
        self.unlabeled_size_history = []
        self.collocation_size_history = []

        # Timing
        self.al_iteration_times = []  # (epoch, data_time, coll_time)
        self.total_training_time = 0.0

        mode_name = self._get_mode_name()
        print(f"\nDualActiveLearningTrainer initialized ({mode_name})")
        print(f"   Initial labeled: {len(x_labeled)}, Unlabeled pool: {len(x_unlabeled)}")
        print(f"   Val: {len(x_val)}, Test: {len(x_test)}")

    def _get_mode_name(self):
        """Get training mode name"""
        if self.use_data_al and self.use_collocation_al:
            return "Full AL"
        elif not self.use_data_al and not self.use_collocation_al:
            return "Random-Random"
        elif self.use_data_al and not self.use_collocation_al:
            return "AL Data + Random Coll"
        else:
            return "Random Data + AL Coll"

    def add_data_points(self, n_add, epoch, target_size, adaptive_k=True):
        """
        Add new labeled data points via AL or random selection

        Args:
            n_add: Number of points to add
            epoch: Current epoch
            target_size: Stop growing if train set reaches this size
            adaptive_k: Use adaptive K for clustering
        """
        # Check if we've reached target size
        if len(self.x_train) >= target_size:
            return

        # Check if unlabeled pool is empty
        if len(self.x_unlabeled) == 0:
            print(f"\n[warn]  [Epoch {epoch}] Unlabeled pool exhausted. Cannot add more data.")
            return

        # Adjust n_add if needed
        n_add = min(n_add, len(self.x_unlabeled), target_size - len(self.x_train))

        t0 = time.time()

        if self.use_data_al:
            print(f"\n[Epoch {epoch}] Adding {n_add} data points via AL...")
            self.data_selector.update_labeled_points(self.x_train)
            selected_x, selected_y, selected_indices, score_info = self.data_selector.select_data_points(
                self.x_unlabeled, self.y_unlabeled, n_select=n_add,
                adaptive_k=adaptive_k, verbose=True
            )
            self.data_score_info_history.append(score_info)
        else:
            print(f"\n[Epoch {epoch}] Adding {n_add} data points randomly...")
            selected_x, selected_y, selected_indices, score_info = self.data_selector.select_data_points(
                self.x_unlabeled, self.y_unlabeled, n_select=n_add, verbose=True
            )
            self.data_score_info_history.append(score_info)

        data_time = time.time() - t0

        # Add to training set
        self.x_train = torch.cat([self.x_train, selected_x], dim=0)
        self.y_train = torch.cat([self.y_train, selected_y], dim=0)

        # [FIX 2] Ensure selected_indices is a LongTensor for consistent masking
        if isinstance(selected_indices, np.ndarray):
            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)
        elif isinstance(selected_indices, torch.Tensor):
            selected_indices_tensor = selected_indices.long().to(self.device)
        else:
            selected_indices_tensor = torch.tensor(list(selected_indices), dtype=torch.long, device=self.device)

        mask = torch.ones(len(self.x_unlabeled), dtype=torch.bool, device=self.device)
        mask[selected_indices_tensor] = False
        self.x_unlabeled = self.x_unlabeled[mask]
        self.y_unlabeled = self.y_unlabeled[mask]

        self.data_al_iterations.append(epoch)
        self.al_iteration_times.append({'epoch': epoch, 'data_time': data_time, 'coll_time': 0.0})

        print(f"   [ok] Training set: {len(self.x_train)}, Unlabeled pool: {len(self.x_unlabeled)}")

    def generate_random_collocation_points(self, n_points):
        """Generate random collocation points using LHS"""
        sampler = LHC(d=6)
        samples = sampler.random(n=n_points)
        x_physics = 2 * torch.tensor(samples, dtype=torch.float64, device=self.device) - 1.0
        return x_physics

    def add_collocation_points(self, n_add, n_candidates, epoch, adaptive_k=True):
        """
        Add new collocation points via AL or random selection

        Args:
            n_add: Number of points to add
            n_candidates: Candidate pool size (for AL)
            epoch: Current epoch
            adaptive_k: Use adaptive K for clustering
        """
        t0 = time.time()

        if self.use_collocation_al and self.collocation_selector is not None:
            print(f"\n[Epoch {epoch}] Adding {n_add} collocation points via AL...")
            selected_points, score_info = self.collocation_selector.select_collocation_points(
                n_candidates=n_candidates,
                n_select=n_add,
                adaptive_k=adaptive_k,
                seed=None,
                verbose=True
            )
            self.collocation_score_info_history.append(score_info)
        else:
            print(f"\n[Epoch {epoch}] Adding {n_add} collocation points randomly...")
            selected_points = self.generate_random_collocation_points(n_add)

        coll_time = time.time() - t0

        # Add to collocation set
        if self.x_physics is None:
            self.x_physics = selected_points
        else:
            self.x_physics = torch.cat([self.x_physics, selected_points], dim=0)

        self.collocation_al_iterations.append(epoch)

        # Update timing record (merge with data record if same epoch)
        if self.al_iteration_times and self.al_iteration_times[-1]['epoch'] == epoch:
            self.al_iteration_times[-1]['coll_time'] = coll_time
        else:
            self.al_iteration_times.append({'epoch': epoch, 'data_time': 0.0, 'coll_time': coll_time})

        print(f"   [ok] Total collocation points: {len(self.x_physics)}")

    def compute_data_loss(self):
        """Compute data fitting loss"""
        gamma, nu, alpha, beta = self.model.forward_evidential(self.x_train)
        loss = evidential_loss(gamma, nu, alpha, beta, self.y_train)
        return loss

    def compute_physics_loss(self):
        """Compute physics residual loss"""
        if self.x_physics is None or len(self.x_physics) == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        residuals = self.model.compute_pde_residuals(self.x_physics, self.y_norm_bounds)

        physics_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        valid_residuals = 0

        for residual in residuals:
            if residual is not None:
                res_mse = torch.mean(residual ** 2)
                physics_loss = physics_loss + res_mse
                valid_residuals += 1

        if valid_residuals > 0:
            physics_loss = physics_loss / valid_residuals

        return physics_loss

    def validate(self):
        """Validation with per-output metrics"""
        self.model.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = self.model.forward_evidential(self.x_val)
            val_loss = evidential_loss(gamma, nu, alpha, beta, self.y_val)

            predictions = gamma
            mse = torch.mean((predictions - self.y_val) ** 2)
            mae = torch.mean(torch.abs(predictions - self.y_val))

            # [FIX 3] Consistent epsilon in epistemic uncertainty formula
            epistemic_var = beta / (alpha * nu + 1e-8)
            total_uncertainty = torch.mean(epistemic_var)

            # Per-output metrics
            per_output_mse = []
            per_output_mae = []
            for i in range(self.y_val.shape[1]):
                mse_i = torch.mean((predictions[:, i] - self.y_val[:, i]) ** 2).item()
                mae_i = torch.mean(torch.abs(predictions[:, i] - self.y_val[:, i])).item()
                per_output_mse.append(mse_i)
                per_output_mae.append(mae_i)

        self.model.train()
        return (val_loss.item(), mse.item(), mae.item(),
                total_uncertainty.item(), per_output_mse, per_output_mae)

    def train(self, epochs, lr, weight_decay, lambda_physics,
              data_al_frequency, n_add_data, target_labeled,
              coll_al_frequency, n_add_coll, n_candidates_coll,
              n_initial_coll):
        """
        Main training loop with dual AL

        Args:
            epochs: Total training epochs
            lr: Learning rate
            weight_decay: Weight decay
            lambda_physics: Physics loss weight
            data_al_frequency: Add data points every N epochs
            n_add_data: Number of data points to add per iteration
            target_labeled: Stop growing training set at this size
            coll_al_frequency: Add collocation points every N epochs
            n_add_coll: Number of collocation points to add per iteration
            n_candidates_coll: Candidate pool size for collocation AL
            n_initial_coll: Initial number of collocation points
        """
        mode_name = self._get_mode_name()

        print(f"\n{'='*100}")
        print(f"TRAINING: {mode_name}")
        print(f"{'='*100}")
        print(f"   Total Epochs: {epochs}")
        print(f"   Learning Rate: {lr}")
        print(f"   Physics Loss Weight: {lambda_physics}")
        print(f"   Data AL Frequency: Every {data_al_frequency} epochs")
        print(f"   Collocation AL Frequency: Every {coll_al_frequency} epochs")
        print(f"   Initial Training Size: {len(self.x_train)}")
        print(f"   Target Training Size: {target_labeled}")
        print(f"   Initial Collocation: {n_initial_coll}")

        # Initialize collocation points
        print(f"\nInitializing collocation points...")
        self.add_collocation_points(n_initial_coll, n_candidates_coll, epoch=0, adaptive_k=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
            min_lr=1e-6
        )

        self.model.train()
        start_time = time.time()
        patience_counter = 0
        max_patience = self.max_patience

        for epoch in range(epochs):
            # Data AL: Add new training points
            if epoch > 0 and epoch % data_al_frequency == 0:
                self.add_data_points(n_add_data, epoch, target_labeled, adaptive_k=True)

            # Collocation AL: Add new physics points
            if epoch > 0 and epoch % coll_al_frequency == 0:
                self.add_collocation_points(n_add_coll, n_candidates_coll, epoch, adaptive_k=True)

            optimizer.zero_grad()

            # Data loss
            data_loss = self.compute_data_loss()

            # Physics loss
            physics_loss = self.compute_physics_loss()

            # Combined loss
            total_loss = data_loss + lambda_physics * physics_loss

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation (extended)
            val_loss, val_mse, val_mae, avg_uncertainty, per_out_mse, per_out_mae = self.validate()

            # Core history
            self.data_loss_history.append(data_loss.item())
            self.physics_loss_history.append(physics_loss.item())
            self.total_loss_history.append(total_loss.item())
            self.val_loss_history.append(val_loss)
            self.val_mse_history.append(val_mse)
            self.uncertainty_history.append(avg_uncertainty)

            # Per-output history
            for i, name in enumerate(self.output_names):
                self.val_mse_per_output_history[name].append(per_out_mse[i])
                self.val_mae_per_output_history[name].append(per_out_mae[i])

            # Size history
            self.train_size_history.append(len(self.x_train))
            self.unlabeled_size_history.append(len(self.x_unlabeled))
            self.collocation_size_history.append(len(self.x_physics) if self.x_physics is not None else 0)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # [FIX 1] Deep copy to avoid aliasing when model weights update later
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler.step(val_loss)

            # Progress
            if epoch % 50 == 0 or epoch == epochs - 1 or \
               epoch % data_al_frequency == 0 or epoch % coll_al_frequency == 0:
                elapsed = time.time() - start_time
                n_train = len(self.x_train)
                n_unlabeled = len(self.x_unlabeled)
                n_coll = len(self.x_physics) if self.x_physics is not None else 0

                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Data: {data_loss.item():9.4e} | "
                      f"Phys: {physics_loss.item():9.4e} | "
                      f"Total: {total_loss.item():9.4e} | "
                      f"Val: {val_loss:9.4e} | "
                      f"Train: {n_train:3d} | Unlab: {n_unlabeled:3d} | "
                      f"Coll: {n_coll:5d} | Pat: {patience_counter:3d}/{max_patience}")

            # Early stopping check
            if patience_counter >= max_patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        self.total_training_time = time.time() - start_time
        final_train = len(self.x_train)
        final_coll = len(self.x_physics) if self.x_physics is not None else 0

        print(f"\n{'='*100}")
        print(f"[ok] TRAINING COMPLETED ({mode_name}) in {self.total_training_time:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.6e}")
        print(f"Final training size: {final_train}")
        print(f"Final collocation size: {final_coll}")
        print(f"Data AL iterations: {len(self.data_al_iterations)}")
        print(f"Collocation AL iterations: {len(self.collocation_al_iterations)}")
        print(f"{'='*100}\n")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights\n")

    def test(self):
        """Test evaluation with per-output metrics"""
        mode_name = self._get_mode_name()

        self.model.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = self.model.forward_evidential(self.x_test)

            test_loss = evidential_loss(gamma, nu, alpha, beta, self.y_test)
            predictions = gamma
            test_mse = torch.mean((predictions - self.y_test) ** 2)
            test_mae = torch.mean(torch.abs(predictions - self.y_test))

            # [FIX 3] Consistent epsilon
            epistemic_var = beta / (alpha * nu + 1e-8)
            total_uncertainty = torch.mean(epistemic_var)

            # Per-output
            output_mse = []
            output_mae = []
            for i in range(self.y_test.shape[1]):
                mse_i = torch.mean((predictions[:, i] - self.y_test[:, i]) ** 2).item()
                mae_i = torch.mean(torch.abs(predictions[:, i] - self.y_test[:, i])).item()
                output_mse.append(mse_i)
                output_mae.append(mae_i)

        print(f"{'='*100}")
        print(f"TEST SET EVALUATION ({mode_name})")
        print(f"{'='*100}")
        print(f"Overall Test Loss: {test_loss.item():.6e}")
        print(f"Overall Test MSE:  {test_mse.item():.6e}")
        print(f"Overall Test MAE:  {test_mae.item():.6e}")
        print(f"Average Uncertainty: {total_uncertainty.item():.6e}")
        print(f"\nPer-Output Metrics:")
        for i, name in enumerate(self.output_names):
            print(f"  {name:3s}: MSE={output_mse[i]:.4e}, MAE={output_mae[i]:.4e}")
        print(f"{'='*100}\n")

        # Store for retrieval by results dict
        self.test_predictions = predictions.cpu().numpy()
        self.test_targets = self.y_test.cpu().numpy()
        self.test_epistemic = epistemic_var.cpu().numpy()
        self.test_output_mse = output_mse
        self.test_output_mae = output_mae

        return test_loss.item(), test_mse.item(), test_mae.item()
