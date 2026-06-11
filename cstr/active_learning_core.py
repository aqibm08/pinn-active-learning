"""
ACTIVE LEARNING CORE MODULE
===========================
Contains scoring functions and point selection logic for collocation points.

Three scoring mechanisms:
1. Physics-Based Scoring: PDE residual magnitudes
2. Uncertainty-Based Scoring: Epistemic uncertainty from evidential head
3. Gradient-Based Scoring : Per-point input-space gradient d(physics_loss)/dx
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats.qmc import LatinHypercube as LHC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class ActiveLearningSelector:
    """
    Active Learning point selection for Physics-Informed Neural Networks
    """

    def __init__(self, model, device, y_norm_bounds,
                 w_physics=1/3, w_uncertainty=1/3, w_gradient=1/3):
        """
        Args:
            model: DualHeadPhysicsPINN model
            device: torch device
            y_norm_bounds: Normalization bounds for outputs
            w_physics: Weight for physics-based score
            w_uncertainty: Weight for uncertainty-based score
            w_gradient: Weight for gradient-based score
        """
        self.model = model
        self.device = device
        self.y_norm_bounds = y_norm_bounds
        self.w_physics = w_physics
        self.w_uncertainty = w_uncertainty
        self.w_gradient = w_gradient

        print(f"ActiveLearningSelector initialized")
        print(f"   Weights: Physics={w_physics:.3f}, Uncertainty={w_uncertainty:.3f}, Gradient={w_gradient:.3f}")

    def generate_candidate_points(self, n_candidates=20000, seed=None):
        """
        Generate candidate collocation points using Latin Hypercube Sampling

        Args:
            n_candidates: Number of candidate points to generate
            seed: Random seed for reproducibility

        Returns:
            Tensor of candidate points in normalized space [-1, 1]
        """
        if seed is not None:
            sampler = LHC(d=6, seed=seed)
        else:
            sampler = LHC(d=6)

        candidates = sampler.random(n=n_candidates)
        x_candidates = 2 * torch.tensor(candidates, dtype=torch.float64, device=self.device) - 1.0

        return x_candidates

    def compute_physics_score(self, x_candidates, batch_size=4000):
        """
        Score based on PDE residual magnitudes (higher residual = more informative)

        Args:
            x_candidates: Candidate collocation points
            batch_size: Batch size for processing

        Returns:
            Physics scores (higher = more informative)
        """
        self.model.eval()
        n_candidates = x_candidates.shape[0]
        physics_scores = []

        for i in range(0, n_candidates, batch_size):
            # Algebraic SS residuals: no requires_grad needed on x
            batch = x_candidates[i:i + batch_size]

            # Compute SS algebraic residuals
            residuals = self.model.compute_pde_residuals(batch, self.y_norm_bounds)

            # Sum squared residuals per point across all 4 equations
            batch_scores = torch.zeros(batch.shape[0], device=self.device, dtype=torch.float64)
            valid_count = 0
            for residual in residuals:
                if residual is not None:
                    batch_scores = batch_scores + residual.detach() ** 2
                    valid_count += 1

            if valid_count > 0:
                batch_scores = batch_scores / valid_count
            physics_scores.append(batch_scores.detach())

        physics_scores = torch.cat(physics_scores, dim=0)
        return physics_scores

    def compute_uncertainty_score(self, x_candidates, batch_size=4000):
        """
        Score based on epistemic uncertainty from evidential head

        Args:
            x_candidates: Candidate collocation points
            batch_size: Batch size for processing

        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        self.model.eval()
        n_candidates = x_candidates.shape[0]
        uncertainty_scores = []

        with torch.no_grad():
            for i in range(0, n_candidates, batch_size):
                batch = x_candidates[i:i + batch_size]

                # Get evidential parameters
                gamma, nu, alpha, beta = self.model.forward_evidential(batch)

                # Epistemic uncertainty: beta / (alpha * nu)  [consistent formula]
                epistemic_var = beta / (alpha * nu + 1e-8)
                batch_uncertainty = torch.mean(epistemic_var, dim=1)

                uncertainty_scores.append(batch_uncertainty)

        uncertainty_scores = torch.cat(uncertainty_scores, dim=0)
        return uncertainty_scores

    def compute_gradient_score(self, x_candidates, batch_size=4000):
        """
        Per-point gradient score for SS algebraic residuals.

        Computes the L2 norm of d(sum f_i(x)^2)/dx for each candidate point.
        For algebraic SS residuals, this measures how rapidly the physics
        constraint changes with operating conditions - identifying regions
        of the input space where the model needs extra enforcement.

        Because the SS residuals go through forward_deterministic(x),
        we still need requires_grad on x for the chain-rule to work.
        """
        self.model.eval()
        n_candidates = x_candidates.shape[0]
        gradient_scores = []

        for i in range(0, n_candidates, batch_size):
            batch = x_candidates[i:i + batch_size].clone().detach().requires_grad_(True)

            residuals = self.model.compute_pde_residuals(batch, self.y_norm_bounds)

            per_point_loss = torch.zeros(batch.shape[0], device=self.device, dtype=torch.float64)
            valid_count = 0
            for residual in residuals:
                if residual is not None:
                    per_point_loss = per_point_loss + residual ** 2
                    valid_count += 1

            if valid_count > 0:
                per_point_loss = per_point_loss / valid_count

            total = per_point_loss.sum()
            grads = torch.autograd.grad(
                outputs=total,
                inputs=batch,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]   # [batch_size, 6]

            gradient_scores.append(torch.norm(grads, dim=1).detach())

        return torch.cat(gradient_scores, dim=0)

    def normalize_scores(self, scores):
        """
        Normalize scores to [0, 1] range using min-max normalization

        Args:
            scores: Raw scores

        Returns:
            Normalized scores
        """
        min_score = torch.min(scores)
        max_score = torch.max(scores)

        if max_score - min_score < 1e-10:
            return torch.ones_like(scores) * 0.5

        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

    def combine_scores(self, physics_scores, uncertainty_scores, gradient_scores):
        """
        Combine all three scores with normalization

        Args:
            physics_scores: Physics-based scores
            uncertainty_scores: Uncertainty-based scores
            gradient_scores: Gradient-based scores

        Returns:
            Combined scores
        """
        # Normalize each score to [0, 1]
        physics_norm = self.normalize_scores(physics_scores)
        uncertainty_norm = self.normalize_scores(uncertainty_scores)
        gradient_norm = self.normalize_scores(gradient_scores)

        # Weighted combination
        combined = (self.w_physics * physics_norm +
                    self.w_uncertainty * uncertainty_norm +
                    self.w_gradient * gradient_norm)

        return combined, physics_norm, uncertainty_norm, gradient_norm

    def find_optimal_k(self, x_candidates, scores, k_range=(5, 25)):
        """
        Find optimal number of clusters using Silhouette score

        Args:
            x_candidates: Candidate points
            scores: Combined scores
            k_range: Range of K to test

        Returns:
            Optimal K
        """
        # Combine features and scores for clustering
        features = torch.cat([x_candidates, scores.unsqueeze(1)], dim=1).cpu().numpy()

        best_k = k_range[0]
        best_score = -1

        # Limit samples for efficiency
        n_samples = min(5000, features.shape[0])
        indices = np.random.choice(features.shape[0], n_samples, replace=False)
        features_sample = features[indices]

        for k in range(k_range[0], k_range[1] + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_sample)

                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(features_sample, labels,
                                                 metric='euclidean', sample_size=1000)

                    if sil_score > best_score:
                        best_score = sil_score
                        best_k = k
            except Exception:
                continue

        print(f"   Optimal K={best_k} (Silhouette={best_score:.4f})")
        return best_k

    def select_points_via_clustering(self, x_candidates, scores, n_select=2000,
                                     adaptive_k=True, k_fixed=10):
        """
        Select top points using K-means clustering

        Strategy: Cluster points, then select highest-scoring points from each cluster

        Args:
            x_candidates: Candidate points
            scores: Combined scores
            n_select: Number of points to select
            adaptive_k: Whether to find optimal K automatically
            k_fixed: Fixed K if adaptive_k=False

        Returns:
            Selected points and their indices
        """
        # Find optimal K
        if adaptive_k:
            k = self.find_optimal_k(x_candidates, scores)
        else:
            k = k_fixed

        # Combine features and scores for clustering
        features = torch.cat([x_candidates, scores.unsqueeze(1)], dim=1).cpu().numpy()

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Select points from each cluster
        selected_indices = []
        points_per_cluster = n_select // k
        remaining_points = n_select % k

        scores_np = scores.cpu().numpy()

        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_scores = scores_np[cluster_indices]

            # Number of points to select from this cluster
            n_from_cluster = points_per_cluster
            if cluster_id < remaining_points:
                n_from_cluster += 1

            # Select top scoring points from this cluster
            if len(cluster_indices) > 0:
                n_from_cluster = min(n_from_cluster, len(cluster_indices))
                top_indices_in_cluster = np.argsort(cluster_scores)[-n_from_cluster:]
                selected_indices.extend(cluster_indices[top_indices_in_cluster])

        selected_indices = np.array(selected_indices)
        selected_points = x_candidates[selected_indices]

        return selected_points, selected_indices

    def select_collocation_points(self, n_candidates=20000, n_select=2000,
                                  adaptive_k=True, seed=None, verbose=True):
        """
        Main function to select informative collocation points

        Pipeline:
        1. Generate candidate points via LHS
        2. Score using physics, uncertainty, and gradient metrics
        3. Combine and normalize scores
        4. Cluster and select top points

        Args:
            n_candidates: Number of candidate points to generate
            n_select: Number of points to select
            adaptive_k: Whether to find optimal K automatically
            seed: Random seed
            verbose: Print progress

        Returns:
            selected_points: Selected collocation points
            score_info: Dictionary with all scores for analysis
        """
        if verbose:
            print(f"\nActive Learning Point Selection")
            print(f"   Generating {n_candidates} candidates via LHS...")

        # Generate candidates
        x_candidates = self.generate_candidate_points(n_candidates, seed=seed)

        if verbose:
            print(f"   Computing physics scores...")
        physics_scores = self.compute_physics_score(x_candidates)

        if verbose:
            print(f"   Computing uncertainty scores...")
        uncertainty_scores = self.compute_uncertainty_score(x_candidates)

        if verbose:
            print(f"   Computing gradient scores (v2: per-point input-space gradients)...")
        gradient_scores = self.compute_gradient_score(x_candidates)

        if verbose:
            print(f"   Combining and normalizing scores...")
        combined_scores, phys_norm, unc_norm, grad_norm = self.combine_scores(
            physics_scores, uncertainty_scores, gradient_scores
        )

        if verbose:
            print(f"   Clustering and selecting top {n_select} points...")
        selected_points, selected_indices = self.select_points_via_clustering(
            x_candidates, combined_scores, n_select=n_select, adaptive_k=adaptive_k
        )

        # Prepare score info for analysis
        score_info = {
            'physics_scores': physics_scores.cpu().numpy(),
            'uncertainty_scores': uncertainty_scores.cpu().numpy(),
            'gradient_scores': gradient_scores.cpu().numpy(),
            'combined_scores': combined_scores.cpu().numpy(),
            'physics_norm': phys_norm.cpu().numpy(),
            'uncertainty_norm': unc_norm.cpu().numpy(),
            'gradient_norm': grad_norm.cpu().numpy(),
            'selected_indices': selected_indices,
            'all_candidates': x_candidates.cpu().numpy()
        }

        if verbose:
            print(f"   [ok] Selected {len(selected_points)} informative points\n")
            print(f"      Gradient score stats: mean={gradient_scores.mean():.4e}, "
                  f"std={gradient_scores.std():.4e}, "
                  f"max/min ratio={gradient_scores.max()/(gradient_scores.min()+1e-10):.2f}x")

        return selected_points, score_info
