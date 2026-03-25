"""
Data quality metrics for comparing original and synthetic datasets.

Provides simple comparison metrics including:
- KL Divergence (per-column distribution similarity)
- Statistical Fidelity (mean, std, correlation preservation)
- Utility Score (based on column statistics)
- Privacy Risk Score (based on nearest neighbor distances)
- Column-wise Distance Metrics
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


@dataclass
class ColumnMetrics:
    """Metrics for a single column comparison."""
    column_name: str
    column_type: str  # 'numeric' or 'categorical'
    kl_divergence: float
    js_divergence: float  # Jensen-Shannon (symmetric, bounded 0-1)
    wasserstein_distance: Optional[float] = None  # Only for numeric
    mean_diff: Optional[float] = None  # Only for numeric
    std_diff: Optional[float] = None  # Only for numeric
    category_coverage: Optional[float] = None  # Only for categorical


@dataclass
class FidelityMetrics:
    """Statistical fidelity metrics."""
    mean_preservation: float  # How well means are preserved (0-1)
    std_preservation: float  # How well stds are preserved (0-1)
    correlation_preservation: float  # How well correlations are preserved (0-1)
    overall_fidelity: float  # Combined score (0-1)


@dataclass
class UtilityMetrics:
    """Utility metrics for synthetic data."""
    column_correlation: float  # Correlation of column statistics
    distribution_similarity: float  # Overall distribution match
    overall_utility: float  # Combined score (0-1)


@dataclass
class PrivacyMetrics:
    """Simple privacy risk metrics."""
    min_distance_ratio: float  # Avg min distance to nearest real record
    duplicate_rate: float  # Rate of near-duplicates
    privacy_score: float  # Overall privacy score (higher = more private)
    dcr: float = 0.0  # DCR: Average Distance to Closest Record (synth→original)
    dcr_5th_percentile: float = 0.0  # 5th percentile DCR (worst-case outlier risk)


@dataclass
class DifferentialPrivacyMetrics:
    """Differential privacy metrics computed against an evaluation dataset.
    
    These are empirical privacy metrics that approximate DP guarantees
    by measuring information leakage between training and evaluation sets.
    """
    dcr_train: float  # Distance to Closest Record (synthetic to training)
    dcr_eval: float   # Distance to Closest Record (synthetic to evaluation)
    dcr_ratio: float  # Ratio: dcr_eval / dcr_train (>1 means better privacy)
    
    # Membership inference metrics
    membership_advantage: float  # Attacker advantage in distinguishing train/eval
    membership_auc: float  # AUC for membership inference attack
    
    # Attribute inference metrics
    attribute_inference_risk: float  # Risk of inferring sensitive attributes
    
    # Overall DP score (empirical, 0-1, higher = more private)
    empirical_dp_score: float
    
    # Estimated epsilon (if applicable, based on empirical metrics)
    estimated_epsilon: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark comparison result."""
    original_rows: int
    synthetic_rows: int
    column_count: int
    
    # Per-column metrics
    column_metrics: Dict[str, ColumnMetrics] = field(default_factory=dict)
    
    # Aggregate metrics
    avg_kl_divergence: float = 0.0
    avg_js_divergence: float = 0.0
    
    # High-level scores
    fidelity: Optional[FidelityMetrics] = None
    utility: Optional[UtilityMetrics] = None
    privacy: Optional[PrivacyMetrics] = None
    
    # Overall quality score (0-1, higher is better)
    overall_quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_rows": self.original_rows,
            "synthetic_rows": self.synthetic_rows,
            "column_count": self.column_count,
            "avg_kl_divergence": self.avg_kl_divergence,
            "avg_js_divergence": self.avg_js_divergence,
            "overall_quality_score": self.overall_quality_score,
            "fidelity": {
                "mean_preservation": self.fidelity.mean_preservation if self.fidelity else None,
                "std_preservation": self.fidelity.std_preservation if self.fidelity else None,
                "correlation_preservation": self.fidelity.correlation_preservation if self.fidelity else None,
                "overall_fidelity": self.fidelity.overall_fidelity if self.fidelity else None,
            },
            "utility": {
                "column_correlation": self.utility.column_correlation if self.utility else None,
                "distribution_similarity": self.utility.distribution_similarity if self.utility else None,
                "overall_utility": self.utility.overall_utility if self.utility else None,
            },
            "privacy": {
                "min_distance_ratio": self.privacy.min_distance_ratio if self.privacy else None,
                "duplicate_rate": self.privacy.duplicate_rate if self.privacy else None,
                "privacy_score": self.privacy.privacy_score if self.privacy else None,
                "dcr": self.privacy.dcr if self.privacy else None,
                "dcr_5th_percentile": self.privacy.dcr_5th_percentile if self.privacy else None,
            },
            "column_metrics": {
                name: {
                    "type": m.column_type,
                    "kl_divergence": m.kl_divergence,
                    "js_divergence": m.js_divergence,
                    "wasserstein_distance": m.wasserstein_distance,
                    "mean_diff": m.mean_diff,
                    "std_diff": m.std_diff,
                    "category_coverage": m.category_coverage,
                }
                for name, m in self.column_metrics.items()
            },
        }


class DataQualityBenchmark:
    """Compare original and synthetic datasets using quality metrics."""
    
    def __init__(self, n_bins: int = 50, sample_size: int = 1000):
        """
        Initialize benchmark.
        
        Args:
            n_bins: Number of bins for discretizing continuous distributions
            sample_size: Sample size for privacy metrics (to limit computation)
        """
        self.n_bins = n_bins
        self.sample_size = sample_size
    
    def compare(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> BenchmarkResult:
        """
        Compare original and synthetic datasets.
        
        Args:
            original: Original dataset
            synthetic: Synthetic dataset
            
        Returns:
            BenchmarkResult with all metrics
        """
        # Ensure columns match
        common_cols = list(set(original.columns) & set(synthetic.columns))
        if not common_cols:
            raise ValueError("No common columns between original and synthetic datasets")
        
        original = original[common_cols].copy()
        synthetic = synthetic[common_cols].copy()
        
        # Calculate per-column metrics
        column_metrics = {}
        kl_values = []
        js_values = []
        
        for col in common_cols:
            col_metric = self._compare_column(original[col], synthetic[col])
            column_metrics[col] = col_metric
            kl_values.append(col_metric.kl_divergence)
            js_values.append(col_metric.js_divergence)
        
        # Calculate aggregate metrics
        avg_kl = np.mean(kl_values) if kl_values else 0.0
        avg_js = np.mean(js_values) if js_values else 0.0
        
        # Calculate fidelity
        fidelity = self._calculate_fidelity(original, synthetic)
        
        # Calculate utility
        utility = self._calculate_utility(original, synthetic, avg_js)
        
        # Calculate privacy
        privacy = self._calculate_privacy(original, synthetic)
        
        # Overall quality score (weighted combination)
        overall_score = (
            0.4 * fidelity.overall_fidelity +
            0.3 * utility.overall_utility +
            0.3 * (1 - min(avg_js, 1.0))  # Lower divergence = higher quality
        )
        
        return BenchmarkResult(
            original_rows=len(original),
            synthetic_rows=len(synthetic),
            column_count=len(common_cols),
            column_metrics=column_metrics,
            avg_kl_divergence=avg_kl,
            avg_js_divergence=avg_js,
            fidelity=fidelity,
            utility=utility,
            privacy=privacy,
            overall_quality_score=overall_score,
        )
    
    def _compare_column(
        self, 
        original: pd.Series, 
        synthetic: pd.Series
    ) -> ColumnMetrics:
        """Compare a single column between datasets."""
        col_name = original.name
        
        # Determine column type
        if pd.api.types.is_numeric_dtype(original):
            return self._compare_numeric_column(original, synthetic, col_name)
        else:
            return self._compare_categorical_column(original, synthetic, col_name)
    
    def _compare_numeric_column(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        col_name: str,
    ) -> ColumnMetrics:
        """Compare numeric columns."""
        # Drop NaN values
        orig_clean = original.dropna()
        synth_clean = synthetic.dropna()
        
        if len(orig_clean) == 0 or len(synth_clean) == 0:
            return ColumnMetrics(
                column_name=col_name,
                column_type="numeric",
                kl_divergence=float('inf'),
                js_divergence=1.0,
            )
        
        # Discretize for KL/JS divergence
        all_values = np.concatenate([orig_clean.values, synth_clean.values])
        min_val, max_val = np.min(all_values), np.max(all_values)
        
        if min_val == max_val:
            # Constant column
            bins = np.array([min_val - 0.5, max_val + 0.5])
        else:
            bins = np.linspace(min_val, max_val, self.n_bins + 1)
        
        orig_hist, _ = np.histogram(orig_clean, bins=bins, density=True)
        synth_hist, _ = np.histogram(synth_clean, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        orig_hist = orig_hist + eps
        synth_hist = synth_hist + eps
        
        # Normalize
        orig_hist = orig_hist / orig_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()
        
        # KL divergence
        kl_div = stats.entropy(orig_hist, synth_hist)
        
        # JS divergence (symmetric, bounded 0-1)
        js_div = jensenshannon(orig_hist, synth_hist) ** 2
        
        # Wasserstein distance
        try:
            wass_dist = stats.wasserstein_distance(orig_clean, synth_clean)
            # Normalize by range
            data_range = max_val - min_val if max_val != min_val else 1.0
            wass_dist = wass_dist / data_range
        except Exception:
            wass_dist = None
        
        # Mean and std differences
        orig_mean, synth_mean = orig_clean.mean(), synth_clean.mean()
        orig_std, synth_std = orig_clean.std(), synth_clean.std()
        
        mean_diff = abs(orig_mean - synth_mean) / (abs(orig_mean) + 1e-10)
        std_diff = abs(orig_std - synth_std) / (abs(orig_std) + 1e-10)
        
        return ColumnMetrics(
            column_name=col_name,
            column_type="numeric",
            kl_divergence=float(kl_div),
            js_divergence=float(js_div),
            wasserstein_distance=wass_dist,
            mean_diff=mean_diff,
            std_diff=std_diff,
        )
    
    def _compare_categorical_column(
        self,
        original: pd.Series,
        synthetic: pd.Series,
        col_name: str,
    ) -> ColumnMetrics:
        """Compare categorical columns."""
        # Get value counts
        orig_counts = original.value_counts(normalize=True)
        synth_counts = synthetic.value_counts(normalize=True)
        
        # Get all categories
        all_cats = set(orig_counts.index) | set(synth_counts.index)
        
        if not all_cats:
            return ColumnMetrics(
                column_name=col_name,
                column_type="categorical",
                kl_divergence=float('inf'),
                js_divergence=1.0,
            )
        
        # Build probability vectors
        eps = 1e-10
        orig_probs = np.array([orig_counts.get(c, 0) + eps for c in all_cats])
        synth_probs = np.array([synth_counts.get(c, 0) + eps for c in all_cats])
        
        # Normalize
        orig_probs = orig_probs / orig_probs.sum()
        synth_probs = synth_probs / synth_probs.sum()
        
        # KL and JS divergence
        kl_div = stats.entropy(orig_probs, synth_probs)
        js_div = jensenshannon(orig_probs, synth_probs) ** 2
        
        # Category coverage
        orig_cats = set(orig_counts.index)
        synth_cats = set(synth_counts.index)
        coverage = len(orig_cats & synth_cats) / len(orig_cats) if orig_cats else 0.0
        
        return ColumnMetrics(
            column_name=col_name,
            column_type="categorical",
            kl_divergence=float(kl_div),
            js_divergence=float(js_div),
            category_coverage=coverage,
        )
    
    def _calculate_fidelity(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> FidelityMetrics:
        """Calculate statistical fidelity metrics."""
        # Get numeric columns
        numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return FidelityMetrics(
                mean_preservation=1.0,
                std_preservation=1.0,
                correlation_preservation=1.0,
                overall_fidelity=1.0,
            )
        
        # Mean preservation
        orig_means = original[numeric_cols].mean()
        synth_means = synthetic[numeric_cols].mean()
        mean_diffs = np.abs(orig_means - synth_means) / (np.abs(orig_means) + 1e-10)
        mean_preservation = float(1 - np.clip(mean_diffs.mean(), 0, 1))
        
        # Std preservation
        orig_stds = original[numeric_cols].std()
        synth_stds = synthetic[numeric_cols].std()
        std_diffs = np.abs(orig_stds - synth_stds) / (np.abs(orig_stds) + 1e-10)
        std_preservation = float(1 - np.clip(std_diffs.mean(), 0, 1))
        
        # Correlation preservation
        if len(numeric_cols) > 1:
            orig_corr = original[numeric_cols].corr().values.copy()
            synth_corr = synthetic[numeric_cols].corr().values.copy()
            
            # Handle NaN in correlation matrices
            orig_corr = np.nan_to_num(orig_corr, nan=0.0, copy=True)
            synth_corr = np.nan_to_num(synth_corr, nan=0.0, copy=True)
            
            # Flatten and compute correlation
            orig_flat = orig_corr[np.triu_indices_from(orig_corr, k=1)]
            synth_flat = synth_corr[np.triu_indices_from(synth_corr, k=1)]
            
            if len(orig_flat) > 0:
                corr_preservation = float(np.corrcoef(orig_flat, synth_flat)[0, 1])
                corr_preservation = max(0, corr_preservation)  # Clamp to 0-1
            else:
                corr_preservation = 1.0
        else:
            corr_preservation = 1.0
        
        overall = (mean_preservation + std_preservation + corr_preservation) / 3
        
        return FidelityMetrics(
            mean_preservation=mean_preservation,
            std_preservation=std_preservation,
            correlation_preservation=corr_preservation,
            overall_fidelity=overall,
        )
    
    def _calculate_utility(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        avg_js: float,
    ) -> UtilityMetrics:
        """Calculate utility metrics."""
        numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return UtilityMetrics(
                column_correlation=1.0,
                distribution_similarity=1 - avg_js,
                overall_utility=1 - avg_js,
            )
        
        # Column-wise correlation of statistics
        orig_stats = pd.DataFrame({
            'mean': original[numeric_cols].mean(),
            'std': original[numeric_cols].std(),
            'min': original[numeric_cols].min(),
            'max': original[numeric_cols].max(),
        })
        synth_stats = pd.DataFrame({
            'mean': synthetic[numeric_cols].mean(),
            'std': synthetic[numeric_cols].std(),
            'min': synthetic[numeric_cols].min(),
            'max': synthetic[numeric_cols].max(),
        })
        
        # Flatten and correlate
        orig_flat = orig_stats.values.flatten()
        synth_flat = synth_stats.values.flatten()
        
        col_corr = float(np.corrcoef(orig_flat, synth_flat)[0, 1])
        col_corr = max(0, col_corr)  # Clamp to 0-1
        
        dist_sim = 1 - min(avg_js, 1.0)
        
        overall = (col_corr + dist_sim) / 2
        
        return UtilityMetrics(
            column_correlation=col_corr,
            distribution_similarity=dist_sim,
            overall_utility=overall,
        )
    
    def _calculate_privacy(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> PrivacyMetrics:
        """Calculate simple privacy risk metrics."""
        numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return PrivacyMetrics(
                min_distance_ratio=1.0,
                duplicate_rate=0.0,
                privacy_score=1.0,
                dcr=1.0,
                dcr_5th_percentile=1.0,
            )
        
        # Sample for efficiency
        n_orig = min(self.sample_size, len(original))
        n_synth = min(self.sample_size, len(synthetic))
        
        orig_sample = original[numeric_cols].sample(n=n_orig, random_state=42).values.copy()
        synth_sample = synthetic[numeric_cols].sample(n=n_synth, random_state=42).values.copy()
        
        # Normalize
        scaler_min = np.min(orig_sample, axis=0)
        scaler_max = np.max(orig_sample, axis=0)
        scaler_range = scaler_max - scaler_min
        scaler_range = np.where(scaler_range == 0, 1, scaler_range)  # Avoid division by zero
        
        orig_norm = (orig_sample - scaler_min) / scaler_range
        synth_norm = (synth_sample - scaler_min) / scaler_range
        
        # Calculate min distances from synthetic to original
        min_distances = []
        near_duplicate_count = 0
        duplicate_threshold = 0.01  # 1% in normalized space
        
        for synth_row in synth_norm:
            distances = np.sqrt(np.sum((orig_norm - synth_row) ** 2, axis=1))
            min_dist = np.min(distances)
            min_distances.append(min_dist)
            
            if min_dist < duplicate_threshold:
                near_duplicate_count += 1
        
        avg_min_dist = np.mean(min_distances) if min_distances else 1.0
        duplicate_rate = near_duplicate_count / n_synth if n_synth > 0 else 0.0

        # DCR: use _compute_dcr on normalized data for the average
        dcr_value = self._compute_dcr(synth_norm, orig_norm)
        # 5th percentile DCR: shows closest outlier risk
        dcr_5th = float(np.percentile(min_distances, 5)) if min_distances else 0.0

        # Privacy score: higher distance and lower duplicate rate = better
        privacy_score = min(1.0, avg_min_dist) * (1 - duplicate_rate)

        return PrivacyMetrics(
            min_distance_ratio=float(avg_min_dist),
            duplicate_rate=float(duplicate_rate),
            privacy_score=float(privacy_score),
            dcr=float(dcr_value),
            dcr_5th_percentile=float(dcr_5th),
        )

    def calculate_differential_privacy(
        self,
        training_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        evaluation_data: pd.DataFrame,
    ) -> DifferentialPrivacyMetrics:
        """Calculate differential privacy metrics using an evaluation dataset.
        
        This method computes empirical DP metrics by comparing how the synthetic
        data relates to the training set vs. a held-out evaluation set. A model
        with good DP properties should not reveal whether a record was in the
        training set.
        
        Args:
            training_data: Original training dataset (used to train the synthesizer)
            synthetic_data: Generated synthetic dataset
            evaluation_data: Held-out evaluation dataset (NOT used in training)
            
        Returns:
            DifferentialPrivacyMetrics with DCR ratio, membership inference metrics,
            and empirical DP score.
        """
        # Get common numeric columns
        common_cols = list(
            set(training_data.columns) & 
            set(synthetic_data.columns) & 
            set(evaluation_data.columns)
        )
        numeric_cols = [
            c for c in common_cols 
            if pd.api.types.is_numeric_dtype(training_data[c])
        ]
        
        if not numeric_cols:
            # Return default values if no numeric columns
            return DifferentialPrivacyMetrics(
                dcr_train=0.0,
                dcr_eval=0.0,
                dcr_ratio=1.0,
                membership_advantage=0.0,
                membership_auc=0.5,
                attribute_inference_risk=0.0,
                empirical_dp_score=1.0,
            )
        
        # Sample for efficiency
        n_train = min(self.sample_size, len(training_data))
        n_synth = min(self.sample_size, len(synthetic_data))
        n_eval = min(self.sample_size, len(evaluation_data))
        
        train_sample = training_data[numeric_cols].sample(
            n=n_train, random_state=42
        ).values.astype(float)
        synth_sample = synthetic_data[numeric_cols].sample(
            n=n_synth, random_state=42
        ).values.astype(float)
        eval_sample = evaluation_data[numeric_cols].sample(
            n=n_eval, random_state=42
        ).values.astype(float)
        
        # Normalize using training data statistics
        scaler_min = np.nanmin(train_sample, axis=0)
        scaler_max = np.nanmax(train_sample, axis=0)
        scaler_range = scaler_max - scaler_min
        scaler_range = np.where(scaler_range == 0, 1, scaler_range)
        
        train_norm = (train_sample - scaler_min) / scaler_range
        synth_norm = (synth_sample - scaler_min) / scaler_range
        eval_norm = (eval_sample - scaler_min) / scaler_range
        
        # Handle any remaining NaN values
        train_norm = np.nan_to_num(train_norm, nan=0.0)
        synth_norm = np.nan_to_num(synth_norm, nan=0.0)
        eval_norm = np.nan_to_num(eval_norm, nan=0.0)
        
        # === 1. Distance to Closest Record (DCR) ===
        dcr_train = self._compute_dcr(synth_norm, train_norm)
        dcr_eval = self._compute_dcr(synth_norm, eval_norm)
        
        # DCR ratio: > 1 means synthetic is closer to train (potential leakage)
        dcr_ratio = dcr_eval / (dcr_train + 1e-10)
        
        # === 2. Membership Inference Attack Simulation ===
        membership_advantage, membership_auc = self._membership_inference_attack(
            synth_norm, train_norm, eval_norm
        )
        
        # === 3. Attribute Inference Risk ===
        attribute_risk = self._attribute_inference_risk(
            synth_norm, train_norm, eval_norm
        )
        
        # === 4. Compute Empirical DP Score ===
        # Higher is better (more private)
        # Components:
        # - dcr_ratio close to 1.0 is good (no preference for train)
        # - membership_advantage close to 0 is good
        # - attribute_risk close to 0.5 (random) is good
        
        dcr_component = 1.0 / (1.0 + abs(dcr_ratio - 1.0))  # Peak at ratio=1
        membership_component = 1.0 - min(membership_advantage, 1.0)
        attribute_component = 1.0 - 2 * abs(attribute_risk - 0.5)  # Peak at 0.5
        
        empirical_dp_score = (
            0.4 * dcr_component +
            0.4 * membership_component +
            0.2 * attribute_component
        )
        
        # === 5. Estimate Epsilon (heuristic) ===
        # Based on membership advantage: advantage ≈ exp(ε) - 1
        # So ε ≈ ln(1 + advantage)
        if membership_advantage > 0:
            estimated_epsilon = float(np.log(1 + membership_advantage * 10))
        else:
            estimated_epsilon = 0.0
        
        return DifferentialPrivacyMetrics(
            dcr_train=float(dcr_train),
            dcr_eval=float(dcr_eval),
            dcr_ratio=float(dcr_ratio),
            membership_advantage=float(membership_advantage),
            membership_auc=float(membership_auc),
            attribute_inference_risk=float(attribute_risk),
            empirical_dp_score=float(empirical_dp_score),
            estimated_epsilon=estimated_epsilon,
        )
    
    def _compute_dcr(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute average Distance to Closest Record from source to target."""
        min_distances = []
        for row in source:
            distances = np.sqrt(np.sum((target - row) ** 2, axis=1))
            min_distances.append(np.min(distances))
        return float(np.mean(min_distances)) if min_distances else 0.0
    
    def _membership_inference_attack(
        self,
        synth_data: np.ndarray,
        train_data: np.ndarray,
        eval_data: np.ndarray,
    ) -> Tuple[float, float]:
        """Simulate a membership inference attack.
        
        Uses distance-based classifier: if a record is closer to synthetic
        data, predict it was in training set.
        
        Returns:
            Tuple of (advantage, auc)
            - advantage: attacker's advantage over random guessing
            - auc: Area Under ROC Curve for the attack
        """
        # Compute min distances from train records to synthetic
        train_distances = []
        for row in train_data:
            distances = np.sqrt(np.sum((synth_data - row) ** 2, axis=1))
            train_distances.append(np.min(distances))
        
        # Compute min distances from eval records to synthetic
        eval_distances = []
        for row in eval_data:
            distances = np.sqrt(np.sum((synth_data - row) ** 2, axis=1))
            eval_distances.append(np.min(distances))
        
        # Labels: 1 for train (member), 0 for eval (non-member)
        labels = np.array([1] * len(train_distances) + [0] * len(eval_distances))
        # Scores: negative distance (smaller distance = more likely member)
        scores = np.array([-d for d in train_distances] + [-d for d in eval_distances])
        
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            return 0.0, 0.5
        
        # Calculate AUC manually (to avoid sklearn dependency)
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_indices]
        
        # Calculate AUC using trapezoidal rule
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.5
        
        tpr_prev, fpr_prev = 0.0, 0.0
        tp, fp = 0, 0
        auc = 0.0
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr
        
        # Advantage: 2 * (AUC - 0.5), clamped to [0, 1]
        advantage = max(0, 2 * (auc - 0.5))
        
        return advantage, auc
    
    def _attribute_inference_risk(
        self,
        synth_data: np.ndarray,
        train_data: np.ndarray,
        eval_data: np.ndarray,
    ) -> float:
        """Calculate attribute inference risk.
        
        Measures how well we can predict the last column (as a proxy for
        sensitive attributes) using nearest neighbor from synthetic data.
        
        Returns:
            Risk score: 0.5 = random (good), 1.0 = perfect inference (bad)
        """
        if synth_data.shape[1] < 2:
            return 0.5  # Cannot do attribute inference with 1 column
        
        # Use last column as "sensitive attribute"
        # Predict based on nearest neighbor in synthetic data
        
        def predict_attribute(query_data: np.ndarray) -> float:
            """Predict last column for query data using synth neighbors."""
            correct = 0
            for row in query_data:
                # Find nearest neighbor in synthetic (excluding last col)
                query_features = row[:-1]
                synth_features = synth_data[:, :-1]
                
                distances = np.sqrt(np.sum((synth_features - query_features) ** 2, axis=1))
                nearest_idx = np.argmin(distances)
                
                # Predict last column
                predicted = synth_data[nearest_idx, -1]
                actual = row[-1]
                
                # For numeric: check if within 10% of range
                if abs(predicted - actual) < 0.1:  # Already normalized to 0-1
                    correct += 1
            
            return correct / len(query_data) if len(query_data) > 0 else 0.5
        
        # Average accuracy on train vs eval (should be similar if private)
        train_accuracy = predict_attribute(train_data)
        eval_accuracy = predict_attribute(eval_data)
        
        # Risk is the average accuracy (higher = worse privacy)
        return (train_accuracy + eval_accuracy) / 2
