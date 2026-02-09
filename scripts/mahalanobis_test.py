"""
Simple Trajectory-Based Peak Test
==================================

Instead of multiple metrics, this uses a SINGLE distance measure that captures
how "extreme" an N-year trajectory is compared to the null hypothesis.

The key insight: With autocorrelation, some trajectories are less likely than 
others even if they reach the same endpoint. For example:
- (0, 0, -300) requires a sudden 300 MtCO2 jump → unlikely given autocorrelation
- (-100, -100, -100) is a consistent decline → more plausible as noise

This test is built to allow for both with and without autocorrelation noise generators

We measure "extremeness" using the trajectory's likelihood under the null model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression




class MahalanobisTest:
    """
    Trajectory test using Mahalanobis distance with proper covariance structure.
    
    Works for both AR(1) correlated noise and white noise.
    """
    
    def __init__(self, peak_test):
        """
        Initialize with a reference to an EmissionsPeakTest instance.
        
        Args:
            peak_test: An instance of EmissionsPeakTest with data already loaded
        """
        self.peak_test = peak_test
        self.results = None
        self.covariance_matrix = None
    
    def run_test(
        self,
        n_bootstrap: int = 10000,
        null_hypothesis: str = "zero_trend",
        bootstrap_method: str = "ar_bootstrap",
        use_empirical_covariance: bool = False
    ) -> Dict:
        """
        Run Mahalanobis distance test.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            null_hypothesis: "zero_trend", "recent_trend", or a float
            bootstrap_method: "ar_bootstrap" or "white_noise"
            use_empirical_covariance: If True, estimate covariance from bootstrap samples
                                      If False, use theoretical covariance 
                                      from the noise parameterisation step
        
        Returns:
            Dictionary with test results
        """
        if self.peak_test.noise_generator is None:
            self.peak_test.create_noise_generator()
        
        print("\nRunning Mahalanobis distance trajectory test...")
        print(f"  Null hypothesis: {null_hypothesis}")
        print(f"  Bootstrap method: {bootstrap_method}")
        print(f"  Bootstrap samples: {n_bootstrap}")
        print(f"  Covariance: {'Empirical' if use_empirical_covariance else 'Theoretical AR(1)'}")
        
        # Generate bootstrap trajectories
        bootstrap_data = self._generate_bootstrap_trajectories(
            n_bootstrap, null_hypothesis, bootstrap_method
        )
        
        # Calculate covariance matrix
        if use_empirical_covariance:
            # Estimate from bootstrap samples
            self.covariance_matrix = self._estimate_empirical_covariance(
                bootstrap_data['deviations']
            )
            print(f"\nCovariance estimated from {n_bootstrap} bootstrap samples")
        else:
            # Use theoretical AR(1) or white noise covariance
            self.covariance_matrix = self._build_theoretical_covariance()
            phi = self.peak_test.autocorr_params["phi"]
            if abs(phi) > 0.1:
                print(f"\nUsing AR(1) covariance with φ={phi:.3f}")
            else:
                print(f"\nUsing white noise covariance (φ≈0)")
        
        # Calculate Mahalanobis distance for observed trajectory
        observed_emissions = self.peak_test.test_data['emissions'].values
        null_baseline = bootstrap_data['null_baseline']
        observed_deviations = observed_emissions - null_baseline
        
        observed_mahal = self._calculate_mahalanobis(
            observed_deviations, 
            self.covariance_matrix
        )
        
        # Calculate Mahalanobis distances for bootstrap samples
        bootstrap_mahal = []
        for deviations in bootstrap_data['deviations']:
            mahal = self._calculate_mahalanobis(deviations, self.covariance_matrix)
            bootstrap_mahal.append(mahal)
        
        bootstrap_mahal = np.array(bootstrap_mahal)
        
        # Calculate p-value
        # For DECLINE detection: we want large distances in the NEGATIVE direction
        # observed_mahal is already signed (negative if declining)
        # p = P(bootstrap distance ≤ observed distance)
        # Lower (more negative) = more extreme decline
        p_value = np.sum(bootstrap_mahal <= observed_mahal) / len(bootstrap_mahal)
        
        # Effect size
        null_mean = np.mean(bootstrap_mahal)
        null_std = np.std(bootstrap_mahal)
        effect_size = (null_mean - observed_mahal) / null_std if null_std > 0 else 0
        
        # Also calculate total deviation and slope for comparison
        total_deviation = np.sum(observed_deviations)
        
        years = self.peak_test.test_data.year.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(years, observed_emissions)
        observed_slope = model.coef_[0]
        observed_r2 = model.score(years, observed_emissions)
        
        self.results = {
            'observed_mahal': observed_mahal,
            'observed_deviations': observed_deviations,
            'total_deviation': total_deviation,
            'observed_slope': observed_slope,
            'observed_r2': observed_r2,
            'bootstrap_mahal': bootstrap_mahal,
            'bootstrap_deviations': bootstrap_data['deviations'],
            'null_baseline': null_baseline,
            'p_value': p_value,
            'effect_size': effect_size,
            'null_mean': null_mean,
            'null_std': null_std,
            'significant_at_0.10': p_value < 0.10,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'null_hypothesis': null_hypothesis,
            'n_bootstrap': n_bootstrap,
            'has_autocorrelation': self.peak_test.autocorr_params["has_autocorr"],
            'phi': self.peak_test.autocorr_params["phi"]
        }
        
        print(f"\nObserved trajectory:")
        print(f"  Deviations from null: {observed_deviations}")
        print(f"  Total deviation: {total_deviation:.0f} {self.peak_test.unit}")
        print(f"  Mahalanobis distance: {observed_mahal:.2f}")
        print(f"  Slope: {observed_slope:.2f} {self.peak_test.unit}/year (R²={observed_r2:.3f})")
        
        print(f"\nNull distribution:")
        print(f"  Mean Mahalanobis: {null_mean:.2f}")
        print(f"  Std dev: {null_std:.2f}")
        
        print(f"\nResults:")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant at α=0.05: {self.results['significant_at_0.05']}")
        print(f"  Effect size: {effect_size:.2f} standard deviations")
        
        return self.results
    
    def _build_theoretical_covariance(self) -> np.ndarray:
        """
        Build theoretical covariance matrix based on noise model.
        
        For AR(1): Σ[i,j] = σ² * φ^|i-j| / (1-φ²)
        For white noise (φ≈0): Σ[i,j] = σ² if i==j, else 0
        """
        n = len(self.peak_test.test_data)
        phi = self.peak_test.autocorr_params["phi"]
        sigma = self.peak_test.autocorr_params["sigma_residuals"]
        
        if abs(phi) < 0.1:
            # White noise: diagonal covariance matrix
            Sigma = np.eye(n) * sigma**2
            print(f"  Building white noise covariance (σ={sigma:.1f})")
        else:
            # AR(1): full covariance structure
            variance = sigma**2 / (1 - phi**2) if abs(phi) < 1 else sigma**2
            Sigma = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    Sigma[i, j] = variance * (phi ** abs(i - j))
            print(f"  Building AR(1) covariance (φ={phi:.3f}, σ={sigma:.1f})")
        
        return Sigma
    
    def _estimate_empirical_covariance(self, bootstrap_deviations: np.ndarray) -> np.ndarray:
        """
        Estimate covariance matrix empirically from bootstrap samples.
        
        This is the most general approach - works for any noise structure!
        """
        # bootstrap_deviations is shape (n_bootstrap, n_years)
        # Covariance is n_years x n_years
        return np.cov(bootstrap_deviations.T)
    
    def _calculate_mahalanobis(
        self, 
        deviations: np.ndarray, 
        Sigma: np.ndarray
    ) -> float:
        """
        Calculate Mahalanobis distance.
        
        d² = x' Σ^(-1) x
        
        Returns SIGNED distance (negative if cumulative deviation is negative).
        """
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            mahal_squared = deviations.T @ Sigma_inv @ deviations
            mahal = np.sqrt(mahal_squared)
            
            # Sign based on direction of total deviation
            sign = -1 if np.sum(deviations) < 0 else 1
            return sign * mahal
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            print("Warning: Singular covariance matrix, using Euclidean distance")
            euclidean = np.sqrt(np.sum(deviations**2))
            sign = -1 if np.sum(deviations) < 0 else 1
            return sign * euclidean
    
    def _generate_bootstrap_trajectories(
        self,
        n_bootstrap: int,
        null_hypothesis: str,
        bootstrap_method: str
    ) -> Dict:
        """Generate bootstrap trajectories under null hypothesis."""
        n_test_points = len(self.peak_test.test_data)
        years = self.peak_test.test_data.year.values
        
        baseline_emissions = self.peak_test.test_data.emissions.iloc[0]
        baseline_year = self.peak_test.test_data.year.iloc[0]
        
        # Determine null trend
        if null_hypothesis == "recent_trend":
            if self.peak_test.recent_historical_trend is None:
                recent_data = self.peak_test.historical_data.tail(5)
                X = recent_data["year"].values.reshape(-1, 1)
                y = recent_data["emissions"].values
                model = LinearRegression()
                model.fit(X, y)
                null_trend = model.coef_[0]
            else:
                null_trend = self.peak_test.recent_historical_trend
        elif isinstance(null_hypothesis, (int,float)):
            null_trend = null_hypothesis
        elif null_hypothesis == "zero_trend":
            null_trend = 0.0
        else:
            null_trend = 0.0
        
        # Null baseline trajectory (no noise)
        trend_component = null_trend * (years - baseline_year)
        null_baseline = baseline_emissions + trend_component
        
        # Generate bootstrap trajectories
        bootstrap_trajectories = []
        bootstrap_deviations = []
        
        for i in range(n_bootstrap):
            # Add noise
            if (bootstrap_method == "ar_bootstrap" and 
                self.peak_test.autocorr_params["has_autocorr"]):
                noise = self.peak_test.noise_generator(
                    n_test_points, 
                    initial_value=None
                )
            else:
                noise = np.random.normal(
                    0, 
                    self.peak_test.autocorr_params["sigma_residuals"], 
                    n_test_points
                )
            
            null_trajectory = null_baseline + noise
            deviations = null_trajectory - null_baseline  # This is just the noise
            
            bootstrap_trajectories.append(null_trajectory)
            bootstrap_deviations.append(deviations)
        
        return {
            'trajectories': np.array(bootstrap_trajectories),
            'deviations': np.array(bootstrap_deviations),
            'null_baseline': null_baseline
        }
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None):
        """Create comprehensive visualization."""
        if self.results is None:
            raise ValueError("Must run test first")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory with null baseline
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_trajectory(ax1)
        
        # 2. Covariance matrix
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_covariance_matrix(ax2)
        
        # 3. Mahalanobis distribution
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_mahalanobis_distribution(ax3)
        
        # 4. Deviations
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_deviations(ax4)
        
        # 5. Bootstrap trajectories
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_bootstrap_trajectories(ax5)
        
        # 6. Summary
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_summary(ax6)
        
        fig.suptitle(
            f'Mahalanobis Trajectory Test: {self.peak_test.variable} ({self.peak_test.region})',
            fontsize=14,
            fontweight='bold'
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _plot_trajectory(self, ax):
        """Plot observed trajectory vs null baseline."""
        years = self.peak_test.test_data.year.values
        observed = self.peak_test.test_data.emissions.values
        null_baseline = self.results['null_baseline']
        
        ax.plot(years, null_baseline, 'b--', linewidth=2, label='Null baseline', alpha=0.7)
        ax.plot(years, observed, 'r-', linewidth=3, marker='o', 
               label='Observed', markersize=8, zorder=10)
        
        # Shade deviation
        ax.fill_between(years, null_baseline, observed, 
                       alpha=0.3, color='red')
        
        # Add deviation annotations
        for i, year in enumerate(years):
            dev = observed[i] - null_baseline[i]
            if abs(dev) > 0:
                ax.annotate(f'{dev:.0f}', 
                           xy=(year, (observed[i] + null_baseline[i])/2),
                           fontsize=9, ha='center', color='darkred', fontweight='bold')
        
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{self.peak_test.variable} ({self.peak_test.unit})')
        ax.set_title('Observed Trajectory vs Null Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_covariance_matrix(self, ax):
        """Plot the covariance matrix."""
        im = ax.imshow(self.covariance_matrix, cmap='RdBu_r', aspect='auto')
        ax.set_title('Covariance Matrix Σ')
        ax.set_xlabel('Year index')
        ax.set_ylabel('Year index')
        ax.set_xticks(range(self.covariance_matrix.shape[0]))
        ax.set_yticks(range(self.covariance_matrix.shape[0]))
        # ax.set_xticks(self.covariance_matrix)
        plt.colorbar(im, ax=ax, label='Covariance')
        
        # Add text annotations
        n = self.covariance_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{self.covariance_matrix[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    def _plot_mahalanobis_distribution(self, ax):
        """Plot Mahalanobis distance distribution."""
        mahal_distances = self.results['bootstrap_mahal']
        observed_mahal = self.results['observed_mahal']
        
        ax.hist(mahal_distances, bins=50, density=True, alpha=0.7,
               color='lightblue', edgecolor='blue', linewidth=0.5)
        
        ax.axvline(observed_mahal, color='red', linewidth=3,
                  label=f'Observed: {observed_mahal:.2f}')
        ax.axvline(self.results['null_mean'], color='blue', linewidth=2,
                  linestyle='--', alpha=0.7, label=f'Null mean: {self.results["null_mean"]:.2f}')
        
        # Add p-value annotation
        ax.text(0.05, 0.95, f'p = {self.results["p_value"]:.4f}',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Mahalanobis Distance')
        ax.set_ylabel('Density')
        ax.set_title('Bootstrap Distribution of Mahalanobis Distances')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_deviations(self, ax):
        """Plot deviations from null baseline."""
        years = self.peak_test.test_data.year.values
        deviations = self.results['observed_deviations']
        
        colors = ['red' if d < 0 else 'green' for d in deviations]
        ax.bar(years, deviations, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='k', linewidth=1)
        
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Deviation ({self.peak_test.unit})')
        ax.set_title('Deviations from Null Baseline')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_bootstrap_trajectories(self, ax):
        """Plot sample of bootstrap trajectories."""
        years = self.peak_test.test_data.year.values
        null_baseline = self.results['null_baseline']
        
        # Plot a sample of bootstrap trajectories
        n_sample = min(100, len(self.results['bootstrap_deviations']))
        for i in range(n_sample):
            trajectory = null_baseline + self.results['bootstrap_deviations'][i]
            ax.plot(years, trajectory, 'b-', alpha=0.05, linewidth=0.5)
        
        # Plot percentiles
        bootstrap_trajectories = null_baseline + self.results['bootstrap_deviations']
        for percentile, color, label in [(5, 'blue', '5-95%ile'), (50, 'darkblue', 'Median'), (95, 'blue', None)]:
            perc = np.percentile(bootstrap_trajectories, percentile, axis=0)
            ax.plot(years, perc, color=color, linewidth=2, label=label, alpha=0.8)
        
        # Plot observed
        observed = self.peak_test.test_data.emissions.values
        ax.plot(years, observed, 'r-', linewidth=3, marker='o',
               label='Observed', markersize=8, zorder=10)
        
        # Plot null baseline
        ax.plot(years, null_baseline, 'k--', linewidth=2, label='Null baseline', alpha=0.7)
        
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{self.peak_test.variable} ({self.peak_test.unit})')
        ax.set_title(f'Bootstrap Trajectories (n={n_sample} shown)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_summary(self, ax):
        """Plot summary statistics."""
        ax.axis('off')
        
        has_autocorr = "Yes" if self.results['has_autocorrelation'] else "No"
        phi_str = f"φ = {self.results['phi']:.3f}" if self.results['has_autocorrelation'] else "φ ≈ 0"
        
        summary_text = f"""
Mahalanobis Distance:
  Observed: {self.results['observed_mahal']:.2f}
  
Linear Fit:
  Slope: {self.results['observed_slope']:.1f}/yr
  R²: {self.results['observed_r2']:.3f}

Statistical Test:
  p-value: {self.results['p_value']:.4f}
  α=0.05: {'✓ Significant' if self.results['significant_at_0.05'] else '✗ Not significant'}
  
Noise Structure:
  Autocorrelated: {has_autocorr}
  {phi_str}

CONCLUSION:
{self._get_conclusion()}
        """
        
        ax.text(0.5, 0.4, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    def _get_conclusion(self):
        """Get interpretation."""
        p = self.results['p_value']
        if p < 0.01:
            return "Very strong evidence\nof peak"
        elif p < 0.05:
            return "Strong evidence\nof peak"
        elif p < 0.10:
            return "Moderate evidence\nof peak"
        else:
            return "Insufficient evidence"
