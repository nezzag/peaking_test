"""
Peak Tester
==================================

A statistical test to determine if an emissions series has peaked
based on N years of declining emissions data, and accounting for the
noise in historical data (which is assumed to continue)

Usage:
    # Initialize the test
    peak_test = EmissionsPeakTest()
    
    # Load your data
    peak_test.load_historical_data('path/to/your/data.csv')  # or use simulate_data()
    
    # Characterize noise
    peak_test.characterize_noise(method='all_data')  # or 'segments'
    
    # Set test data
    test_data = [(2021, 36500), (2022, 35800), (2023, 35100)]
    peak_test.set_test_data(test_data)
    
    # Run the test
    results = peak_test.run_bootstrap_test(n_bootstrap=1000)
    
    # Visualize results
    peak_test.plot_analysis()
    
    # Get interpretation
    interpretation = peak_test.interpret_results()

Author: Neil Grant and Claire Fyson
"""

# =================================
# Module Imports
# =================================

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict, Optional, Callable, Union
import warnings

warnings.filterwarnings("ignore")

# =================================
# Define Key Class
# =================================


class EmissionsPeakTest:
    """
    A comprehensive statistical test for detecting emissions peaks.

    This class implements a bootstrap hypothesis test to determine if observed
    declining emissions represent a statistically significant trend beyond
    random year-to-year fluctuations.

    Attributes:
        historical_data (pd.DataFrame): Historical emissions data
        test_data (pd.DataFrame): Recent emissions data showing potential decline
        noise_params (Dict): Parameters of the fitted noise distribution
        noise_generator (Callable): Function to generate noise samples
        bootstrap_results (Dict): Results from the bootstrap hypothesis test
        residuals (np.ndarray): Residuals from trend fitting
    """

    def __init__(self):
        """
        Initialize the emissions peak test.

        Args:
            random_state: Random seed for reproducibility
        """
        self.historical_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.noise_params: Optional[Dict] = None
        self.noise_generator: Optional[Callable] = None
        self.bootstrap_results: Optional[Dict] = None
        self.residuals: Optional[pd.Series] = None
        self.trend_info: Optional[Dict] = None

    # =================================
    # A) Loading data
    # =================================

    def load_historical_data(
        self,
        data_source: Union[str, pd.DataFrame],
        year_col: str = "year",
        emissions_col: str = "emissions",
        year_range: range = range(1970, 2020),
    ) -> "EmissionsPeakTest":
        """
        Load historical emissions data.

        Args:
            data_source: Path to CSV file or pandas DataFrame
            year_col: Name of the year column
            emissions_col: Name of the emissions column

        Returns:
            Self for method chaining
        """
        if isinstance(data_source, str):
            # Load from file, which should be stored in the data folder
            data_path = Path(__file__).resolve().parent / f"../data/{data_source}"

            try:
                data = pd.read_csv(data_path)
            except Exception as e:
                raise ValueError(f"Could not load data from {data_path}: {e}")

        elif isinstance(data_source, pd.DataFrame):
            data = data_source.copy()

        else:
            raise ValueError("data_source must be a file path or DataFrame")

        # Validate columns
        if year_col not in data.columns or emissions_col not in data.columns:
            raise ValueError(
                f"Data must contain '{year_col}' and '{emissions_col}' columns"
            )

        # Standardize column names
        data = (
            data[[year_col, emissions_col]]
            .rename(columns={year_col: "year", emissions_col: "emissions"})
            .sort_values("year")
            .reset_index(drop=True)
        )

        # Filter to the selected years
        self.historical_data = data.loc[data["year"].isin(year_range)]

        # Validate data
        self._validate_historical_data()

        print(
            f"Loaded historical data: {self.historical_data['year'].min()}-{self.historical_data['year'].max()}"
        )
        print(f"Data points: {len(self.historical_data)}")

        return self

    def _validate_historical_data(self) -> None:
        """Validate the loaded historical data."""
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        if len(self.historical_data) < 10:
            raise ValueError("Need at least 10 years of historical data")

        if self.historical_data["emissions"].isna().any():
            raise ValueError("Historical data contains missing values")

        if (self.historical_data["emissions"] < 0).any():
            raise ValueError("Historical emissions cannot be negative")

    # =================================
    # B) Characterising the noise in the data
    # =================================

    def characterize_noise(
        self,
        method: str = "all_data",
        segment_length: int = 10,
        distribution: str = "normal",
        ignore_years: list = None,
        clip_distribution: tuple = None,
    ) -> "EmissionsPeakTest":
        """
        Characterize the noise distribution in historical emissions data.

        Args:
            method: 'all_data' or 'segments' for residual calculation
            segment_length: Length of segments if using 'segments' method
            distribution: 'normal' or 't' for noise distribution type

        Returns:
            Self for method chaining
        """
        if self.historical_data is None:
            raise ValueError("Must load historical data first")

        self.residuals, self.trend_info = self._calculate_residuals(
            method, segment_length
        )
        self.noise_params, self.noise_generator = self._fit_noise_distribution(
            ignore_years, clip_distribution, distribution
        )

        print("Noise characterization complete:")
        print(f"  Method: {method}")
        print(f"  Distribution: {distribution}")
        print(f"  Noise std: {self.noise_params['sigma']:.1f} Mt")
        print(f"  Residuals: {len(self.residuals)} points")

        return self

    def _calculate_residuals(
        self, method: str, segment_length: int
    ) -> Tuple[pd.Series, Dict]:
        """Calculate residuals from trend fitting."""
        residuals_list = []
        trend_info = {}

        if method == "all_data":
            X = self.historical_data["year"].values.reshape(-1, 1)
            y = self.historical_data["emissions"].values

            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            residuals = y - trend

            residuals_list.append(pd.Series(index=X.reshape(X.shape[0]),data=residuals))
            trend_info["all_data"] = {
                "year_min": X.min(),
                "year_max": X.max(),
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "r2": model.score(X, y),
                "n_points": len(y),
            }

        elif method == "segments":
            years = self.historical_data["year"].values
            emissions = self.historical_data["emissions"].values

            start_year = years[0]
            end_year = years[-1]

            for year_start in range(start_year, end_year, segment_length):
                year_end = min(year_start + segment_length, end_year)

                mask = (years >= year_start) & (years <= year_end)
                # Check that there are enough data points (>=3) in each segment to calculate a trend and residuals
                if np.sum(mask) < 3:
                    continue

                X_seg = years[mask].reshape(-1, 1)
                y_seg = emissions[mask]

                model = LinearRegression()
                model.fit(X_seg, y_seg)
                trend_seg = model.predict(X_seg)
                residuals_seg = y_seg - trend_seg

                residuals_list.append(
                    pd.Series(index=X_seg.reshape(X_seg.shape[0]),data=residuals_seg)
                )
                
                trend_info[f"{year_start}-{year_end}"] = {
                    "year_min": year_start,
                    "year_max": year_end,
                    "slope": model.coef_[0],
                    "intercept": model.intercept_,
                    "r2": model.score(X_seg, y_seg),
                    "n_points": len(y_seg),
                }
        else:
            raise ValueError("Method must be 'all_data' or 'segments'")

        residuals = pd.concat(residuals_list)
        return residuals, trend_info

    def _fit_noise_distribution(
            self,
            ignore_years: list = None,
            clip_distribution: tuple = None,
            noise_type: str = "auto"):
        """
        Fit noise distribution with proper parameter handling for both normal and t-distributions.

        Args:
            ignore_years [list]: List of specific years to ignore from the residuals 
            because they are too large a variation and could bias the results (e.g. the GFC or COVID-19)

            clip_distribution [tuple]: Clip the distribution of residuals to exclude
            outlying percentiles (e.g. <5 and >95) before calculating
            
            noise_type [str]: 'normal', 't', or 'auto'
                - 'normal': Fit normal distribution
                - 't': Fit t-distribution
                - 'auto': Test both and select better fit based on AIC

        Returns:
            params: Dictionary with standardized parameter names
            generator: Function to generate noise samples
        """
        if self.residuals is None:
            raise ValueError(
                "Must call characterize_noise() first to calculate residuals"
            )

        residuals = self.residuals

        # Only either drop specific years from the investigation, or clip the distribution
        # Don't do both
        if ignore_years:
            residuals = residuals.drop(ignore_years)

        elif clip_distribution:
            if len(clip_distribution) != 2:
                raise ValueError('Clip_distribution must be a tuple of type (float,float)')
            residuals = residuals[
                (residuals >= residuals.quantile(clip_distribution[0])) & 
                (residuals <= residuals.quantile(clip_distribution[1]))]

        if noise_type == "auto":
            # Test both distributions and choose better fit
            normal_params = stats.norm.fit(residuals)
            t_params = stats.t.fit(residuals)

            # Compare using AIC (lower is better)
            normal_aic = (
                -2 * np.sum(stats.norm.logpdf(residuals, *normal_params)) + 2 * 2
            )
            t_aic = -2 * np.sum(stats.t.logpdf(residuals, *t_params)) + 2 * 3

            if normal_aic <= t_aic:
                noise_type = "normal"
                print(
                    f"Auto-selected normal distribution (AIC: {normal_aic:.1f} vs {t_aic:.1f})"
                )
            else:
                noise_type = "t"
                print(
                    f"Auto-selected t-distribution (AIC: {t_aic:.1f} vs {normal_aic:.1f})"
                )

        # Fit the selected distribution with proper parameter handling
        if noise_type == "normal":
            params_tuple = stats.norm.fit(residuals)

            # Store with consistent naming - both distributions have 'mu', 'sigma', 'scale'
            params = {
                "type": "normal",
                "mu": params_tuple[0],  # Location parameter
                "sigma": params_tuple[1],  # Scale parameter (standard deviation)
                "scale": params_tuple[1],  # Alias for consistency
                "fitted_params": params_tuple,
            }

            # Create generator function specific to normal distribution
            def noise_generator(size):
                return stats.norm.rvs(
                    loc=params["mu"], scale=params["sigma"], size=size
                )

        elif noise_type == "t":
            params_tuple = stats.t.fit(residuals)  # Returns (df, loc, scale)

            # Store with consistent naming - ensure 'sigma' is available
            params = {
                "type": "t",
                "df": params_tuple[0],  # Degrees of freedom (unique to t-dist)
                "mu": params_tuple[1],  # Location parameter
                "scale": params_tuple[2],  # Scale parameter
                "sigma": params_tuple[
                    2
                ],  # Use scale as sigma equivalent for compatibility
                "fitted_params": params_tuple,
            }

            # Create generator function specific to t-distribution
            def noise_generator(size):
                return stats.t.rvs(
                    df=params["df"], loc=params["mu"], scale=params["scale"], size=size
                )

        else:
            raise ValueError("noise_type must be 'normal', 't', or 'auto'")

        # Store parameters and generator for later use
        self.noise_params = params
        self.noise_generator = noise_generator

        print(f"Fitted {params['type']} distribution:")
        if params["type"] == "normal":
            print(f"  μ (mean) = {params['mu']:.2f}")
            print(f"  σ (std)  = {params['sigma']:.2f}")
        else:
            print(f"  df (degrees of freedom) = {params['df']:.2f}")
            print(f"  μ (location) = {params['mu']:.2f}")
            print(f"  scale = {params['scale']:.2f}")

        return params, noise_generator

    def set_test_data(self, test_data: List[Tuple[int, float]]) -> "EmissionsPeakTest":
        """
        Set the test data (recent emissions showing potential decline).

        Args:
            test_data: List of (year, emissions) tuples

        Returns:
            Self for method chaining
        """

        self.test_data = pd.DataFrame(test_data, columns=["year", "emissions"])
        self.test_data = self.test_data.sort_values("year").reset_index(drop=True)

        # Calculate test trend
        self.test_slope, self.test_r2 = self._calculate_test_slope()

        print(
            f"Test data set: {self.test_data['year'].min()}-{self.test_data['year'].max()}"
        )
        print(f"Test slope: {self.test_slope:.2f} Mt/year (R² = {self.test_r2:.3f})")

        return self

    def _calculate_test_slope(self) -> Tuple[float, float]:
        """Calculate slope and R² for test data."""
        X = self.test_data["year"].values.reshape(-1, 1)
        y = self.test_data["emissions"].values

        model = LinearRegression()
        model.fit(X, y)

        return model.coef_[0], model.score(X, y)

    def run_bootstrap_test(self, n_bootstrap: int = 10000, alpha: float = 0.05) -> Dict:
        """
        Run a one-tailed bootstrap hypothesis test.

        Args:
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        if self.test_data is None:
            raise ValueError("Must set test data first")
        if self.noise_generator is None:
            raise ValueError("Must characterize noise first")

        print(f"Running bootstrap test with {n_bootstrap} samples...")

        # Generate bootstrap samples under null hypothesis
        bootstrap_slopes = self._generate_bootstrap_slopes(n_bootstrap)

        # Calculate p-values
        p_value_one_tail = self._calculate_p_value(bootstrap_slopes, self.test_slope)

        # Store results
        self.bootstrap_results = {
            "test_slope": self.test_slope,
            "test_r2": self.test_r2,
            "bootstrap_slopes": bootstrap_slopes,
            "p_value_one_tail": p_value_one_tail,
            "significant_one_tail": p_value_one_tail < alpha,
            "alpha": alpha,
            "n_bootstrap": n_bootstrap,
            "bootstrap_mean": np.mean(bootstrap_slopes),
            "bootstrap_std": np.std(bootstrap_slopes),
        }

        print("Bootstrap test complete:")
        print(f"  P-value (one-tail): {p_value_one_tail:.4f}")
        print(
            f"  Significant (α={alpha}): {self.bootstrap_results['significant_one_tail']}"
        )

        return

    def _generate_bootstrap_slopes(self, n_bootstrap: int) -> np.ndarray:
        """Generate bootstrap slopes under null hypothesis."""
        bootstrap_slopes = []
        n_points = len(self.test_data)

        # Use mean emissions as baseline for null hypothesis
        baseline_emissions = np.mean(self.test_data["emissions"])

        for _ in range(n_bootstrap):
            # Generate null hypothesis data (zero trend + noise)
            years = np.arange(2020, 2020 + n_points)  # Arbitrary years
            null_emissions = np.full(
                n_points, baseline_emissions
            ) + self.noise_generator(n_points)

            # Calculate slope
            X = years.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, null_emissions)
            bootstrap_slopes.append(model.coef_[0])

        return np.array(bootstrap_slopes)

    def _calculate_p_value(
        self, bootstrap_slopes: np.ndarray, observed_slope: float
    ) -> float:
        """Calculate one-tailed p-value."""
        # Testing for decline
        return np.sum(bootstrap_slopes <= observed_slope) / len(bootstrap_slopes)

    def interpret_results(self, verbose: bool = True) -> Dict[str, str]:
        """
        Provide interpretation of the test results.

        Args:
            verbose: Whether to print interpretation

        Returns:
            Dictionary with interpretation strings
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap test first")

        results = self.bootstrap_results

        # Determine trend direction
        if results["test_slope"] < 0:
            direction = "decline"
            trend_desc = "declining"
        else:
            direction = "increase"
            trend_desc = "increasing"

        # Statistical significance
        if results["significant_one_tail"]:
            if results["p_value_one_tail"] < 0.01:
                strength = "very strong"
            elif results["p_value_one_tail"] < 0.05:
                strength = "strong"
            else:
                strength = "moderate"
            significance = f"{strength} evidence"
        else:
            significance = "insufficient evidence"

        # Peak conclusion
        if direction == "decline" and results["significant_one_tail"]:
            peak_conclusion = "Strong evidence that CO₂ emissions have peaked"
            confidence = "high"
        elif direction == "decline" and not results["significant_one_tail"]:
            peak_conclusion = (
                "Declining trend present but not statistically significant"
            )
            confidence = "low"
        else:
            peak_conclusion = "No evidence of emissions peak"
            confidence = "none"

        interpretation = {
            "direction": direction,
            "trend_description": trend_desc,
            "significance": significance,
            "peak_conclusion": peak_conclusion,
            "confidence_in_peak": confidence,
            "p_value": f"{results['p_value_one_tail']:.4f}",
            "slope": f"{results['test_slope']:.1f} Mt/year",
        }

        if verbose:
            print("\n" + "=" * 50)
            print("INTERPRETATION OF RESULTS")
            print("=" * 50)
            print(f"Observed trend: {trend_desc} at {interpretation['slope']}")
            print(
                f"Statistical evidence: {significance} (p = {interpretation['p_value']})"
            )
            print(f"Conclusion: {peak_conclusion}")
            print("=" * 50)

        return interpretation

    def plot_analysis(
        self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive visualization of the analysis.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap test first")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Historical data with test data overlay
        self._plot_historical_and_test_data(axes[0, 0])

        # 2. Noise distribution
        self._plot_noise_distribution(axes[0, 1])

        # 3. Bootstrap results
        self._plot_bootstrap_results(axes[1, 0])

        # 4. Summary statistics
        self._plot_summary_statistics(axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return

    def _plot_historical_and_test_data(self, ax: plt.Axes) -> None:
        """Plot historical data with test data overlay and calculated trend lines."""
        ax.plot(
            self.historical_data["year"],
            self.historical_data["emissions"],
            "b-",
            alpha=0.7,
            label="Historical emissions",
        )
        ax.plot(
            self.test_data["year"],
            self.test_data["emissions"],
            "k+",
            linewidth=1,
            markersize=5,
            label="Recent test data",
        )

        for period in list(self.trend_info.keys()):
            trend_data = self.trend_info[period]
            x_trend = range(trend_data["year_min"], trend_data["year_max"] + 1)
            y_trend = trend_data["slope"] * x_trend + trend_data["intercept"]
            ax.plot(x_trend, y_trend)

            xmin = ax.get_xlim()[0]
            ymin = ax.get_ylim()[0]
            xmax = ax.get_xlim()[1]
            ymax = ax.get_ylim()[1]

            ax.text(
                (np.mean(x_trend) - xmin) / (xmax - xmin) + 0.02,
                (np.mean(y_trend) - ymin) / (ymax - ymin) - 0.02,
                "R$^{2}$ = " + "{0:.2f}".format(trend_data["r2"]),
                transform=ax.transAxes,
            )

        ax.set_xlabel("Year")
        ax.set_ylabel("CO₂ Emissions (Mt)")
        ax.set_title("Historical CO₂ Emissions and Test Data")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_noise_distribution(self, ax: plt.Axes) -> None:
        """Plot the fitted noise distribution."""
        ax.hist(
            self.residuals,
            bins=30,
            density=True,
            alpha=0.7,
            color="skyblue",
            label="Historical residuals",
        )

        # Overlay fitted distribution
        if self.noise_params["type"] == "normal":
            x_range = np.linspace(self.residuals.min(), self.residuals.max(), 100)
            fitted_density = stats.norm.pdf(
                x_range, self.noise_params["mu"], self.noise_params["sigma"]
            )
            ax.plot(x_range, fitted_density, "r-", linewidth=2, label="Fitted normal")

        ax.set_xlabel("Residuals (Mt)")
        ax.set_ylabel("Density")
        ax.set_title(f'Noise Distribution (σ = {self.noise_params["sigma"]:.0f} Mt)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bootstrap_results(self, ax: plt.Axes) -> None:
        """Plot bootstrap distribution with observed slope."""
        ax.hist(
            self.bootstrap_results["bootstrap_slopes"],
            bins=50,
            density=True,
            alpha=0.7,
            color="lightgreen",
            label="Bootstrap slopes\n(null hypothesis)",
        )

        ax.axvline(
            self.bootstrap_results["test_slope"],
            color="red",
            linewidth=2,
            label=f'Observed slope\n{self.bootstrap_results["test_slope"]:.1f} Mt/yr',
        )

        ax.set_xlabel("Slope (Mt/year)")
        ax.set_ylabel("Density")
        ax.set_title("Bootstrap Distribution vs Observed Slope")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_summary_statistics(self, ax: plt.Axes) -> None:
        """Plot summary statistics and interpretation."""
        ax.axis("off")

        results = self.bootstrap_results

        # Create summary text
        summary_text = f"""
        STATISTICAL TEST RESULTS
        ________________________
        
        Test Data: {len(self.test_data)} years
        Observed Slope: {results['test_slope']:.2f} Mt/year
        R²: {results['test_r2']:.3f}
        
        Bootstrap Analysis:
        • Samples: {results['n_bootstrap']:,}
        • P-value: {results['p_value_one_tail']:.4f}
        • Significant: {results['significant_one_tail']}
        
        Noise Characteristics:
        • Method: {list(self.trend_info.keys())[0] if len(self.trend_info) == 1 else 'segments'}
        • Std Dev: {self.noise_params['sigma']:.1f} Mt
        • Distribution: {self.noise_params['type']}
        
        CONCLUSION:
        """

        interpretation = self.interpret_results(verbose=False)
        summary_text += interpretation["peak_conclusion"]

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

    def export_results(self, filepath: str) -> None:
        """
        Export analysis results to CSV file.

        Args:
            filepath: Path for output CSV file
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap test first")

        # Prepare results data
        results_data = {"metric": [], "value": []}

        # Add test results
        results_data["metric"].extend(
            [
                "test_slope_mt_per_year",
                "test_r_squared",
                "p_value_one_tail",
                "p_value_two_tail",
                "significant_at_0.05",
                "n_bootstrap_samples",
                "noise_std_mt",
                "n_test_years",
                "test_year_start",
                "test_year_end",
            ]
        )

        results_data["value"].extend(
            [
                self.bootstrap_results["test_slope"],
                self.bootstrap_results["test_r2"],
                self.bootstrap_results["p_value_one_tail"],
                self.bootstrap_results["p_value_two_tail"],
                self.bootstrap_results["significant_one_tail"],
                self.bootstrap_results["n_bootstrap"],
                self.noise_params["sigma"],
                len(self.test_data),
                self.test_data["year"].min(),
                self.test_data["year"].max(),
            ]
        )

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")


# Example usage and demonstration
if __name__ == "__main__":
    print("Emissions Peak Detection Test")
    print("=" * 50)

    # Initialize test with random state for reproducibility
    peak_test = EmissionsPeakTest()

    # Method chaining example
    peak_test.load_historical_data(
        "gcb_hist_co2.csv",
        emissions_col="fossil_co2_emissions",
        year_range=range(1970, 2020),
    ).characterize_noise(
        method="segments",  # Try 'segments' for alternative approach
        distribution="normal",
    ).set_test_data(
        [
            (2025, 37700),
            (2026, 37400),
            (2027, 37100),
            # (2028, 37100)
        ]
    ).run_bootstrap_test(
        n_bootstrap=1000, alpha=0.05
    )

    # Get interpretation
    interpretation = peak_test.interpret_results(verbose=True)

    # Create visualizations
    # fig = peak_test.plot_analysis()

    # Export results
    # peak_test.export_results('_peak_test_results.csv')

    print("\nAnalysis complete!")
    print("\nKey methods:")
    print("• load_historical_data() or simulate_historical_data()")
    print("• characterize_noise()")
    print("• set_test_data()")
    print("• run_bootstrap_test()")
    print("• interpret_results()")
    print("• plot_analysis()")
    print("• export_results()")
