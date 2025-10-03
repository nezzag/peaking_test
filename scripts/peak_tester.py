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
    peak_test.load_historical_data('path/to/your/data.csv')
    
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

Authors: Neil Grant and Claire Fyson
"""

# =================================
# Module Imports
# =================================

import numpy as np
import pandas as pd
import pandas_indexing as pix
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS
from typing import List, Tuple, Dict, Optional, Callable, Union
from statsmodels.tsa.arima.model import ARIMA
import warnings
from itertools import combinations

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
        self.recent_historical_trend: Optional[float] = None
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
                data = pd.read_csv(data_path, index_col=[0,1,2])
            except Exception as e:
                raise ValueError(f"Could not load data from {data_path}: {e}")

        elif isinstance(data_source, pd.DataFrame):
            assert data_source.index.names == ['region','variable','unit']
            data = data_source.copy()

        else:
            raise ValueError("data_source must be a file path or DataFrame")

        self.variable = data.pix.unique('variable')[0]
        self.unit = data.pix.unique('unit')[0]
        # Cast into long-form and rename
        data.columns = data.columns.astype(int)
        data = pd.Series(index=data.columns,data=data.values.squeeze()).reset_index()
        data.columns = ['year','emissions']
        
        # # Validate columns
        # if year_col not in data.columns or emissions_col not in data.columns:
        #     raise ValueError(
        #         f"Data must contain '{year_col}' and '{emissions_col}' columns"
        #     )

        # # Standardize column names
        # data = (
        #     data[[year_col, emissions_col]]
        #     .rename(columns={year_col: "year", emissions_col: "emissions"})
        #     .sort_values("year")
        #     .reset_index(drop=True)
        # )

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
        method: str = "loess",
        ignore_years: list = None,
        **kwargs,
        # clip_distribution: tuple = None, #TODO: Clip noise distribution to e.g. 5-95 percentiles?
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
            method, ignore_years, **kwargs
        )

        self.autocorr_params = self._analyze_autocorrelation()
    

        print("Noise characterization complete:")
        print(f"  Method used: {method}")
        # print(f"  Distribution: {distribution}")
        # print(f"  Noise std: {self.noise_params['sigma']:.1f} Mt")
        # print(f"  Residuals: {len(self.residuals)} points")

        return self

    def _calculate_residuals(self, method: str, ignore_years: list = None, **kwargs) -> Tuple[pd.Series, Dict]:
        residuals_list = []
        trend_info = {}
    
        hist_data = self.historical_data.copy()
        if ignore_years:
            hist_data = hist_data.loc[~hist_data.year.isin(ignore_years)]

        years = hist_data["year"].values
        emissions = hist_data["emissions"].values

        if method in ["hp", "hodrick_prescott"]:
            # HP filter
            cycle, trend = hpfilter(emissions, lamb=100)  #100 is typical for annual data
            residuals = emissions - trend
            residuals_list.append(pd.Series(index=years, data=residuals))
            trend_info = {
                'method': 'Hodrick-Prescott filter',
                "trend": pd.Series(trend, index=years),
                "n_points": len(emissions),
                'parameters': {'lambda': 100}
            }

        elif method == "hamilton":
            """Fit Hamilton's regression-based method."""
            
            # Horizon ahead at which we calculate y, can be specified, otherwise default 4
            h = kwargs.get('h',4)
            # Length of timeseries that we use to predict y -> this is fixed by Hamilton's method
            p = 4
            n = len(years)
            y = emissions.copy()
            
            if n <= p:
                return None
            
            # Build regression matrix
            # Both X and Y are reduced in size by p + h
            # (you can't forecast before p+h: into the data, as you don't have sufficient data)
            X = np.ones((n - p - h, p))
            Y = y[p+h:]
            
            for i in range(p):
                lag = p - i
                X[:, i] = y[p + h - lag : n - lag]
            
            # Fit regression
            model = OLS(Y, X).fit()

            
            # Create full series
            full_trend = np.full(n, np.nan)
            full_cycle = np.full(n, np.nan)
            
            full_trend[p+h:] = model.fittedvalues
            full_cycle[p+h:] = model.resid

            residuals_list.append(pd.Series(index=years[p+h:], data=model.resid))

            # Fill initial periods (Neil: I removed this as there is no real way to backfill residuals)
            # if max_lag > 0:
            #     trend_slope = (model.fittedvalues[1] - model.fittedvalues[0]) if len(model.fittedvalues) > 1 else 0
            #     for i in range(max_lag):
            #         full_trend[i] = model.fittedvalues[0] - trend_slope * (max_lag - i)
            #         full_cycle[i] = y[i] - full_trend[i]
            
            trend_info = {
                'method': 'Hamilton (2018)',
                'trend': pd.Series(full_trend, index=years),
                # 'cycle': pd.Series(full_cycle, index=self.historical_data.index),
                'parameters': {'h': h, 'p': p},
                'parameter_info': f'h={h} (forecast horizon), p={p} (lags)',
                'r_squared': model.rsquared,
                'model': model
            }

        elif method == "loess":
            # LOESS smoothing
            frac =  kwargs.get('fraction',0.3)  # Smoothing parameter
            smoothed = lowess(emissions, years, frac=frac, return_sorted=False)
            residuals = emissions - smoothed
            residuals_list.append(pd.Series(index=years, data=residuals))
            trend_info = {
                "method": 'LOESS',
                "trend": smoothed,
                "parameter_info": {'fraction':frac},
                "n_points": len(smoothed),
            }
        else:
            raise ValueError("Method must be one of 'hp', 'hamilton', or 'loess'")

        residuals = pd.concat(residuals_list)
        return residuals, trend_info

    def _analyze_autocorrelation(self) -> Dict:
        """
        Analyze autocorrelation in temporally-ordered residuals.
        Parameters:
        - data_version: 'raw', 'excluded', 'interpolated' 
        - use_segmentation_data: If True, use same data as segmentation
        """

        residuals = self.residuals.values
    
        # Calculate lag-1 autocorrelation via the acf function
        # (this provides additional statistical outputs if desired, but not the residuals)
        acf_values, _, pvalues = acf(residuals, nlags=5, qstat=True, fft=True)
        phi = acf_values[1] if len(acf_values) > 1 else 0
        p_autocorr = pvalues[1] if len(acf_values) > 1 else 1

        # Simple AR(1) model fitting
        if len(residuals) > 1:
            y = residuals[1:]
            X = residuals[:-1].reshape(-1, 1)
            
            ar_model = LinearRegression() #TODO: Currently has non-zero intercept, which is weird...
            ar_model.fit(X, y)
            phi_fitted = ar_model.coef_[0]

            # Check that manual fit gives similar to the ACF results
            # assert np.isclose(phi_fitted, phi, rtol=1e-2)

            #Innovation residuals are the residuals left after accounting for autocorrelation
            innovation_residuals = y - ar_model.predict(X)
            sigma_innovation = np.std(innovation_residuals)
            mean_innovation = np.mean(innovation_residuals)
        else:
            phi_fitted = 0
            sigma_innovation = np.std(self.residuals)
            mean_innovation = np.mean(self.residuals)
        
        #TODO: Add more checks on autocorrelation, and explore this in more detail
        self.autocorr_params = {
            'phi': phi_fitted,
            'residuals': innovation_residuals,
            'sigma_residuals': sigma_innovation,
            'mean_residuals': mean_innovation,
            'has_autocorr': abs(phi_fitted) > 0.1,
            'is_stationary': abs(phi_fitted) < 1,
            'likelihood_of_autocorr': 1 - p_autocorr
        }
        
        print(f"Autocorrelation analysis:")
        print(f"  Lag-1 autocorr: {phi_fitted:.3f}")
        print(f"  Residual σ (post-autocorrelation): {sigma_innovation:.1f}")
        print(f"  Has significant autocorr: {self.autocorr_params['has_autocorr']}")
        print(f"  Likelihood of autocorr: {self.autocorr_params['likelihood_of_autocorr']}")
        
        return self.autocorr_params

    # =================================
    # C) Create a noise generator that can randomly reproduce noise
    # =================================

    def create_noise_generator(
            self):
        """
        Create noise generator

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
    
        if self.autocorr_params is None:
            self._analyze_autocorrelation()
  
        #TODO: Currently only loads up params from autocorrelation analysis -> if there is limited autocorrelation
        # should we load a non-autocorrelated statistical analysis? (TBD)
        phi = self.autocorr_params['phi']
        sigma = self.autocorr_params['sigma_residuals']
        mean = self.autocorr_params['mean_residuals']
        residuals_post_ar = self.autocorr_params['residuals']
        
        if self.autocorr_params['has_autocorr'] and self.autocorr_params['is_stationary']:
            print(f"Using AR(1) noise generator with φ={phi:.3f}")
            
            def ar1_noise_generator(size: int, initial_value: float | None = 0) -> np.ndarray:
                """Generate AR(1) autocorrelated noise."""
                #TODO: Think about t-distribution here instead of normal, and allowing non-zero mean
                innovations = np.random.normal(0, sigma, size)
                series = np.zeros(size)
                if initial_value is None:
                    # Draw a random start value based on the historical residuals
                    series[0] = np.random.normal(0, self.residuals.abs().mean())
                else:
                    series[0] = initial_value
                
                for t in range(1, size):
                    series[t] = phi * series[t-1] + innovations[t]
                
                return series
            
            self.noise_generator = ar1_noise_generator
        
        else:
            print(f"Using white noise generator with σ={sigma:.1f}")
            
            def white_noise_generator(size: int, initial_value: float = 0) -> np.ndarray:
                """Generate white noise."""
                return np.random.normal(0, sigma, size)
            
            self.noise_generator = white_noise_generator
        
        return self.noise_generator
    
    def set_test_data(self, test_data: List[Tuple[int, float]], recent_years_for_trend: int = 10) -> "EmissionsPeakTest":
        """
        Set the test data (recent emissions showing potential decline).

        Args:
            test_data: List of (year, emissions) tuples

        Returns:
            Self for method chaining
        """

        recent_data = self.historical_data.tail(recent_years_for_trend)
        X_recent = recent_data["year"].values.reshape(-1, 1)
        y_recent = recent_data["emissions"].values
        
        model_recent = LinearRegression()
        model_recent.fit(X_recent, y_recent)
        self.recent_historical_trend = model_recent.coef_[0]
        

        self.test_data = pd.DataFrame(test_data, columns=["year", "emissions"])
        self.test_data = self.test_data.sort_values("year").reset_index(drop=True)

        # Calculate test trend
        self.test_slope, self.test_r2 = self._calculate_test_slope()

        print(
            f"Test data set: {self.test_data['year'].min()}-{self.test_data['year'].max()}"
        )
        print(f"Test slope: {self.test_slope:.2f} {self.unit} (R² = {self.test_r2:.3f})")
        print(f"Recent historical trend: {self.recent_historical_trend:.2f} {self.unit}")

        return self

    def _calculate_test_slope(self) -> Tuple[float, float]:
        """Calculate slope and R² for test data."""
        X = self.test_data["year"].values.reshape(-1, 1)
        y = self.test_data["emissions"].values

        model = LinearRegression()
        model.fit(X, y)

        return model.coef_[0], model.score(X, y)

    def run_complete_bootstrap_test(self, n_bootstrap: int = 10000, 
                                   null_hypothesis: str | float = "zero_trend",
                                   bootstrap_method: str = "ar_bootstrap") -> Dict:
        """
        Run complete bootstrap test with all enhancements.
        
        Args:
            null_hypothesis: Three options:
                - "recent_trend" (testing if new data is consistent with recent trend)
                - "zero_trend" (testing if new data is consistent with a zero trend)
                - float: Give a specific trend to test if new data is consistent with
            bootstrap_method: "ar_bootstrap", "block_bootstrap", or "white_noise"
        """
        if self.noise_generator is None:
            self.create_noise_generator()
        
        print(f"Running complete bootstrap test...")
        if isinstance(null_hypothesis, str):
            print(f"  Null hypothesis: {null_hypothesis}")
        else: 
            print(f"  Null hypothesis: trend of {null_hypothesis} / yr")
        print(f"  Bootstrap method: {bootstrap_method}")
        print(f"  Bootstrap samples: {n_bootstrap}")
        
        bootstrap_slopes = self._generate_bootstrap_slopes(
            n_bootstrap, null_hypothesis, bootstrap_method
        )
        
        # Calculate p-values
        p_value_one_tail = np.sum(bootstrap_slopes <= self.test_slope) / len(bootstrap_slopes)
        
        # Effect size: how many standard deviations below the null distribution
        null_mean = np.mean(bootstrap_slopes)
        null_std = np.std(bootstrap_slopes)
        effect_size = (null_mean - self.test_slope) / null_std if null_std > 0 else 0
        
        self.bootstrap_results = {
            'test_slope': self.test_slope,
            'test_r2': self.test_r2,
            'recent_historical_trend': self.recent_historical_trend,
            'bootstrap_slopes': bootstrap_slopes,
            'p_value_one_tail': p_value_one_tail,
            'significant_at_0.1': p_value_one_tail < 0.1,
            'significant_at_0.05': p_value_one_tail < 0.05,
            'significant_at_0.01': p_value_one_tail < 0.01,
            'null_hypothesis': null_hypothesis,
            'bootstrap_method': bootstrap_method,
            'effect_size': effect_size,
            'null_mean': null_mean,
            'null_std': null_std,
            'n_bootstrap': n_bootstrap,
            'autocorr_phi': self.autocorr_params['phi'],
            # 'n_segments': len(self.optimal_segments['best']['segments'])
        }
        
        print(f"Results:")
        print(f"  P-value: {p_value_one_tail:.4f}")
        print(f"  Significant at α=0.05: {self.bootstrap_results['significant_at_0.05']}")
        print(f"  Effect size: {effect_size:.2f} standard deviations")
        
        return
    
    def _generate_bootstrap_slopes(self, n_bootstrap: int, null_hypothesis: str | float, 
                                  bootstrap_method: str) -> np.ndarray:
        """Generate bootstrap slope distribution."""
        bootstrap_slopes = []
        n_test_points = len(self.test_data)

        years = self.test_data.year.values # Arbitrary years for test
        
        # Baseline emissions level
        baseline_emissions = np.mean(self.test_data["emissions"])
        
        # Null hypothesis trend
        if null_hypothesis == "recent_trend":
            null_trend = self.recent_historical_trend
        elif isinstance(null_hypothesis,float):
            null_trend = null_hypothesis
        else:  # zero_trend
            null_trend = 0.0
        
        for i in range(n_bootstrap):
            # Generate null hypothesis emissions trajectory
            trend_component = null_trend * (years - years[0])
            base_emissions = baseline_emissions + trend_component
            
            # Add autocorrelated noise
            if bootstrap_method == "ar_bootstrap" and self.autocorr_params['has_autocorr']:
                # Use AR(1) noise generator. This creates the noise for all n_test_points in one go
                noise = self.noise_generator(n_test_points, initial_value=None)
            else:  # white_noise
                noise = np.random.normal(0, self.autocorr_params['sigma_residuals'], n_test_points)
            
            null_emissions = base_emissions + noise
            
            # Calculate slope
            X = years.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, null_emissions)
            bootstrap_slopes.append(model.coef_[0])
        
        return np.array(bootstrap_slopes)

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
        strength = None

        # Determine trend direction
        if results["test_slope"] < 0:
            direction = "decline"
            trend_desc = "declining"
        else:
            direction = "increase"
            trend_desc = "increasing"

        # Statistical significance
        if results["significant_at_0.01"]:
            strength = "very strong"
        elif results["significant_at_0.05"]:
            strength = "strong"
        elif results["significant_at_0.1"]:
            strength = "moderate"
        else:
            strength = "insufficient"

        significance = f"{strength} evidence"

        # Peak conclusion
        if direction == "decline" and strength != "insufficient":
            peak_conclusion = "Evidence that CO₂ emissions have peaked"
            confidence = strength
        elif direction == "decline":
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
            # "significance": significance,
            "peak_conclusion": peak_conclusion,
            "confidence_in_peak": confidence,
            "p_value": f"{results['p_value_one_tail']:.4f}",
            "slope": f"{results['test_slope']:.1f} {self.unit}",
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

        # fig, axes = plt.subplots(2, 3, figsize=figsize)

        f = plt.figure(layout=None, figsize=(15,15))
        gs = f.add_gridspec(nrows=16, ncols=7, left=0.05, right=0.75,
                            hspace=0.1, wspace=0.05)

        a0 = f.add_subplot(gs[:5, :3])
        a1 = f.add_subplot(gs[:2, 4:])
        a2 = f.add_subplot(gs[3:5, 4:])
        a3 = f.add_subplot(gs[6:10, :3])
        a4 = f.add_subplot(gs[6:10, 4:])
        a5 = f.add_subplot(gs[11:, :])
        axes = [a0, a1, a2, a3, a4, a5]


        # 1. Historical data with test data overlay
        self._plot_historical_and_test_data(axes[0])

        # 2. Split of historical data into noise vs. signal
        self._plot_historical_trend(axes[1])
        self._plot_historical_noise(axes[2])

        # 3. Noise distribution
        self._plot_noise_distribution(axes[3])

        # 4. Bootstrap results
        self._plot_bootstrap_results(axes[4])

        # 5. Summary statistics
        self._plot_summary_statistics(axes[5])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return

    def _plot_historical_and_test_data(self, ax: plt.Axes) -> None:
        """Plot historical data with test data overlay"""
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

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{self.variable}\n({self.unit})")
        ax.set_title("Historical data and Test data")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_historical_trend(self, ax: plt.Axes) -> None:
        """Plot historical trend data"""
        ax.plot(
            self.historical_data["year"],
            self.trend_info["trend"],
            "b-",
            alpha=0.7,
            label="Historical emissions",
        )

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{self.variable}\n({self.unit})")
        ax.set_title(f"Historical Trend on {self.variable}")
        ax.legend()
        ax.grid(True, alpha=0.3)


    def _plot_historical_noise(self, ax: plt.Axes) -> None:
        """Plot historical noise data"""
        self.residuals.plot(
            ax=ax,
            label="Historical emissions",
            color="orange",
            style="-"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{self.variable}\n({self.unit})")
        ax.set_title(f"Historical Residuals on {self.variable}")
        ax.legend()
        ax.axhline(y=0,color='k',lw=1)
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
        #TODO: Does this make sense to do with an autocorrelated system?
        # if self.noise_params["type"] == "normal":
        #     x_range = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        #     fitted_density = stats.norm.pdf(
        #         x_range, self.noise_params["mu"], self.noise_params["sigma"]
        #     )
        #     ax.plot(x_range, fitted_density, "r-", linewidth=2, label="Fitted normal")

        ax.set_xlabel(f"Residuals {self.unit}")
        ax.set_ylabel("Density")
        ax.set_title(f'Noise Distribution') #(σ = {self.noise_params["sigma"]:.0f} Mt)')
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
            label=f'Observed slope\n{self.bootstrap_results["test_slope"]:.1f} {self.unit}',
        )

        ax.set_xlabel(f"Slope: {self.unit}")
        ax.set_ylabel("Density")
        ax.set_title("Bootstrap Distribution vs Observed Slope")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_summary_statistics(self, ax: plt.Axes, signif_level: float = 0.1) -> None:
        """Plot summary statistics and interpretation."""
        ax.axis("off")

        results = self.bootstrap_results

        # Create summary text
        summary_text = f"""
        STATISTICAL TEST RESULTS
        ________________________
        
        Test Data: {len(self.test_data)} years
        Observed Slope: {results['test_slope']:.2f} {self.unit}
        R²: {results['test_r2']:.3f}
        
        Bootstrap Analysis:
        • Samples: {results['n_bootstrap']:,}
        • P-value: {results['p_value_one_tail']:.4f}
        • Significant: {results[f'significant_at_{signif_level}']}
        
        Noise Characteristics:
        • Method: {self.trend_info['method']}
        • Autocorrelation present: {self.autocorr_params['has_autocorr']}
        • Autocorrelation: {self.autocorr_params['phi']}
        • Std Dev: {self.autocorr_params['sigma_residuals']:.1f} {self.unit}
        
        CONCLUSION:
        """

        interpretation = self.interpret_results(verbose=False)
        summary_text += interpretation["peak_conclusion"]

        ax.text(
            0.5,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="center",
            fontfamily="monospace",
        )

    def export_results(self, filepath: str, signif_level: float=0.1) -> None:
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
                "significance",
                "n_bootstrap_samples",
                # "noise_std_mt",
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
                self.bootstrap_results[f'significant_at_{signif_level}'],
                self.bootstrap_results["n_bootstrap"],
                # self.noise_params["sigma"],
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
        "fossil_intensity.csv",
        # emissions_col="fossil_co2_emissions",
        year_range=range(1970, 2024),
    )
    
    peak_test.characterize_noise(method="loess")
    peak_test.create_noise_generator()
    peak_test.set_test_data(
        [
            (2025, 37700),
            (2026, 37400),
            (2027, 37100),
            # (2028, 37400)
        ]
    )
    peak_test.run_complete_bootstrap_test(bootstrap_method='ar_bootstrap')

    # Get interpretation
    interpretation = peak_test.interpret_results(verbose=True)

    # Create visualizations
    peak_test.plot_analysis()

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



