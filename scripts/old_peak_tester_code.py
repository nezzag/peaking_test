# Alternative coding for an enhanced emissions test - created on August 25, superceded by 'peak_tester2.py'

# =================================
# Define enhanced class
# =================================

# Enhanced Emissions Peak Test
# Improvements:
# 1. Dynamic segment length optimization
# 2. Autocorrelation handling
# 3. Recent historical trend null hypothesis


class EnhancedEmissionsPeakTest:
    """
    Enhanced emissions peak test with variable segments and autocorrelation handling.
    """
    
    def __init__(self):
        self.historical_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.noise_params: Optional[Dict] = None
        self.noise_generator: Optional[Callable] = None
        self.bootstrap_results: Optional[Dict] = None
        self.residuals: Optional[pd.Series] = None
        self.trend_info: Optional[Dict] = None
        self.autocorr_results: Optional[Dict] = None
        self.optimal_segments: Optional[Dict] = None
        
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
    
    def optimize_segment_lengths(self, min_segment: int = 5, max_segment: int = 20, 
                                max_segments: int = 10) -> Dict:
        """
        Find optimal variable segment lengths by minimizing within-segment residual variance
        while ensuring adequate sample size for noise characterization.
        """
        years = self.historical_data["year"].values
        emissions = self.historical_data["emissions"].values
        n_years = len(years)
        
        print("Optimizing segment lengths...")
        
        # Try different segmentation approaches
        best_segmentation = None
        best_score = float('inf')
        segmentations_tested = []
        
        # Method 1: Equal length segments of varying sizes
        for seg_len in range(min_segment, min(max_segment + 1, n_years // 2)):
            segments = self._create_equal_segments(years, emissions, seg_len)
            if len(segments) > max_segments:
                continue
                
            score = self._evaluate_segmentation(segments)
            segmentations_tested.append({
                'method': 'equal',
                'segment_length': seg_len,
                'n_segments': len(segments),
                'score': score,
                'segments': segments
            })
            
            if score < best_score:
                best_score = score
                best_segmentation = segmentations_tested[-1]
        
        # Method 2: Dynamic segmentation based on structural breaks
        dynamic_segments = self._create_dynamic_segments(years, emissions, min_segment)
        if len(dynamic_segments) <= max_segments:
            score = self._evaluate_segmentation(dynamic_segments)
            segmentations_tested.append({
                'method': 'dynamic',
                'n_segments': len(dynamic_segments),
                'score': score,
                'segments': dynamic_segments
            })
            
            if score < best_score:
                best_score = score
                best_segmentation = segmentations_tested[-1]
        
        # Method 3: Overlapping segments (for comparison)
        overlap_segments = self._create_overlapping_segments(years, emissions, 
                                                            segment_length=12, overlap=4)
        if len(overlap_segments) <= max_segments:
            score = self._evaluate_segmentation(overlap_segments)
            segmentations_tested.append({
                'method': 'overlapping',
                'segment_length': 12,
                'overlap': 4,
                'n_segments': len(overlap_segments),
                'score': score,
                'segments': overlap_segments
            })
            
            if score < best_score:
                best_score = score
                best_segmentation = segmentations_tested[-1]
        
        self.optimal_segments = {
            'best': best_segmentation,
            'all_tested': segmentations_tested,
            'selection_criteria': 'minimum_pooled_residual_variance'
        }
        
        print(f"Optimal segmentation: {best_segmentation['method']} method")
        print(f"Number of segments: {best_segmentation['n_segments']}")
        print(f"Score (residual variance): {best_score:.2f}")
        
        return self.optimal_segments
    
    def _create_equal_segments(self, years: np.ndarray, emissions: np.ndarray, 
                              segment_length: int) -> List[Dict]:
        """Create equal-length segments."""
        segments = []
        for i in range(0, len(years), segment_length):
            end_idx = min(i + segment_length, len(years))
            if end_idx - i >= 3:  # Need at least 3 points for regression
                segments.append({
                    'years': years[i:end_idx],
                    'emissions': emissions[i:end_idx],
                    'start_year': years[i],
                    'end_year': years[end_idx-1]
                })
        return segments
    
    def _create_dynamic_segments(self, years: np.ndarray, emissions: np.ndarray,
                                min_segment: int) -> List[Dict]:
        """
        Create variable-length segments based on structural breaks detected
        through rolling R² analysis.
        """
        segments = []
        start_idx = 0
        
        while start_idx < len(years):
            best_end_idx = start_idx + min_segment
            best_end_idx = min(best_end_idx, len(years))
            best_r2 = 0
            
            # Find the segment length that maximizes R² for this starting point
            for end_idx in range(start_idx + min_segment, len(years) + 1):
                if end_idx - start_idx > 25:  # Don't make segments too long
                    break
                    
                X = years[start_idx:end_idx].reshape(-1, 1)
                y = emissions[start_idx:end_idx]
                
                if len(X) >= 3:
                    model = LinearRegression()
                    model.fit(X, y)
                    r2 = model.score(X, y)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_end_idx = end_idx
            
            # Create segment
            segments.append({
                'years': years[start_idx:best_end_idx],
                'emissions': emissions[start_idx:best_end_idx],
                'start_year': years[start_idx],
                'end_year': years[best_end_idx-1],
                'r2': best_r2
            })
            
            start_idx = best_end_idx
        
        return segments
    
    def _create_overlapping_segments(self, years: np.ndarray, emissions: np.ndarray,
                                   segment_length: int, overlap: int) -> List[Dict]:
        """Create overlapping segments."""
        segments = []
        step = segment_length - overlap
        
        for i in range(0, len(years) - segment_length + 1, step):
            end_idx = i + segment_length
            segments.append({
                'years': years[i:end_idx],
                'emissions': emissions[i:end_idx],
                'start_year': years[i],
                'end_year': years[end_idx-1]
            })
            
        return segments
    
    def _evaluate_segmentation(self, segments: List[Dict]) -> float:
        """
        Evaluate segmentation quality based on pooled residual variance
        and segment characteristics.
        """
        all_residuals = []
        total_points = 0
        weighted_r2 = 0
        
        for segment in segments:
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            r2 = model.score(X, y)
            
            all_residuals.extend(residuals)
            total_points += len(y)
            weighted_r2 += r2 * len(y)
        
        pooled_variance = np.var(all_residuals)
        avg_r2 = weighted_r2 / total_points
        
        # Score combines residual variance (lower is better) with R² penalty
        # We want low residual variance but also reasonable fit quality
        score = pooled_variance * (2 - avg_r2)  # Penalty increases if R² is low
        
        return score
    
    def analyze_autocorrelation(self, max_lags: int = 10) -> Dict:
        """
        Analyze autocorrelation in residuals and determine appropriate
        correction methods.
        """
        if self.optimal_segments is None:
            raise ValueError("Must optimize segments first")
        
        # Calculate residuals from optimal segmentation
        segments = self.optimal_segments['best']['segments']
        all_residuals = []
        residuals_with_years = []
        
        for segment in segments:
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            
            all_residuals.extend(residuals)
            for i, year in enumerate(segment['years']):
                residuals_with_years.append((year, residuals[i]))
        
        # Sort by year to ensure proper temporal order
        residuals_with_years.sort()
        temporal_residuals = np.array([r[1] for r in residuals_with_years])
        
        # Calculate autocorrelation function
        acf_values = acf(temporal_residuals, nlags=max_lags, fft=True)
        
        # Ljung-Box test for serial correlation
        lb_test = acorr_ljungbox(temporal_residuals, lags=max_lags, return_df=True)
        
        # Determine if significant autocorrelation exists
        significant_lags = []
        for lag in range(1, max_lags + 1):
            if abs(acf_values[lag]) > 1.96 / np.sqrt(len(temporal_residuals)):
                significant_lags.append(lag)
        
        self.autocorr_results = {
            'acf_values': acf_values,
            'ljung_box_test': lb_test,
            'significant_lags': significant_lags,
            'has_autocorrelation': len(significant_lags) > 0,
            'temporal_residuals': temporal_residuals,
            'max_autocorr_lag1': abs(acf_values[1]) if len(acf_values) > 1 else 0
        }
        
        print(f"Autocorrelation analysis complete:")
        print(f"  Lag-1 autocorrelation: {acf_values[1]:.3f}")
        print(f"  Significant lags: {significant_lags}")
        print(f"  Has significant autocorrelation: {self.autocorr_results['has_autocorrelation']}")
        
        return self.autocorr_results
    
    def characterize_noise(self, distribution: str = "auto", 
                          correct_autocorr: bool = True) -> "EnhancedEmissionsPeakTest":
        """
        Characterize noise using optimal segments and autocorrelation correction.
        """
        if self.optimal_segments is None:
            self.optimize_segment_lengths()
        
        if self.autocorr_results is None:
            self.analyze_autocorrelation()
        
        # Get residuals from optimal segmentation
        segments = self.optimal_segments['best']['segments']
        all_residuals = []
        
        for segment in segments:
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            all_residuals.extend(residuals)
        
        self.residuals = pd.Series(all_residuals)
        
        # Apply autocorrelation correction if needed
        if correct_autocorr and self.autocorr_results['has_autocorrelation']:
            print("Applying autocorrelation correction...")
            effective_residuals = self._apply_autocorr_correction(self.residuals)
        else:
            effective_residuals = self.residuals
        
        # Fit noise distribution
        self.noise_params, self.noise_generator = self._fit_noise_distribution(
            effective_residuals, distribution
        )
        
        print("Enhanced noise characterization complete:")
        print(f"  Segments: {self.optimal_segments['best']['n_segments']}")
        print(f"  Autocorrelation correction: {correct_autocorr and self.autocorr_results['has_autocorrelation']}")
        print(f"  Noise std: {self.noise_params['sigma']:.1f} Mt")
        
        return self
    
    def _apply_autocorr_correction(self, residuals: pd.Series) -> np.ndarray:
        """
        Apply autocorrelation correction by pre-whitening or effective sample size adjustment.
        """
        # Method 1: Pre-whitening (removing AR(1) component)
        if len(self.autocorr_results['significant_lags']) > 0:
            lag1_corr = self.autocorr_results['acf_values'][1]
            
            # Simple AR(1) pre-whitening
            temporal_residuals = self.autocorr_results['temporal_residuals']
            prewhitened = temporal_residuals[1:] - lag1_corr * temporal_residuals[:-1]
            
            # Scale up variance to account for correlation
            variance_inflation = 1 / (1 - lag1_corr**2)
            corrected_residuals = prewhitened * np.sqrt(variance_inflation)
            
            return corrected_residuals
        
        return residuals.values
    
    def _fit_noise_distribution(self, residuals: np.ndarray, 
                               noise_type: str = "auto") -> Tuple[Dict, Callable]:
        """Fit noise distribution with autocorrelation considerations."""
        
        if noise_type == "auto":
            # Test both distributions
            normal_params = stats.norm.fit(residuals)
            t_params = stats.t.fit(residuals)
            
            normal_aic = -2 * np.sum(stats.norm.logpdf(residuals, *normal_params)) + 2 * 2
            t_aic = -2 * np.sum(stats.t.logpdf(residuals, *t_params)) + 2 * 3
            
            noise_type = "normal" if normal_aic <= t_aic else "t"
            print(f"Auto-selected {noise_type} distribution")
        
        if noise_type == "normal":
            params_tuple = stats.norm.fit(residuals)
            params = {
                "type": "normal",
                "mu": params_tuple[0],
                "sigma": params_tuple[1],
                "scale": params_tuple[1],
                "fitted_params": params_tuple,
            }
            
            def noise_generator(size):
                return stats.norm.rvs(loc=params["mu"], scale=params["sigma"], size=size)
        
        elif noise_type == "t":
            params_tuple = stats.t.fit(residuals)
            params = {
                "type": "t",
                "df": params_tuple[0],
                "mu": params_tuple[1],
                "scale": params_tuple[2],
                "sigma": params_tuple[2],
                "fitted_params": params_tuple,
            }
            
            def noise_generator(size):
                return stats.t.rvs(df=params["df"], loc=params["mu"], 
                                 scale=params["scale"], size=size)
        
        return params, noise_generator
    
    def set_test_data(self, test_data: List[Tuple[int, float]]) -> "EnhancedEmissionsPeakTest":
        """Set test data and calculate recent historical trend for null hypothesis."""
        self.test_data = pd.DataFrame(test_data, columns=["year", "emissions"])
        self.test_data = self.test_data.sort_values("year").reset_index(drop=True)
        
        # Calculate test trend
        X = self.test_data["year"].values.reshape(-1, 1)
        y = self.test_data["emissions"].values
        model = LinearRegression()
        model.fit(X, y)
        
        self.test_slope = model.coef_[0]
        self.test_r2 = model.score(X, y)
        
        # Calculate recent historical trend for null hypothesis
        self.recent_historical_trend = self._calculate_recent_historical_trend()
        
        print(f"Test data set: {self.test_data['year'].min()}-{self.test_data['year'].max()}")
        print(f"Test slope: {self.test_slope:.2f} Mt/year (R² = {self.test_r2:.3f})")
        print(f"Recent historical trend: {self.recent_historical_trend:.2f} Mt/year")
        
        return self
    
    def _calculate_recent_historical_trend(self, n_recent_years: int = 10) -> float:
        """Calculate trend from recent historical data for null hypothesis."""
        recent_data = self.historical_data.tail(n_recent_years)
        
        X = recent_data["year"].values.reshape(-1, 1)
        y = recent_data["emissions"].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.coef_[0]
    
    def run_enhanced_bootstrap_test(self, n_bootstrap: int = 10000, 
                                  null_hypothesis: str = "recent_trend",
                                  alpha: float = 0.05) -> Dict:
        """
        Run enhanced bootstrap test with choice of null hypothesis.
        
        Args:
            null_hypothesis: "zero_trend" or "recent_trend"
        """
        if self.test_data is None:
            raise ValueError("Must set test data first")
        if self.noise_generator is None:
            raise ValueError("Must characterize noise first")
        
        print(f"Running enhanced bootstrap test ({null_hypothesis} null hypothesis)...")
        
        if null_hypothesis == "recent_trend":
            bootstrap_slopes = self._generate_bootstrap_slopes_recent_trend(n_bootstrap)
        else:  # zero_trend
            bootstrap_slopes = self._generate_bootstrap_slopes_zero_trend(n_bootstrap)
        
        # Calculate p-values
        p_value_one_tail = np.sum(bootstrap_slopes <= self.test_slope) / len(bootstrap_slopes)
        
        self.bootstrap_results = {
            "test_slope": self.test_slope,
            "test_r2": self.test_r2,
            "bootstrap_slopes": bootstrap_slopes,
            "p_value_one_tail": p_value_one_tail,
            "significant_one_tail": p_value_one_tail < alpha,
            "alpha": alpha,
            "n_bootstrap": n_bootstrap,
            "null_hypothesis": null_hypothesis,
            "recent_historical_trend": getattr(self, 'recent_historical_trend', 0),
            "bootstrap_mean": np.mean(bootstrap_slopes),
            "bootstrap_std": np.std(bootstrap_slopes),
        }
        
        print(f"Enhanced bootstrap test complete:")
        print(f"  Null hypothesis: {null_hypothesis}")
        print(f"  P-value (one-tail): {p_value_one_tail:.4f}")
        print(f"  Significant (α={alpha}): {self.bootstrap_results['significant_one_tail']}")
        
        return self.bootstrap_results
    
    def _generate_bootstrap_slopes_recent_trend(self, n_bootstrap: int) -> np.ndarray:
        """Generate bootstrap slopes under recent historical trend null hypothesis."""
        bootstrap_slopes = []
        n_points = len(self.test_data)
        years = np.arange(2020, 2020 + n_points)
        
        # Use recent historical trend as baseline
        baseline_trend = self.recent_historical_trend
        baseline_emissions = np.mean(self.test_data["emissions"])
        
        for _ in range(n_bootstrap):
            # Generate null hypothesis data (recent trend + noise)
            trend_emissions = baseline_emissions + baseline_trend * (years - years[0])
            null_emissions = trend_emissions + self.noise_generator(n_points)
            
            # Calculate slope
            X = years.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, null_emissions)
            bootstrap_slopes.append(model.coef_[0])
        
        return np.array(bootstrap_slopes)
    
    def _generate_bootstrap_slopes_zero_trend(self, n_bootstrap: int) -> np.ndarray:
        """Generate bootstrap slopes under zero trend null hypothesis."""
        bootstrap_slopes = []
        n_points = len(self.test_data)
        years = np.arange(2020, 2020 + n_points)
        baseline_emissions = np.mean(self.test_data["emissions"])
        
        for _ in range(n_bootstrap):
            null_emissions = np.full(n_points, baseline_emissions) + self.noise_generator(n_points)
            
            X = years.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, null_emissions)
            bootstrap_slopes.append(model.coef_[0])
        
        return np.array(bootstrap_slopes)
    
    def plot_enhanced_analysis(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """Create comprehensive visualization of enhanced analysis."""
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Historical data with optimal segments
        self._plot_optimal_segments(axes[0, 0])
        
        # 2. Autocorrelation analysis
        self._plot_autocorrelation(axes[0, 1])
        
        # 3. Noise distribution comparison
        self._plot_noise_comparison(axes[1, 0])
        
        # 4. Bootstrap results with both null hypotheses
        self._plot_enhanced_bootstrap_results(axes[1, 1])
        
        # 5. Segmentation comparison
        self._plot_segmentation_comparison(axes[2, 0])
        
        # 6. Summary
        self._plot_enhanced_summary(axes[2, 1])
        
        plt.tight_layout()
        # plt.show()
        return fig
    
    def _plot_optimal_segments(self, ax: plt.Axes) -> None:
        """Plot historical data with optimal segments highlighted."""
        ax.plot(self.historical_data["year"], self.historical_data["emissions"], 
                'b-', alpha=0.7, label="Historical emissions")
        
        # Overlay segments with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.optimal_segments['best']['segments'])))
        
        for i, (segment, color) in enumerate(zip(self.optimal_segments['best']['segments'], colors)):
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            
            ax.plot(segment['years'], trend, color=color, linewidth=2, 
                   label=f"Segment {i+1}" if i < 3 else "")
        
        if hasattr(self, 'test_data') and self.test_data is not None:
            ax.plot(self.test_data["year"], self.test_data["emissions"], 
                   'ro', markersize=8, label="Test data")
        
        ax.set_xlabel("Year")
        ax.set_ylabel("CO₂ Emissions (Mt)")
        ax.set_title(f"Optimal Segmentation ({self.optimal_segments['best']['method']})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_autocorrelation(self, ax: plt.Axes) -> None:
        """Plot autocorrelation function."""
        if self.autocorr_results is None:
            ax.text(0.5, 0.5, "No autocorrelation analysis", ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        lags = range(len(self.autocorr_results['acf_values']))
        ax.plot(lags, self.autocorr_results['acf_values'], 'bo-')
        
        # Add confidence intervals
        n = len(self.autocorr_results['temporal_residuals'])
        conf_interval = 1.96 / np.sqrt(n)
        ax.axhline(conf_interval, color='red', linestyle='--', alpha=0.7)
        ax.axhline(-conf_interval, color='red', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Autocorrelation Function of Residuals")
        ax.grid(True, alpha=0.3)
    
    def _plot_noise_comparison(self, ax: plt.Axes) -> None:
        """Plot noise distribution."""
        ax.hist(self.residuals, bins=30, density=True, alpha=0.7, 
               color="skyblue", label="Residuals")
        
        if self.noise_params["type"] == "normal":
            x_range = np.linspace(self.residuals.min(), self.residuals.max(), 100)
            fitted_density = stats.norm.pdf(x_range, self.noise_params["mu"], 
                                          self.noise_params["sigma"])
            ax.plot(x_range, fitted_density, "r-", linewidth=2, label="Fitted distribution")
        
        ax.set_xlabel("Residuals (Mt)")
        ax.set_ylabel("Density")
        ax.set_title(f'Enhanced Noise Distribution (σ = {self.noise_params["sigma"]:.1f} Mt)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_enhanced_bootstrap_results(self, ax: plt.Axes) -> None:
        """Plot bootstrap results."""
        if self.bootstrap_results is None:
            return
            
        ax.hist(self.bootstrap_results["bootstrap_slopes"], bins=50, density=True, 
               alpha=0.7, color="lightgreen", 
               label=f'Bootstrap slopes\n({self.bootstrap_results["null_hypothesis"]} null)')
        
        ax.axvline(self.bootstrap_results["test_slope"], color="red", linewidth=2,
                  label=f'Observed slope\n{self.bootstrap_results["test_slope"]:.1f} Mt/yr')
        
        if self.bootstrap_results["null_hypothesis"] == "recent_trend":
            ax.axvline(self.bootstrap_results["recent_historical_trend"], 
                      color="orange", linewidth=2, linestyle='--',
                      label=f'Recent historical trend\n{self.bootstrap_results["recent_historical_trend"]:.1f} Mt/yr')
        
        ax.set_xlabel("Slope (Mt/year)")
        ax.set_ylabel("Density")
        ax.set_title("Enhanced Bootstrap Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_segmentation_comparison(self, ax: plt.Axes) -> None:
        """Compare different segmentation methods."""
        if self.optimal_segments is None:
            return
            
        methods = []
        scores = []
        n_segments = []
        
        for seg in self.optimal_segments['all_tested']:
            methods.append(f"{seg['method']}\n(n={seg['n_segments']})")
            scores.append(seg['score'])
            n_segments.append(seg['n_segments'])
        
        bars = ax.bar(methods, scores)
        
        # Highlight the best method
        best_idx = np.argmin(scores)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(0.8)
        
        ax.set_ylabel("Score (lower is better)")
        ax.set_title("Segmentation Method Comparison")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_enhanced_summary(self, ax: plt.Axes) -> None:
        """Plot enhanced summary statistics."""
        ax.axis("off")
        
        if self.bootstrap_results is None:
            return
        
        summary_text = f"""
        ENHANCED STATISTICAL TEST RESULTS
        _________________________________
        
        Segmentation:
        • Method: {self.optimal_segments['best']['method']}
        • Segments: {self.optimal_segments['best']['n_segments']}
        
        Autocorrelation:
        • Lag-1 correlation: {self.autocorr_results['acf_values'][1]:.3f}
        • Significant lags: {len(self.autocorr_results['significant_lags'])}
        
        Test Results:
        • Observed slope: {self.bootstrap_results['test_slope']:.2f} Mt/year
        • Null hypothesis: {self.bootstrap_results['null_hypothesis']}
        • P-value: {self.bootstrap_results['p_value_one_tail']:.4f}
        • Significant: {self.bootstrap_results['significant_one_tail']}
        
        Noise Model:
        • Distribution: {self.noise_params['type']}
        • Std deviation: {self.noise_params['sigma']:.1f} Mt
        
        CONCLUSION:
        """
        
        # Add conclusion based on results
        if self.bootstrap_results['significant_one_tail'] and self.bootstrap_results['test_slope'] < 0:
            if self.bootstrap_results['null_hypothesis'] == 'recent_trend':
                conclusion = "Strong evidence of acceleration in emissions decline"
            else:
                conclusion = "Strong evidence that emissions have peaked"
        elif self.bootstrap_results['test_slope'] < 0:
            conclusion = "Declining trend but not statistically significant"
        else:
            conclusion = "No evidence of emissions peak"
        
        summary_text += conclusion



# =================================
# Define enhanced classes to factor autocorrelation into bootstrapping
# =================================

class AutocorrAwareBootstrap:
    """
    Demonstrates different methods for incorporating autocorrelation into bootstrap sampling.
    """
    
    def __init__(self):
        self.residuals = None
        self.autocorr_params = None
        self.noise_generator = None
    
    def fit_autocorrelation_model(self, residuals: np.ndarray, method: str = "ar1") -> Dict:
        """
        Fit an autocorrelation model to residuals.
        
        Args:
            residuals: Time-ordered residuals
            method: "ar1", "ar2", or "arma"
        
        Returns:
            Dictionary with model parameters
        """
        self.residuals = residuals
        
        if method == "ar1":
            return self._fit_ar1_model(residuals)
        elif method == "ar2":
            return self._fit_ar2_model(residuals)
        elif method == "arma":
            return self._fit_arma_model(residuals)
        else:
            raise ValueError("Method must be 'ar1', 'ar2', or 'arma'")
    
    def _fit_ar1_model(self, residuals: np.ndarray) -> Dict:
        """
        Fit AR(1) model: r_t = φ * r_{t-1} + ε_t
        """
        # Simple AR(1) estimation using OLS
        y = residuals[1:]  # r_t
        X = residuals[:-1].reshape(-1, 1)  # r_{t-1}
        
        model = LinearRegression()
        model.fit(X, y)
        
        phi = model.coef_[0]
        innovation_residuals = y - model.predict(X)
        sigma_innovation = np.std(innovation_residuals)
        
        # For AR(1): Var(r_t) = σ²_ε / (1 - φ²)
        unconditional_variance = sigma_innovation**2 / (1 - phi**2) if abs(phi) < 1 else sigma_innovation**2
        
        self.autocorr_params = {
            "model": "AR(1)",
            "phi": phi,
            "sigma_innovation": sigma_innovation,
            "unconditional_variance": unconditional_variance,
            "innovation_residuals": innovation_residuals,
            "is_stationary": abs(phi) < 1
        }
        
        print(f"AR(1) model fitted:")
        print(f"  φ = {phi:.3f}")
        print(f"  σ_ε = {sigma_innovation:.1f}")
        print(f"  Stationary: {abs(phi) < 1}")
        
        return self.autocorr_params
    
    def _fit_ar2_model(self, residuals: np.ndarray) -> Dict:
        """
        Fit AR(2) model: r_t = φ₁ * r_{t-1} + φ₂ * r_{t-2} + ε_t
        """
        y = residuals[2:]
        X = np.column_stack([residuals[1:-1], residuals[:-2]])
        
        model = LinearRegression()
        model.fit(X, y)
        
        phi1, phi2 = model.coef_
        innovation_residuals = y - model.predict(X)
        sigma_innovation = np.std(innovation_residuals)
        
        # Check stationarity conditions for AR(2)
        is_stationary = (phi1 + phi2 < 1) and (phi2 - phi1 < 1) and (abs(phi2) < 1)
        
        self.autocorr_params = {
            "model": "AR(2)",
            "phi1": phi1,
            "phi2": phi2,
            "sigma_innovation": sigma_innovation,
            "innovation_residuals": innovation_residuals,
            "is_stationary": is_stationary
        }
        
        return self.autocorr_params
    
    def _fit_arma_model(self, residuals: np.ndarray) -> Dict:
        """
        Fit ARMA model using statsmodels (more robust).
        """
        try:
            # Try ARMA(1,1) first
            model = ARIMA(residuals, order=(1, 0, 1))
            fitted_model = model.fit(disp=False)
            
            self.autocorr_params = {
                "model": "ARMA(1,1)",
                "fitted_model": fitted_model,
                "aic": fitted_model.aic,
                "residuals": fitted_model.resid,
                "sigma_innovation": np.sqrt(fitted_model.params['sigma2'])
            }
            
            # If ARMA fails, fall back to AR(1)
        except:
            print("ARMA fitting failed, falling back to AR(1)")
            return self._fit_ar1_model(residuals)
        
        return self.autocorr_params
    
    def create_autocorr_noise_generator(self) -> Callable:
        """
        Create a noise generator that preserves autocorrelation structure.
        """
        if self.autocorr_params is None:
            raise ValueError("Must fit autocorrelation model first")
        
        if self.autocorr_params["model"] == "AR(1)":
            return self._create_ar1_generator()
        elif self.autocorr_params["model"] == "AR(2)":
            return self._create_ar2_generator()
        elif "ARMA" in self.autocorr_params["model"]:
            return self._create_arma_generator()
        else:
            raise ValueError("Unknown autocorrelation model")
    
    def _create_ar1_generator(self) -> Callable:
        """
        Create AR(1) noise generator.
        """
        phi = self.autocorr_params["phi"]
        sigma_eps = self.autocorr_params["sigma_innovation"]
        
        def ar1_generator(size: int, initial_value: float = 0) -> np.ndarray:
            """
            Generate AR(1) time series: r_t = φ * r_{t-1} + ε_t
            """
            innovations = np.random.normal(0, sigma_eps, size)
            series = np.zeros(size)
            series[0] = initial_value
            
            for t in range(1, size):
                series[t] = phi * series[t-1] + innovations[t]
            
            return series
        
        self.noise_generator = ar1_generator
        return ar1_generator
    
    def _create_ar2_generator(self) -> Callable:
        """
        Create AR(2) noise generator.
        """
        phi1 = self.autocorr_params["phi1"]
        phi2 = self.autocorr_params["phi2"]
        sigma_eps = self.autocorr_params["sigma_innovation"]
        
        def ar2_generator(size: int, initial_values: Tuple[float, float] = (0, 0)) -> np.ndarray:
            """
            Generate AR(2) time series: r_t = φ₁ * r_{t-1} + φ₂ * r_{t-2} + ε_t
            """
            if size < 2:
                return np.random.normal(0, sigma_eps, size)
            
            innovations = np.random.normal(0, sigma_eps, size)
            series = np.zeros(size)
            series[0], series[1] = initial_values
            
            for t in range(2, size):
                series[t] = phi1 * series[t-1] + phi2 * series[t-2] + innovations[t]
            
            return series
        
        self.noise_generator = ar2_generator
        return ar2_generator
    
    def _create_arma_generator(self) -> Callable:
        """
        Create ARMA noise generator using fitted model.
        """
        fitted_model = self.autocorr_params["fitted_model"]
        
        def arma_generator(size: int) -> np.ndarray:
            """
            Generate ARMA time series using fitted model.
            """
            # Use the fitted model to simulate new series
            simulated = fitted_model.simulate(nsimulations=size)
            return simulated
        
        self.noise_generator = arma_generator
        return arma_generator


class AutocorrBootstrapMethods:
    """
    Different bootstrap methods for handling autocorrelated data.
    """
    
    def __init__(self, residuals: np.ndarray):
        self.residuals = residuals
        self.n = len(residuals)
    
    def block_bootstrap(self, block_size: int, n_samples: int) -> List[np.ndarray]:
        """
        Block bootstrap: Sample blocks of consecutive residuals.
        Preserves short-term autocorrelation structure.
        """
        bootstrap_samples = []
        n_blocks = self.n - block_size + 1
        
        for _ in range(n_samples):
            # Randomly select starting positions for blocks
            n_blocks_needed = int(np.ceil(self.n / block_size))
            block_starts = np.random.choice(n_blocks, size=n_blocks_needed, replace=True)
            
            sample = []
            for start in block_starts:
                block = self.residuals[start:start + block_size]
                sample.extend(block)
            
            # Trim to exact length needed
            sample = np.array(sample[:self.n])
            bootstrap_samples.append(sample)
        
        return bootstrap_samples
    
    def circular_block_bootstrap(self, block_size: int, n_samples: int) -> List[np.ndarray]:
        """
        Circular block bootstrap: Treats the series as circular to handle edge effects.
        """
        bootstrap_samples = []
        
        # Create circular version of residuals
        circular_residuals = np.concatenate([self.residuals, self.residuals])
        
        for _ in range(n_samples):
            n_blocks_needed = int(np.ceil(self.n / block_size))
            block_starts = np.random.choice(self.n, size=n_blocks_needed, replace=True)
            
            sample = []
            for start in block_starts:
                block = circular_residuals[start:start + block_size]
                sample.extend(block)
            
            sample = np.array(sample[:self.n])
            bootstrap_samples.append(sample)
        
        return bootstrap_samples
    
    def stationary_bootstrap(self, avg_block_size: float, n_samples: int) -> List[np.ndarray]:
        """
        Stationary bootstrap: Variable block sizes with geometric distribution.
        Better preserves long-run properties.
        """
        bootstrap_samples = []
        p = 1.0 / avg_block_size  # Probability of ending a block
        
        for _ in range(n_samples):
            sample = []
            while len(sample) < self.n:
                # Random starting point
                start = np.random.randint(0, self.n)
                
                # Generate block length from geometric distribution
                block_length = np.random.geometric(p)
                
                # Extract block (with wraparound)
                for i in range(block_length):
                    if len(sample) >= self.n:
                        break
                    idx = (start + i) % self.n
                    sample.append(self.residuals[idx])
            
            sample = np.array(sample[:self.n])
            bootstrap_samples.append(sample)
        
        return bootstrap_samples
    
    def ar_bootstrap(self, autocorr_model: AutocorrAwareBootstrap, n_samples: int) -> List[np.ndarray]:
        """
        AR bootstrap: Generate new series using fitted AR model.
        """
        if autocorr_model.noise_generator is None:
            raise ValueError("Must create noise generator first")
        
        bootstrap_samples = []
        
        for _ in range(n_samples):
            # Generate new autocorrelated series
            if autocorr_model.autocorr_params["model"] == "AR(1)":
                sample = autocorr_model.noise_generator(self.n, initial_value=0)
            elif autocorr_model.autocorr_params["model"] == "AR(2)":
                sample = autocorr_model.noise_generator(self.n, initial_values=(0, 0))
            else:  # ARMA
                sample = autocorr_model.noise_generator(self.n)
            
            bootstrap_samples.append(sample)
        
        return bootstrap_samples


# Enhanced Peak Test with proper autocorrelation bootstrap
class AutocorrEnhancedPeakTest:
    """
    Emissions peak test with proper autocorrelation-aware bootstrapping.
    """
    
    def __init__(self):
        self.historical_data = None
        self.test_data = None
        self.residuals = None
        self.autocorr_model = None
        self.bootstrap_method = "ar_bootstrap"
    
    def set_data(self, historical_data: pd.DataFrame, test_data: List[Tuple[int, float]]):
        """Set historical and test data."""
        self.historical_data = historical_data
        self.test_data = pd.DataFrame(test_data, columns=["year", "emissions"])
        
        # Calculate residuals (simplified for demo)
        X = self.historical_data["year"].values.reshape(-1, 1)
        y = self.historical_data["emissions"].values
        model = LinearRegression()
        model.fit(X, y)
        self.residuals = y - model.predict(X)
        
        return self
    
    def analyze_autocorrelation(self, method: str = "ar1"):
        """Analyze and model autocorrelation."""
        self.autocorr_model = AutocorrAwareBootstrap()
        self.autocorr_model.fit_autocorrelation_model(self.residuals, method=method)
        self.autocorr_model.create_autocorr_noise_generator()
        
        return self
    
    def run_autocorr_bootstrap_test(self, n_bootstrap: int = 1000, 
                                   bootstrap_method: str = "ar_bootstrap") -> Dict:
        """
        Run bootstrap test with proper autocorrelation handling.
        """
        self.bootstrap_method = bootstrap_method
        
        # Calculate observed test slope
        X_test = self.test_data["year"].values.reshape(-1, 1)
        y_test = self.test_data["emissions"].values
        model_test = LinearRegression()
        model_test.fit(X_test, y_test)
        observed_slope = model_test.coef_[0]
        
        # Generate bootstrap samples
        bootstrap_methods = AutocorrBootstrapMethods(self.residuals)
        
        if bootstrap_method == "ar_bootstrap":
            bootstrap_residuals = bootstrap_methods.ar_bootstrap(
                self.autocorr_model, n_bootstrap)
        elif bootstrap_method == "block_bootstrap":
            block_size = max(3, int(len(self.residuals) * 0.1))  # 10% of data or minimum 3
            bootstrap_residuals = bootstrap_methods.block_bootstrap(
                block_size, n_bootstrap)
        elif bootstrap_method == "stationary_bootstrap":
            avg_block_size = max(5, len(self.residuals) * 0.15)
            bootstrap_residuals = bootstrap_methods.stationary_bootstrap(
                avg_block_size, n_bootstrap)
        else:
            raise ValueError("Unknown bootstrap method")
        
        # Calculate bootstrap slopes
        bootstrap_slopes = []
        n_test_points = len(self.test_data)
        baseline_emissions = np.mean(self.test_data["emissions"])
        
        for residuals_sample in bootstrap_residuals:
            # Create null hypothesis emissions with autocorrelated noise
            years = np.arange(2020, 2020 + n_test_points)
            
            # Use only first n_test_points of residuals sample
            noise_sample = residuals_sample[:n_test_points]
            null_emissions = baseline_emissions + noise_sample
            
            # Calculate slope
            X = years.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, null_emissions)
            bootstrap_slopes.append(model.coef_[0])
        
        bootstrap_slopes = np.array(bootstrap_slopes)
        
        # Calculate p-value
        p_value = np.sum(bootstrap_slopes <= observed_slope) / len(bootstrap_slopes)
        
        results = {
            "observed_slope": observed_slope,
            "bootstrap_slopes": bootstrap_slopes,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "bootstrap_method": bootstrap_method,
            "autocorr_model": self.autocorr_model.autocorr_params["model"],
            "phi": self.autocorr_model.autocorr_params.get("phi", "N/A")
        }
        
        print(f"Autocorrelation-aware bootstrap test:")
        print(f"  Method: {bootstrap_method}")
        print(f"  Autocorr model: {results['autocorr_model']}")
        print(f"  Observed slope: {observed_slope:.2f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {results['significant']}")
        
        return results
    
    def compare_bootstrap_methods(self, n_bootstrap: int = 500) -> Dict:
        """Compare different bootstrap methods."""
        methods = ["ar_bootstrap", "block_bootstrap", "stationary_bootstrap"]
        results = {}
        
        print("Comparing bootstrap methods...")
        
        for method in methods:
            try:
                result = self.run_autocorr_bootstrap_test(n_bootstrap, method)
                results[method] = result
                print(f"{method}: p-value = {result['p_value']:.4f}")
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Generate sample data with autocorrelation
    np.random.seed(42)
    
    years = np.arange(1970, 2020)
    base_trend = -50 * (years - 1970) + 35000
    
    # Generate autocorrelated noise
    autocorr_gen = AutocorrAwareBootstrap()
    autocorr_gen.autocorr_params = {
        "model": "AR(1)", "phi": 0.6, "sigma_innovation": 800,
        "unconditional_variance": 800**2 / (1 - 0.6**2), "is_stationary": True
    }
    autocorr_gen.create_autocorr_noise_generator()
    
    correlated_noise = autocorr_gen.noise_generator(len(years), initial_value=0)
    emissions = base_trend + correlated_noise
    
    historical_data = pd.DataFrame({
        'year': years,
        'emissions': emissions
    })
    
    test_data = [(2021, 36000), (2022, 35700), (2023, 35400), (2024, 35100)]
    
    # Run analysis
    peak_test = AutocorrEnhancedPeakTest()
    peak_test.set_data(historical_data, test_data).analyze_autocorrelation("ar1")
    
    # Compare methods
    comparison = peak_test.compare_bootstrap_methods(n_bootstrap=1000)