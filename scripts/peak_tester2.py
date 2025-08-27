"""
Overview
This class implements a statistical method for testing whether CO2 emissions have peaked or are declining faster than historical trends. 

Key Features (Claude's description)
1. Intelligent Data Segmentation

Dynamic segmentation: Automatically finds optimal breakpoints based on R² maximization
Equal segmentation: Tests fixed-length segments of varying durations
Structural break detection: Captures policy changes, economic crises, and technological shifts
Anomaly handling: Built-in methods to exclude, interpolate, or flag anomalous years (e.g., COVID, financial crises, Soviet collapse)

2. Autocorrelation Analysis

AR(1) modeling: Detects and accounts for temporal correlation in emissions residuals
Robust noise generation: Uses appropriate noise models (AR(1) or white noise) for bootstrap testing
Conservative testing: Prevents false positives from ignoring temporal dependencies

3. Different options for Null Hypotheses

Recent trend baseline: Tests against recent historical trend rather than zero trend
Acceleration detection: Identifies whether decline is faster than expected, not just present

4. Bootstrap Testing

Multiple bootstrap methods: AR(1), block bootstrap, or white noise
Default 10,000 bootstrap samples for robust p-values
Effect size calculation: Quantifies practical significance beyond statistical significance

Usage
# 1. Load and clean data with anomaly handling
complete_test = CompleteEnhancedEmissionsPeakTest()
complete_test.load_historical_data(
    'emissions_data.csv',
    emissions_col='co2_emissions',
    year_range=range(1970, 2024),
    anomaly_years=[1992, 1993, 2008, 2009, 2020, 2021],  # Soviet collapse, GFC, COVID
    anomaly_method='interpolate'  # or 'exclude' or 'flag'
)

# 2. Optimize segmentation to capture structural breaks
complete_test.optimize_segments(min_segment=5, max_segment=20)

# 3. Analyze temporal correlation patterns
complete_test.analyze_autocorrelation()

# 4. Create appropriate noise generator
complete_test.create_noise_generator()

# 5. Set test period and calculate recent historical baseline
complete_test.set_test_data(test_data, recent_years_for_trend=10)

# 6. Run comprehensive bootstrap test
results = complete_test.run_complete_bootstrap_test(
    n_bootstrap=10000,
    null_hypothesis="recent_trend",  # Test for acceleration
    bootstrap_method="ar_bootstrap"   # Account for autocorrelation
)

# 7. Interpret and visualize results
interpretation = complete_test.interpret_results()
complete_test.plot_complete_analysis()


Author: Neil Grant and Claire Fyson
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf
from typing import List, Tuple, Dict, Optional, Callable, Union
import warnings

warnings.filterwarnings("ignore")

class CompleteEnhancedEmissionsPeakTest:
    """
    Complete enhanced emissions peak test integrating:
    1. Dynamic segment length optimization
    2. Autocorrelation-aware bootstrapping
    3. Recent historical trend null hypothesis
    """
    
    def __init__(self):
        self.historical_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.optimal_segments: Optional[Dict] = None
        self.residuals_temporal: Optional[np.ndarray] = None
        self.autocorr_params: Optional[Dict] = None
        self.noise_generator: Optional[Callable] = None
        self.recent_historical_trend: Optional[float] = None
        self.bootstrap_results: Optional[Dict] = None

    def load_historical_data(self, data_source, 
                           year_col: str = "year", 
                           emissions_col: str = "emissions",
                           year_range: Optional[range] = None,
                           default_anomaly_years: Optional[range] = None):
        """
        Load and validate historical emissions data with anomaly handling.
        
        Parameters:
        - data_source: CSV filename or DataFrame
        - year_col: Name of year column
        - emissions_col: Name of emissions column  
        - year_range: Range of years to include
        - anomaly_years: List of years to treat as anomalies (default: [2008, 2009, 2020, 2021])
        """
        
        # Default anomaly years if none specified
        if default_anomaly_years is None:
            default_anomaly_years = [1992, 1993, 2008, 2009, 2020, 2021]  # Soviet union fall, GFC and COVID

        # Store raw data (existing loading logic)
        # ... existing loading code ...
        if isinstance(data_source, str):
            # Load from file, which should be stored in the data folder
            data_path = Path(__file__).resolve().parent / f"../data/{data_source}"

            try:
                raw_data = pd.read_csv(data_path)
            except Exception as e:
                raise ValueError(f"Could not load data from {data_path}: {e}")

        elif isinstance(data_source, pd.DataFrame):
            raw_data = data_source.copy()
        
        # Check columns (existing code)
        if emissions_col not in raw_data.columns:
            print(f"Available columns: {list(raw_data.columns)}")
            raise ValueError(f"Emissions column '{emissions_col}' not found in data")
        
        if year_col not in raw_data.columns:
            print(f"Available columns: {list(raw_data.columns)}")
            raise ValueError(f"Year column '{year_col}' not found in data")
        
        # Select and rename columns (existing code)
        self.historical_data = raw_data[[year_col, emissions_col]].rename(
            columns={year_col: "year", emissions_col: "emissions"}
        ).copy()
        
        # Filter by year range (existing code)
        if year_range is not None:
            min_year, max_year = min(year_range), max(year_range)
            print(f"Filtering data to year range: {min_year}-{max_year}")
            self.historical_data = self.historical_data[
                (self.historical_data["year"] >= min_year) & 
                (self.historical_data["year"] <= max_year)
            ]
        
        # Clean data (existing code)
        self.historical_data_raw = self.historical_data.dropna().sort_values("year").reset_index(drop=True).copy()

        self.default_anomaly_years = default_anomaly_years
        
        # Create pre-processed versions for common use cases
        self.historical_data_excluded = self._apply_anomaly_handling(
            self.historical_data_raw, default_anomaly_years, 'exclude'
        )
        self.historical_data_interpolated = self._apply_anomaly_handling(
            self.historical_data_raw, default_anomaly_years, 'interpolate'  
        )
        
        # Default to raw data
        self.historical_data = self.historical_data_raw
        
        print(f"Data loaded: {len(self.historical_data_raw)} raw, "
              f"{len(self.historical_data_excluded)} excluded, "
              f"{len(self.historical_data_interpolated)} interpolated")
        
            # # NEW: Handle anomalous years
            # if anomaly_method != 'none' and anomaly_years:
            #     print(f"Handling anomalous years: {anomaly_years} using method '{anomaly_method}'")
            #     original_count = len(self.historical_data)
                
            #     if anomaly_method == 'exclude':
            #         # Remove anomalous years
            #         self.historical_data = self.historical_data[
            #             ~self.historical_data['year'].isin(anomaly_years)
            #         ].reset_index(drop=True)
            #         excluded_count = original_count - len(self.historical_data)
            #         print(f"Excluded {excluded_count} anomalous years")
                    
            #     elif anomaly_method == 'interpolate':
            #         # Replace anomalous years with interpolated values
            #         for year in anomaly_years:
            #             year_mask = self.historical_data['year'] == year
            #             if year_mask.any():
            #                 idx = self.historical_data[year_mask].index[0]
                            
            #                 # Get surrounding years for interpolation
            #                 before_idx = idx - 1 if idx > 0 else None
            #                 after_idx = idx + 1 if idx < len(self.historical_data) - 1 else None
                            
            #                 if before_idx is not None and after_idx is not None:
            #                     before_val = self.historical_data.iloc[before_idx]['emissions']
            #                     after_val = self.historical_data.iloc[after_idx]['emissions']
            #                     interpolated_val = (before_val + after_val) / 2
            #                     original_val = self.historical_data.iloc[idx]['emissions']
                                
            #                     print(f"  {year}: {original_val:.0f} → {interpolated_val:.0f}")
            #                     self.historical_data.iloc[idx, self.historical_data.columns.get_loc('emissions')] = interpolated_val
                                
            #     elif anomaly_method == 'flag':
            #         # Add a flag column (for potential future weighted analysis)
            #         self.historical_data['is_anomaly'] = self.historical_data['year'].isin(anomaly_years)
            #         flagged_count = self.historical_data['is_anomaly'].sum()
            #         print(f"Flagged {flagged_count} anomalous years")
            
            # Print final data info (existing code)
        print(f"Processed data: {self.historical_data['year'].min()}-{self.historical_data['year'].max()}")
        print(f"Data points: {len(self.historical_data)}")
        print(f"Emissions range: {self.historical_data['emissions'].min():.0f} - {self.historical_data['emissions'].max():.0f}")
            
        return self

    def _apply_anomaly_handling(self, data, anomaly_years, method):
        """Internal method to apply anomaly handling."""
        if method == 'none':
            return data
        
        working_data = data.copy()
        
        if method == 'exclude':
            working_data = working_data[
                ~working_data['year'].isin(anomaly_years)
            ].reset_index(drop=True)
            
        elif method == 'interpolate':
            for year in anomaly_years:
                year_mask = working_data['year'] == year
                if year_mask.any():
                    idx = working_data[year_mask].index[0]
                    before_idx = idx - 1 if idx > 0 else None
                    after_idx = idx + 1 if idx < len(working_data) - 1 else None
                    
                    if before_idx is not None and after_idx is not None:
                        before_val = working_data.iloc[before_idx]['emissions']
                        after_val = working_data.iloc[after_idx]['emissions']
                        interpolated_val = (before_val + after_val) / 2
                        working_data.iloc[idx, working_data.columns.get_loc('emissions')] = interpolated_val
        
        return working_data
        

    def optimize_segments(self, min_segment: int = 5, max_segment: int = 20, 
                    force_method: Optional[str] = None, 
                    event_years: Optional[List[int]] = None,
                    data_version = 'excluded'
                         ) -> Dict:
        """
        Find optimal variable segment lengths.
        
        Parameters:
        - min_segment: Minimum years per segment
        - max_segment: Maximum years per segment (for equal segments)
        - force_method: Force specific method ('equal', 'dynamic', 'event_based', 'hybrid', or None for optimal)
        - event_years: Years for forced breaks (required for 'event_based' and 'hybrid' methods)
        - data_version: Choose whether to exclude anomalous data, and whether to interpolate if so
        """
            # Select data version
        if data_version == 'raw':
            working_data = self.historical_data_raw
        elif data_version == 'excluded':
            working_data = self.historical_data_excluded  
        elif data_version == 'interpolated':
            working_data = self.historical_data_interpolated
        else:
            raise ValueError("data_version must be 'raw', 'excluded', or 'interpolated'")
        
        print(f"Optimizing segments using {data_version} data ({len(working_data)} points)")

        years = working_data["year"].values
        emissions = working_data["emissions"].values
        
        best_segmentation = None
        best_score = float('inf')
        segmentations = []
        
        # Always test equal and dynamic methods (unless force_method is event_based/hybrid only)
        if force_method not in ['event_based', 'hybrid']:
            
            # Method 1: Equal segments of varying lengths
            for seg_len in range(min_segment, min(max_segment + 1, len(years) // 2)):
                segments = self._create_equal_segments(years, emissions, seg_len)
                if len(segments) >= 2:  # Need at least 2 segments
                    score = self._evaluate_segmentation(segments)
                    segmentations.append({
                        'method': 'equal',
                        'segment_length': seg_len,
                        'segments': segments,
                        'score': score
                    })
                    if score < best_score:
                        best_score = score
                        best_segmentation = segmentations[-1]
            
            # Method 2: Dynamic segmentation
            dynamic_segments = self._create_dynamic_segments(years, emissions, min_segment)
            if len(dynamic_segments) >= 2:
                score = self._evaluate_segmentation(dynamic_segments)
                segmentations.append({
                    'method': 'dynamic',
                    'segments': dynamic_segments,
                    'score': score
                })
                if score < best_score:
                    best_score = score
                    best_segmentation = segmentations[-1]
        
        # Method 3: Event-based segmentation (if requested)
        if force_method in ['event_based', 'hybrid'] or (event_years is not None):
            if event_years is None:
                raise ValueError("event_years must be provided for event-based or hybrid methods")
            
            if force_method == 'event_based' or (force_method is None and event_years is not None):
                event_segments = self._create_event_segments(years, emissions, event_years, min_segment)
                if len(event_segments) >= 2:
                    score = self._evaluate_segmentation(event_segments)
                    segmentations.append({
                        'method': 'event_based',
                        'event_years': event_years,
                        'segments': event_segments,
                        'score': score
                    })
                    if score < best_score:
                        best_score = score
                        best_segmentation = segmentations[-1]
            
            # Method 4: Hybrid segmentation
            if force_method == 'hybrid':
                hybrid_segments = self._create_hybrid_segments(years, emissions, event_years, min_segment)
                if len(hybrid_segments) >= 2:
                    score = self._evaluate_segmentation(hybrid_segments)
                    segmentations.append({
                        'method': 'hybrid',
                        'event_years': event_years,
                        'segments': hybrid_segments,
                        'score': score
                    })
                    if score < best_score:
                        best_score = score
                        best_segmentation = segmentations[-1]
        
        # Apply forced method selection
        if force_method is not None:
            forced_segmentation = None
            for seg in segmentations:
                if seg['method'] == force_method:
                    forced_segmentation = seg
                    break
            
            if forced_segmentation is not None:
                best_segmentation = forced_segmentation
                print(f"Forced method '{force_method}': {len(forced_segmentation['segments'])} segments, "
                      f"score = {forced_segmentation['score']:.1f}")
            else:
                raise ValueError(f"Forced method '{force_method}' not found in tested methods")
        
        self.optimal_segments = {
            'best': best_segmentation,
            'all_tested': segmentations
        }
        
        if force_method is None:
            print(f"Optimal: {best_segmentation['method']} method, {len(best_segmentation['segments'])} segments")
        
        # Store which data version was used
        self.segmentation_data_version = data_version        
        
        return self.optimal_segments
    
    def _create_event_segments(self, years: np.ndarray, emissions: np.ndarray, 
                              event_years: List[int], min_segment: int) -> List[Dict]:
        """Create segments with forced breaks at event years."""
        # Filter event years to data range and sort
        event_years = sorted([year for year in event_years 
                             if years.min() < year < years.max()])
        
        # Create breakpoints
        breakpoints = [years.min()] + event_years + [years.max()]
        segments = []
        
        for i in range(len(breakpoints) - 1):
            start_year = breakpoints[i] 
            end_year = breakpoints[i + 1]
            
            # Get indices
            start_idx = np.where(years >= start_year)[0][0]
            end_idx = np.where(years <= end_year)[0][-1] + 1
            
            # Only create if meets minimum length
            if end_idx - start_idx >= min_segment:
                segment_years = years[start_idx:end_idx]
                segment_emissions = emissions[start_idx:end_idx]
                
                segments.append({
                    'years': segment_years,
                    'emissions': segment_emissions,
                    'start_year': segment_years[0],
                    'end_year': segment_years[-1]
                })
        
        return segments
    
    def _create_hybrid_segments(self, years: np.ndarray, emissions: np.ndarray,
                               event_years: List[int], min_segment: int) -> List[Dict]:
        """Create dynamic segments within event-defined periods."""
        # Filter and sort event years
        event_years = sorted([year for year in event_years 
                             if years.min() < year < years.max()])
        
        # Create periods between events
        period_starts = [years.min()] + event_years
        period_ends = event_years + [years.max()]
        
        all_segments = []
        
        for period_start, period_end in zip(period_starts, period_ends):
            # Get data for this period
            period_mask = (years >= period_start) & (years <= period_end)
            period_years = years[period_mask]
            period_emissions = emissions[period_mask]
            
            if len(period_years) >= min_segment:
                # Apply dynamic segmentation within period
                period_segments = self._create_dynamic_segments(
                    period_years, period_emissions, min_segment
                )
                all_segments.extend(period_segments)
            else:
                # Single segment if too short
                all_segments.append({
                    'years': period_years,
                    'emissions': period_emissions,
                    'start_year': period_years[0],
                    'end_year': period_years[-1]
                })
        
        return all_segments
    
    def _create_equal_segments(self, years: np.ndarray, emissions: np.ndarray, 
                              segment_length: int) -> List[Dict]:
        """Create equal-length segments."""
        segments = []
        for i in range(0, len(years), segment_length):
            end_idx = min(i + segment_length, len(years))
            if end_idx - i >= 3:
                segments.append({
                    'years': years[i:end_idx],
                    'emissions': emissions[i:end_idx],
                    'start_year': years[i],
                    'end_year': years[end_idx-1]
                })
        return segments
    
    
    def _create_dynamic_segments(self, years: np.ndarray, emissions: np.ndarray,
                                min_segment: int) -> List[Dict]:
        """Create variable-length segments based on R² optimization."""
        segments = []
        start_idx = 0
        n_years = len(years)  # Fix: Define n_years locally
        
        while start_idx < n_years:
            best_end_idx = start_idx + min_segment
            best_r2 = -1
            
            # Find segment length that maximizes R²
            max_len = min(start_idx + 25, n_years)  # Cap segment length
            for end_idx in range(start_idx + min_segment, max_len + 1):
                X = years[start_idx:end_idx].reshape(-1, 1)
                y = emissions[start_idx:end_idx]
                
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_end_idx = end_idx

            # Clamp best_end_idx to n_years
            best_end_idx = min(best_end_idx, n_years)
            
            segments.append({
                'years': years[start_idx:best_end_idx],
                'emissions': emissions[start_idx:best_end_idx],
                'start_year': years[start_idx],
                'end_year': years[best_end_idx-1],
                'r2': best_r2
            })
            
            start_idx = best_end_idx
        
        return segments
    
    def _evaluate_segmentation(self, segments: List[Dict]) -> float:
        """Evaluate segmentation quality based on pooled residual variance."""
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
        
        # Score: lower variance is better, but penalize very low R²
        score = pooled_variance * (2.5 - avg_r2)
        return score

    
    def analyze_autocorrelation(self, data_version = 'interpolated', use_segmentation_data = False) -> Dict:
        """
        Analyze autocorrelation in temporally-ordered residuals.
        Parameters:
        - data_version: 'raw', 'excluded', 'interpolated' 
        - use_segmentation_data: If True, use same data as segmentation
        """

        if self.optimal_segments is None:
            raise ValueError("No segmentation run yet. Run optimize_segments() first.")
        
        if use_segmentation_data:
            if hasattr(self, 'segmentation_data_version'):
                data_version = self.segmentation_data_version
            else:
                # Fall back to default if segmentation_data_version not set
                data_version = 'excluded'  # or whatever default you prefer
    
        # Select appropriate data  
        if data_version == 'raw':
            working_data = self.historical_data_raw
        elif data_version == 'excluded':
            working_data = self.historical_data_excluded
        elif data_version == 'interpolated': 
            working_data = self.historical_data_interpolated
        else:
            raise ValueError("data_version must be 'raw', 'excluded', or 'interpolated'")
        
        print(f"Analyzing autocorrelation using {data_version} data ({len(working_data)} points)")

        # Extract residuals from segments but use working_data for temporal ordering
        segments = self.optimal_segments['best']['segments']
        residuals_with_years = []

        # Get residuals from segmentation (which used segmentation data)
        for segment in segments:
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            
            for i, year in enumerate(segment['years']):
                residuals_with_years.append((year, residuals[i]))
        
        # Sort by year and filter based on working_data
        residuals_with_years.sort()
        working_years = set(working_data['year'].values)

        # Create residuals_temporal but only for years in working_data
        filtered_residuals = []
        for year, residual in residuals_with_years:
            if year in working_years:
                filtered_residuals.append((year, residual))

        # Now create residuals_temporal array
        self.residuals_temporal = np.array([r[1] for r in filtered_residuals])
    
        # Continue with existing autocorrelation logic
        acf_values = acf(self.residuals_temporal, nlags=5, fft=True)
        phi = acf_values[1] if len(acf_values) > 1 else 0
        
        # Simple AR(1) model fitting
        if len(self.residuals_temporal) > 1:
            y = self.residuals_temporal[1:]
            X = self.residuals_temporal[:-1].reshape(-1, 1)
            
            ar_model = LinearRegression()
            ar_model.fit(X, y)
            phi_fitted = ar_model.coef_[0]
            innovation_residuals = y - ar_model.predict(X)
            sigma_innovation = np.std(innovation_residuals)
        else:
            phi_fitted = 0
            sigma_innovation = np.std(self.residuals_temporal)
        
        self.autocorr_params = {
            'phi': phi_fitted,
            'sigma_innovation': sigma_innovation,
            'acf_lag1': phi,
            'has_autocorr': abs(phi) > 0.1,
            'is_stationary': abs(phi_fitted) < 1
        }
        
        print(f"Autocorrelation analysis:")
        print(f"  Lag-1 autocorr: {phi:.3f}")
        print(f"  AR(1) φ: {phi_fitted:.3f}")
        print(f"  Innovation σ: {sigma_innovation:.1f}")
        print(f"  Has significant autocorr: {self.autocorr_params['has_autocorr']}")
        
        return self.autocorr_params
    
    def create_noise_generator(self) -> Callable:
        """Create autocorrelation-aware noise generator."""
        if self.autocorr_params is None:
            self.analyze_autocorrelation()
        
        phi = self.autocorr_params['phi']
        sigma = self.autocorr_params['sigma_innovation']
        
        if self.autocorr_params['has_autocorr'] and self.autocorr_params['is_stationary']:
            print(f"Using AR(1) noise generator with φ={phi:.3f}")
            
            def ar1_noise_generator(size: int, initial_value: float = 0) -> np.ndarray:
                """Generate AR(1) autocorrelated noise."""
                innovations = np.random.normal(0, sigma, size)
                series = np.zeros(size)
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
    
    def set_test_data(self, test_data: List[Tuple[int, float]], 
                      recent_years_for_trend: int = 10) -> "CompleteEnhancedEmissionsPeakTest":
        """Set test data and calculate recent historical trend."""
        self.test_data = pd.DataFrame(test_data, columns=["year", "emissions"])
        self.test_data = self.test_data.sort_values("year").reset_index(drop=True)
        
        # Calculate test slope
        X_test = self.test_data["year"].values.reshape(-1, 1)
        y_test = self.test_data["emissions"].values
        model_test = LinearRegression()
        model_test.fit(X_test, y_test)
        
        self.test_slope = model_test.coef_[0]
        self.test_r2 = model_test.score(X_test, y_test)
        
        # Calculate recent historical trend for null hypothesis
        recent_data = self.historical_data.tail(recent_years_for_trend)
        X_recent = recent_data["year"].values.reshape(-1, 1)
        y_recent = recent_data["emissions"].values
        
        model_recent = LinearRegression()
        model_recent.fit(X_recent, y_recent)
        self.recent_historical_trend = model_recent.coef_[0]
        
        print(f"Test data: {len(test_data)} years, slope = {self.test_slope:.2f} Mt/yr")
        print(f"Recent historical trend: {self.recent_historical_trend:.2f} Mt/yr")
        
        return self
    
    def run_complete_bootstrap_test(self, n_bootstrap: int = 10000, 
                                   null_hypothesis: str = "recent_trend",
                                   bootstrap_method: str = "ar_bootstrap") -> Dict:
        """
        Run complete bootstrap test with all enhancements.
        
        Args:
            null_hypothesis: "recent_trend" or "zero_trend"
            bootstrap_method: "ar_bootstrap", "block_bootstrap", or "white_noise"
        """
        if self.noise_generator is None:
            self.create_noise_generator()
        
        print(f"Running complete bootstrap test...")
        print(f"  Null hypothesis: {null_hypothesis}")
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
            'significant_at_0_05': p_value_one_tail < 0.05,
            'significant_at_0_01': p_value_one_tail < 0.01,
            'null_hypothesis': null_hypothesis,
            'bootstrap_method': bootstrap_method,
            'effect_size': effect_size,
            'null_mean': null_mean,
            'null_std': null_std,
            'n_bootstrap': n_bootstrap,
            'autocorr_phi': self.autocorr_params['phi'],
            'n_segments': len(self.optimal_segments['best']['segments'])
        }
        
        print(f"Results:")
        print(f"  P-value: {p_value_one_tail:.4f}")
        print(f"  Significant at α=0.05: {self.bootstrap_results['significant_at_0_05']}")
        print(f"  Effect size: {effect_size:.2f} standard deviations")
        
        return self.bootstrap_results
    
    def _generate_bootstrap_slopes(self, n_bootstrap: int, null_hypothesis: str, 
                                  bootstrap_method: str) -> np.ndarray:
        """Generate bootstrap slope distribution."""
        bootstrap_slopes = []
        n_test_points = len(self.test_data)
        years = np.arange(2020, 2020 + n_test_points)  # Arbitrary years for test
        
        # Baseline emissions level
        baseline_emissions = np.mean(self.test_data["emissions"])
        
        # Null hypothesis trend
        if null_hypothesis == "recent_trend":
            null_trend = self.recent_historical_trend
        else:  # zero_trend
            null_trend = 0.0
        
        for i in range(n_bootstrap):
            # Generate null hypothesis emissions trajectory
            trend_component = null_trend * (years - years[0])
            base_emissions = baseline_emissions + trend_component
            
            # Add autocorrelated noise
            if bootstrap_method == "ar_bootstrap" and self.autocorr_params['has_autocorr']:
                # Use AR(1) noise generator
                noise = self.noise_generator(n_test_points, initial_value=0)
            elif bootstrap_method == "block_bootstrap":
                # Simple block bootstrap (for demo - could be improved)
                block_size = max(2, n_test_points // 2)
                noise = self._block_bootstrap_sample(block_size)
            else:  # white_noise
                noise = np.random.normal(0, self.autocorr_params['sigma_innovation'], n_test_points)
            
            null_emissions = base_emissions + noise
            
            # Calculate slope
            X = years.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, null_emissions)
            bootstrap_slopes.append(model.coef_[0])
        
        return np.array(bootstrap_slopes)
    
    def _block_bootstrap_sample(self, block_size: int) -> np.ndarray:
        """Generate a block bootstrap sample from residuals."""
        n_needed = len(self.test_data)
        residuals = self.residuals_temporal
        
        if len(residuals) < block_size:
            return np.random.choice(residuals, n_needed, replace=True)
        
        # Generate blocks
        sample = []
        while len(sample) < n_needed:
            start_idx = np.random.randint(0, len(residuals) - block_size + 1)
            block = residuals[start_idx:start_idx + block_size]
            sample.extend(block)
        
        return np.array(sample[:n_needed])
    
    def interpret_results(self) -> Dict[str, str]:
        """Provide comprehensive interpretation of results."""
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap test first")
        
        r = self.bootstrap_results
        
        # Determine significance level and strength
        if r['p_value_one_tail'] < 0.001:
            strength = "very strong"
        elif r['p_value_one_tail'] < 0.01:
            strength = "strong"
        elif r['p_value_one_tail'] < 0.05:
            strength = "moderate"
        else:
            strength = "weak"
        
        # Interpret relative to null hypothesis
        if r['null_hypothesis'] == 'recent_trend':
            if r['test_slope'] < r['recent_historical_trend'] and r['significant_at_0_05']:
                conclusion = f"{strength.capitalize()} evidence of accelerated emissions decline"
            elif r['test_slope'] < 0 and not r['significant_at_0_05']:
                conclusion = "Emissions declining but not significantly faster than recent trend"
            else:
                conclusion = "No evidence of accelerated emissions decline"
        else:  # zero_trend
            if r['test_slope'] < 0 and r['significant_at_0_05']:
                conclusion = f"{strength.capitalize()} evidence that emissions have peaked"
            elif r['test_slope'] < 0:
                conclusion = "Declining trend but not statistically significant"
            else:
                conclusion = "No evidence of emissions peak"
        
        interpretation = {
            'observed_slope': f"{r['test_slope']:.2f} Mt/year",
            'null_trend': f"{r['recent_historical_trend']:.2f} Mt/year" if r['null_hypothesis'] == 'recent_trend' else "0.00 Mt/year",
            'p_value': f"{r['p_value_one_tail']:.4f}",
            'strength': strength,
            'conclusion': conclusion,
            'effect_size': f"{r['effect_size']:.2f} standard deviations",
            'autocorr_effect': "Yes" if self.autocorr_params['has_autocorr'] else "No",
            'n_segments': str(r['n_segments'])
        }
        
        return interpretation
    
    def plot_complete_analysis(self, figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
        """Create comprehensive visualization."""
        # Changed to 3x3 grid to accommodate residuals density plot
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # 1. Historical data with segments and test data
        self._plot_segmented_data(axes[0, 0])
        
        # 2. Residuals and autocorrelation
        self._plot_residuals_autocorr(axes[0, 1])
        
        # 3. Bootstrap distribution
        self._plot_bootstrap_distribution(axes[0, 2])
        
        # 4. Trend comparison
        self._plot_trend_comparison(axes[1, 0])
        
        # 5. Diagnostic plots
        self._plot_diagnostics(axes[1, 1])
        
        # 6. Summary
        self._plot_summary(axes[1, 2])
        
        # 7. NEW: Residuals density distribution
        self._plot_residuals_density(axes[2, 0])
        
        # Hide unused subplots
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        return fig

    def _plot_segmented_data(self, ax):
        """Plot historical data with optimal segments - MODIFIED for darker colors."""
        # Historical data
        ax.plot(self.historical_data["year"], self.historical_data["emissions"], 
                'b-', alpha=0.6, linewidth=1, label="Historical emissions")
        
        # Segments with darker colors
        segments = self.optimal_segments['best']['segments']
        # Use darker colormap and increase line width
        colors = plt.cm.Dark2(np.linspace(0, 1, len(segments)))  # Changed from Set3 to Dark2
        
        for i, (segment, color) in enumerate(zip(segments, colors)):
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            
            ax.plot(segment['years'], trend, color=color, linewidth=3,  # Increased from 2 to 3
                   label=f"Segment {i+1}" if i < 3 else "", alpha=0.9)  # Added alpha=0.9
        
        # Test data
        ax.plot(self.test_data["year"], self.test_data["emissions"], 
                'ro', markersize=8, label=f"Test data (slope={self.test_slope:.1f})")
        
        ax.set_xlabel("Year")
        ax.set_ylabel("CO₂ Emissions (Mt)")
        ax.set_title(f"Segmentation: {self.optimal_segments['best']['method']} ({len(segments)} segments)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals_density(self, ax):
        """NEW: Plot density distribution of residuals."""
        # Get all residuals from segments
        all_residuals = []
        segments = self.optimal_segments['best']['segments']
        
        for segment in segments:
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            all_residuals.extend(residuals)
        
        all_residuals = np.array(all_residuals)
        
        # Plot histogram and density curve
        ax.hist(all_residuals, bins=20, density=True, alpha=0.7, color='lightblue', 
                edgecolor='black', label='Residuals histogram')
        
        # Add normal distribution overlay for comparison
        mu, sigma = np.mean(all_residuals), np.std(all_residuals)
        x_norm = np.linspace(all_residuals.min(), all_residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        ax.plot(x_norm, y_norm, 'r-', linewidth=2, label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
        
        # Add kernel density estimate
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(all_residuals)
            x_kde = np.linspace(all_residuals.min(), all_residuals.max(), 100)
            y_kde = kde(x_kde)
            ax.plot(x_kde, y_kde, 'g-', linewidth=2, label='Kernel density estimate')
        except ImportError:
            pass  # Skip KDE if scipy not available
        
        # Add vertical line at zero
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Statistics text
        skewness = stats.skew(all_residuals)
        kurtosis = stats.kurtosis(all_residuals)
        ax.text(0.05, 0.95, f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Residuals (Mt)")
        ax.set_ylabel("Density")
        ax.set_title("Residuals Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals_autocorr(self, ax):
        """Plot residuals and autocorrelation."""
        years_resid = []
        segments = self.optimal_segments['best']['segments']
        
        for segment in segments:
            X = segment['years'].reshape(-1, 1)
            y = segment['emissions']
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            
            ax.scatter(segment['years'], residuals, alpha=0.6, s=20)
            years_resid.extend(segment['years'])
        
        # Add autocorrelation info in text
        phi = self.autocorr_params['phi']
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Residuals (Mt)")
        ax.set_title(f"Residuals (AR(1) φ = {phi:.3f})")
        ax.grid(True, alpha=0.3)
    
    def _plot_bootstrap_distribution(self, ax):
        """Plot bootstrap distribution vs observed slope."""
        r = self.bootstrap_results
        
        ax.hist(r['bootstrap_slopes'], bins=50, density=True, alpha=0.7, 
                color='lightblue', label=f'Bootstrap distribution\n({r["bootstrap_method"]})')
        
        ax.axvline(r['test_slope'], color='red', linewidth=2,
                   label=f'Observed: {r["test_slope"]:.1f} Mt/yr')
        
        ax.axvline(r['null_mean'], color='orange', linewidth=2, linestyle='--',
                   label=f'Null mean: {r["null_mean"]:.1f} Mt/yr')
        
        ax.set_xlabel("Slope (Mt/year)")
        ax.set_ylabel("Density")
        ax.set_title(f'Bootstrap Test (p = {r["p_value_one_tail"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trend_comparison(self, ax):
        """Compare test slope with recent historical trend."""
        trends = ['Recent\nHistorical', 'Test\nPeriod']
        values = [self.recent_historical_trend, self.test_slope]
        colors = ['blue', 'red']
        
        bars = ax.bar(trends, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(values) - min(values))*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        ax.set_ylabel("Slope (Mt/year)")
        ax.set_title("Trend Comparison")
        ax.grid(True, alpha=0.3)
        
        # Add significance indicator
        if self.bootstrap_results['significant_at_0_05']:
            ax.text(0.5, 0.9, "Significantly different\n(p < 0.05)", 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    def _plot_diagnostics(self, ax):
        """Plot diagnostic information."""
        ax.axis('off')
        
        diagnostic_text = f"""
        DIAGNOSTIC INFORMATION
        ____________________
        
        Segmentation:
        • Method: {self.optimal_segments['best']['method']}
        • Number of segments: {len(self.optimal_segments['best']['segments'])}
        
        Autocorrelation:
        • AR(1) coefficient: {self.autocorr_params['phi']:.3f}
        • Has autocorrelation: {self.autocorr_params['has_autocorr']}
        • Innovation σ: {self.autocorr_params['sigma_innovation']:.1f}
        
        Bootstrap:
        • Method: {self.bootstrap_results['bootstrap_method']}
        • Samples: {self.bootstrap_results['n_bootstrap']:,}
        • Null hypothesis: {self.bootstrap_results['null_hypothesis']}
        """
        
        ax.text(0.05, 0.95, diagnostic_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
    
    def _plot_summary(self, ax):
        """Plot summary and interpretation."""
        ax.axis('off')
        
        interpretation = self.interpret_results()
        
        summary_text = f"""
        RESULTS SUMMARY
        _______________
        
        Test Results:
        • Observed slope: {interpretation['observed_slope']}
        • Null trend: {interpretation['null_trend']}
        • P-value: {interpretation['p_value']}
        • Effect size: {interpretation['effect_size']}
        
        Statistical Evidence:
        • Strength: {interpretation['strength']}
        • Autocorr-adjusted: {interpretation['autocorr_adjustment']}
        
        CONCLUSION:
        {interpretation['conclusion']}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))


# # Example usage with realistic data
# if __name__ == "__main__":
#     print("Complete Enhanced Emissions Peak Test")
#     print("=" * 50)
    
#     # Create realistic synthetic data
#     np.random.seed(42)
    
#     # Historical data (1970-2019) with structural changes and autocorrelation
#     years = np.arange(1970, 2020)
#     n_years = len(years)
    
#     # Base trend with structural break around 1990
#     base_trend = np.where(years <= 1990, 
#                          100 * (years - 1970) + 25000,  # Growth phase
#                          -30 * (years - 1990) + 100 * (1990 - 1970) + 25000)  # Decline phase
    
#     # Add autocorrelated noise
#     phi_true = 0.4  # Moderate autocorrelation
#     sigma_innovation = 600
    
#     innovations = np.random.normal(0, sigma_innovation, n_years)
#     noise = np.zeros(n_years)
#     noise[0] = innovations[0]
    
#     for t in range(1, n_years):
#         noise[t] = phi_true * noise[t-1] + innovations[t]
    
#     emissions = base_trend + noise
    
#     # Ensure realistic values (no negative emissions)
#     emissions = np.maximum(emissions, 1000)
    
#     historical_data = pd.DataFrame({
#         'year': years,
#         'emissions': emissions
#     })
    
#     # Test data showing potential acceleration in decline
#     test_data = [
#         (2020, 37200),
#         (2021, 36800),
#         (2022, 36200),
#         (2023, 35400)  # Faster decline than recent historical trend
#     ]
    
#     print(f"True autocorrelation φ: {phi_true}")
#     print(f"True innovation σ: {sigma_innovation}")
#     print()
    
#     # Initialize and run complete analysis
#     complete_test = CompleteEnhancedEmissionsPeakTest()
    
#     # Chain all methods together - now this should work properly
#     results = (complete_test
#                .load_historical_data(historical_data)
#                .optimize_segments(min_segment=5, max_segment=15)
#                .analyze_autocorrelation()  # This will now be called automatically
#                .create_noise_generator()
#                .set_test_data(test_data, recent_years_for_trend=10)
#                .run_complete_bootstrap_test(
#                    n_bootstrap=5000,
#                    null_hypothesis="recent_trend",  # Test for acceleration
#                    bootstrap_method="ar_bootstrap"
#                ))
    
#     # Print interpretation
#     print()
#     interpretation = complete_test.interpret_results()
#     print("FINAL INTERPRETATION:")
#     print("=" * 30)
#     for key, value in interpretation.items():
#         print(f"{key.replace('_', ' ').title()}: {value}")
    
#     # Create comprehensive plot
#     print("\nGenerating comprehensive analysis plot...")
#     complete_test.plot_complete_analysis()
    
#     # Compare with naive approach (for demonstration)
#     print("\n" + "="*50)
#     print("COMPARISON WITH NAIVE APPROACH")
#     print("="*50)
    
#     # Run the same test but with naive assumptions
#     print("\nNaive approach (zero trend null, no autocorr, single segment):")
    
#     # Simple single-segment analysis
#     X_hist = historical_data["year"].values.reshape(-1, 1)
#     y_hist = historical_data["emissions"].values
#     model_hist = LinearRegression()
#     model_hist.fit(X_hist, y_hist)
#     naive_residuals = y_hist - model_hist.predict(X_hist)
#     naive_sigma = np.std(naive_residuals)
    
#     # Naive bootstrap (white noise, zero trend null)
#     n_bootstrap_naive = 5000
#     bootstrap_slopes_naive = []
#     test_years = np.array([2020, 2021, 2022, 2023])
#     test_emissions = np.array([37200, 36800, 36200, 35400])
#     baseline_naive = np.mean(test_emissions)
    
#     # Calculate observed slope
#     X_test_naive = test_years.reshape(-1, 1)
#     model_test_naive = LinearRegression()
#     model_test_naive.fit(X_test_naive, test_emissions)
#     observed_slope_naive = model_test_naive.coef_[0]
    
#     for _ in range(n_bootstrap_naive):
#         # Zero trend null + white noise
#         null_emissions_naive = baseline_naive + np.random.normal(0, naive_sigma, 4)
        
#         X = test_years.reshape(-1, 1)
#         model = LinearRegression()
#         model.fit(X, null_emissions_naive)
#         bootstrap_slopes_naive.append(model.coef_[0])
    
#     p_value_naive = np.sum(np.array(bootstrap_slopes_naive) <= observed_slope_naive) / n_bootstrap_naive
    
#     print(f"  Observed slope: {observed_slope_naive:.2f} Mt/year")
#     print(f"  P-value: {p_value_naive:.4f}")
#     print(f"  Significant at α=0.05: {p_value_naive < 0.05}")
#     print(f"  Residual σ: {naive_sigma:.1f} Mt")
    
#     # Compare results
#     print("\n" + "="*50)
#     print("COMPARISON SUMMARY")
#     print("="*50)
    
#     enhanced_p = complete_test.bootstrap_results['p_value_one_tail']
#     enhanced_sig = complete_test.bootstrap_results['significant_at_0_05']
    
#     print(f"Enhanced approach:")
#     print(f"  - P-value: {enhanced_p:.4f}")
#     print(f"  - Significant: {enhanced_sig}")
#     print(f"  - Null hypothesis: {complete_test.bootstrap_results['null_hypothesis']}")
#     print(f"  - Autocorr φ: {complete_test.autocorr_params['phi']:.3f}")
#     print(f"  - Segments: {len(complete_test.optimal_segments['best']['segments'])}")
    
#     print(f"\nNaive approach:")
#     print(f"  - P-value: {p_value_naive:.4f}")
#     print(f"  - Significant: {p_value_naive < 0.05}")
#     print(f"  - Null hypothesis: zero_trend")
#     print(f"  - Autocorr φ: 0.000 (ignored)")
#     print(f"  - Segments: 1")
    
#     print(f"\nKey differences:")
#     print(f"  - P-value ratio (enhanced/naive): {enhanced_p/p_value_naive:.2f}")
#     if enhanced_sig != (p_value_naive < 0.05):
#         print(f"  - Different significance conclusions!")
#     else:
#         print(f"  - Same significance conclusion")
    
#     # Show why the approaches differ
#     print(f"\nWhy they differ:")
#     print(f"  1. Null hypothesis: Enhanced uses recent trend ({complete_test.recent_historical_trend:.1f}), naive uses zero")
#     print(f"  2. Autocorrelation: Enhanced accounts for φ={complete_test.autocorr_params['phi']:.3f}, naive ignores")
#     print(f"  3. Segmentation: Enhanced uses {len(complete_test.optimal_segments['best']['segments'])} segments, naive uses 1")
#     print(f"  4. Noise model: Enhanced σ={complete_test.autocorr_params['sigma_innovation']:.1f}, naive σ={naive_sigma:.1f}")


    def demonstrate_method_sensitivity():
        """
        Demonstrate how different methodological choices affect results.
        """
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS: DIFFERENT METHODOLOGICAL CHOICES")
        print("="*60)
        
        # Use the same data as before
        np.random.seed(42)
        years = np.arange(1970, 2020)
        phi_true = 0.4
        sigma_innovation = 600
        
        base_trend = np.where(years <= 1990, 
                             100 * (years - 1970) + 25000,
                             -30 * (years - 1990) + 100 * (1990 - 1970) + 25000)
        
        innovations = np.random.normal(0, sigma_innovation, len(years))
        noise = np.zeros(len(years))
        noise[0] = innovations[0]
        
        for t in range(1, len(years)):
            noise[t] = phi_true * noise[t-1] + innovations[t]
        
        emissions = np.maximum(base_trend + noise, 1000)
        historical_data = pd.DataFrame({'year': years, 'emissions': emissions})
        
        test_data = [(2020, 37200), (2021, 36800), (2022, 36200), (2023, 35400)]
        
        # Test different combinations
        scenarios = [
            {"null": "zero_trend", "bootstrap": "white_noise", "name": "Naive"},
            {"null": "zero_trend", "bootstrap": "ar_bootstrap", "name": "Autocorr only"},
            {"null": "recent_trend", "bootstrap": "white_noise", "name": "Recent trend only"},
            {"null": "recent_trend", "bootstrap": "ar_bootstrap", "name": "Full enhanced"},
        ]
        
        results_comparison = {}
        
        for scenario in scenarios:
            print(f"\nTesting: {scenario['name']}")
            
            test_instance = CompleteEnhancedEmissionsPeakTest()
            
            try:
                test_instance.load_historical_data(historical_data)
                test_instance.optimize_segments()
                test_instance.analyze_autocorrelation()
                test_instance.create_noise_generator()
                test_instance.set_test_data(test_data)
                
                results = test_instance.run_complete_bootstrap_test(
                    n_bootstrap=2000,
                    null_hypothesis=scenario["null"],
                    bootstrap_method=scenario["bootstrap"]
                )
                
                results_comparison[scenario["name"]] = {
                    'p_value': results['p_value_one_tail'],
                    'significant': results['significant_at_0_05'],
                    'effect_size': results['effect_size'],
                    'null_trend': results.get('recent_historical_trend', 0) if scenario["null"] == "recent_trend" else 0
                }
                
                print(f"  P-value: {results['p_value_one_tail']:.4f}")
                print(f"  Significant: {results['significant_at_0_05']}")
                print(f"  Effect size: {results['effect_size']:.2f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                results_comparison[scenario["name"]] = None
        
        # Summary table
        print(f"\n{'Method':<20} {'P-value':<10} {'Significant':<12} {'Effect Size':<12}")
        print("-" * 60)
        
        for name, result in results_comparison.items():
            if result:
                sig_str = "Yes" if result['significant'] else "No"
                print(f"{name:<20} {result['p_value']:<10.4f} {sig_str:<12} {result['effect_size']:<12.2f}")
            else:
                print(f"{name:<20} {'Error':<10} {'Error':<12} {'Error':<12}")
        
        print(f"\nKey insights:")
        print(f"- Different methodological choices can lead to different conclusions")
        print(f"- Accounting for autocorrelation generally increases p-values (more conservative)")
        print(f"- Using recent trend as null hypothesis provides more realistic baseline")
        print(f"- Effect sizes help interpret practical significance beyond statistical significance")
    
    
    


    
# if __name__ == "__main__":
#     # Run the main analysis first (code above runs automatically)
#     pass
    
#     # Then run sensitivity analysis
#     demonstrate_method_sensitivity()
    
#     print(f"\n" + "="*60)
#     print("RECOMMENDATIONS FOR YOUR ANALYSIS")
#     print("="*60)
    
#     print(f"""
#     Based on this comprehensive implementation, here are key recommendations:
    
#     1. ALWAYS use recent historical trend as null hypothesis
#        - More realistic than zero-trend assumption
#        - Tests for acceleration, not just decline
#        - Reduces false positives
    
#     2. ALWAYS check and account for autocorrelation
#        - Emissions data typically shows persistence
#        - Ignoring it leads to overconfident conclusions
#        - AR(1) model usually sufficient
    
#     3. USE dynamic segmentation when appropriate
#        - Captures structural breaks in emissions trajectories
#        - More robust noise estimation
#        - Better handles policy regime changes
    
#     4. REPORT effect sizes alongside p-values
#        - Helps distinguish statistical vs practical significance
#        - More informative than binary significant/not significant
    
#     5. PERFORM sensitivity analysis
#        - Test different methodological choices
#        - Assess robustness of conclusions
#        - Build confidence in results
    
#     6. VISUALIZE comprehensively
#        - Show segmentation, autocorrelation, bootstrap distribution
#        - Make methodological choices transparent
#        - Aid in interpretation and communication
#     """)
    
#     print(f"Analysis complete! 🎉")