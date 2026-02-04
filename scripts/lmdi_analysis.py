from typing import List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from scipy import stats
from pathlib import Path
from pandas_indexing import isin
import warnings
warnings.filterwarnings('ignore')

class LMDIPeakAnalyzer:

    def __init__(self):
        """Initialize LMDI Peak Detection Analyzer using OWID data"""
        self.target_entities = [
            'China', 'United States', 'India', 'European Union (27)', 'World',
            'Indonesia', 'Brazil', 'Mexico'
        ]
        self.historical_data = {}
        self.variables = {}
        self.regions = None
        self.merged_data = None
        self.lmdi_results = {}
        self.peak_signals = {}
        self.report = None


    def load_all_data(self,
                      data_folder: str = "./data/processed",  # Just specify the folder
                      regions: Union[str, List[str]] = ["WLD", "CHN"],
                      year_range: range = range(1980, 2025),
                      ) -> "LMDIPeakAnalyzer":
        
        """
        Load all required datasets using standard filenames.

        Expects files: gcb_hist_co2.csv, carbon_intensity_energy.csv, energy_intensity_gdp.csv, wdi_gdp.csv
        """
        data_sources = {
            'co2_emissions': 'gcb_hist_co2.csv',
            'carbon_intensity': 'carbon_intensity_energy.csv',
            'energy_intensity': 'energy_intensity_gdp.csv',
            'gdp': 'wdi_gdp.csv',
            'primary_energy': 'primary_energy.csv',
            'carbon_intensity_gdp': 'carbon_intensity_gdp.csv'
        }
        
        for variable_name, filename in data_sources.items():
            self._load_historical_data(
                data_source=filename,
                variable_name=variable_name,
                regions=regions,
                year_range=year_range
            )
        
        self._validate_all_data()
        
        return self


    def _load_historical_data(
        self,
        data_source: Union[str, pd.DataFrame],
        variable_name: str,
        regions: Union[str, List[str]],
        year_range: range = range(1990, 2024),
        ) -> "LMDIPeakAnalyzer":
        """
        Load historical emissions data.

        Load historical data for a specific variable across one or more regions.
        Args:
            data_source: Path to CSV file or pandas DataFrame
            variable_name: Identifier for this dataset (e.g., 'emissions', 'temperature', 'gdp')
            regions: Single region code (str) or list of region codes (e.g., ['USA', 'CHN', 'IND'])
            year_range: Range of years to include
        Returns:
            Self for method chaining
        """
        
            # Handle regions input
        if isinstance(regions, str):
            regions = [regions]  # Convert single region to list
        
        if self.regions is None:
            self.regions = regions
        else:
            # Optionally verify consistency across datasets
            if set(regions) != set(self.regions):
                print(f"Warning: Loading {variable_name} with different regions than previous datasets")
                print(f"Previous: {self.regions}, Current: {regions}")
        
        if isinstance(data_source, str):
            data_path = Path(__file__).resolve().parent / f"../data/processed/{data_source}"
            try:
                data = pd.read_csv(data_path, index_col=[0, 1, 2, 3, 4])
            except Exception as e:
                raise ValueError(f"Could not load data from {data_path}: {e}")
        elif isinstance(data_source, pd.DataFrame):
            assert data_source.index.names == ["region", "region_name", "variable", "unit"]
            data = data_source.copy()
        else:
            raise ValueError("data_source must be a file path or DataFrame")
        
        # Filter to selected regions
        data = data.loc[isin(region=regions)]
        
        # Store variable and unit info by variable_name
        self.variables[variable_name] = data.pix.unique("variable")[0]

    
            # Process data
        data.columns = data.columns.astype(int)
        
        # Handle multiple regions - reshape to long form with region column
        data_long = []
        for region in regions:
            try:
                region_data = data.loc[isin(region=region)]
                series = pd.Series(index=region_data.columns, data=region_data.values.squeeze())
                df = series.reset_index()
                df.columns = ["year", variable_name]
                df["region"] = region
                data_long.append(df)
            except Exception as e:
                print(f"Warning: Could not load data for region {region}: {e}")
        
        # Combine all regions
        if data_long:
            combined_data = pd.concat(data_long, ignore_index=True)
        else:
            raise ValueError(f"No data loaded for any of the specified regions: {regions}")
        
        # Filter to selected years
        filtered_data = combined_data.loc[combined_data["year"].isin(year_range)]
        
        # Store in dictionary
        self.historical_data[variable_name] = filtered_data

        print(f"Loaded {variable_name} data for {len(regions)} region(s): {regions}")
        print(f"Year range: {filtered_data['year'].min()}-{filtered_data['year'].max()}")
        print(f"Data points: {len(filtered_data)}")
        
        return self
    
    def _validate_all_data(self):
        """
        Validate that all required datasets are present and compatible.
        """
        required_vars = ['co2_emissions', 'carbon_intensity', 'energy_intensity', 'gdp']
        
        # Check all datasets are loaded
        for var in required_vars:
            if var not in self.historical_data:
                raise ValueError(f"Required dataset '{var}' not loaded")
        
        # Check that all datasets have the same regions and years (optional but recommended)
        reference_var = required_vars[0]
        ref_data = self.historical_data[reference_var]
        ref_regions = set(ref_data['region'].unique())
        ref_years = set(ref_data['year'].unique())
        
        for var in required_vars[1:]:
            data = self.historical_data[var]
            var_regions = set(data['region'].unique())
            var_years = set(data['year'].unique())
            
            if var_regions != ref_regions:
                print(f"Warning: {var} has different regions than {reference_var}")
                print(f"  Missing in {var}: {ref_regions - var_regions}")
                print(f"  Extra in {var}: {var_regions - ref_regions}")
            
            if var_years != ref_years:
                print(f"Warning: {var} has different years than {reference_var}")
        
        print("✓ Data validation complete")

    # Merge historical data for LMDI analysis
    def prepare_lmdi_data(self, force_merge: bool = False):

        """
        Prepare data for LMDI decomposition
            Args:
        force_merge: If True, re-merge historical data even if merged_data exists.
                    If False, only merge if merged_data doesn't exist yet.
        """
        print("Preparing LMDI variables...")
        
        # Only merge if needed
        if self.merged_data is None or self.merged_data.empty or force_merge:
            df = self._merge_historical_data()
            
            if df.empty:
                print("No data available for LMDI analysis")
                return pd.DataFrame()
            
            self.merged_data = df
        else:
            print("Using existing merged_data (use force_merge=True to re-merge)")
            df = self.merged_data

        # Create LMDI variables for each entity
        lmdi_data = []
        
        for entity in df['country'].unique():
            entity_data = df[df['country'] == entity].sort_values('year').copy()
            
            # Need at least 3 years for analysis
            if len(entity_data) < 3:
                continue
            
            # LMDI factors: CO2 = Activity × (Energy/Activity) × (CO2/Energy)
            entity_data['activity'] = entity_data['gdp']
            entity_data['energy_intensity'] = entity_data['energy_intensity']  # Already primary_energy / gdp
            entity_data['carbon_intensity'] = entity_data['carbon_intensity']  # Already co2 / primary_energy
            entity_data['emissions'] = entity_data['co2']
            
            # Preserve estimation flag if it exists (for recent estimates)
            if 'is_estimated' not in entity_data.columns:
                entity_data['is_estimated'] = False
            
            # Clean data - remove invalid values
            required_vars = ['activity', 'energy_intensity', 'carbon_intensity', 'emissions']
            entity_data = entity_data.replace([np.inf, -np.inf], np.nan)
            valid_mask = (entity_data[required_vars] > 0).all(axis=1)
            entity_data = entity_data[valid_mask]
            
            if len(entity_data) >= 3:
                lmdi_data.append(entity_data)
                actual_years = (entity_data['is_estimated'] != True).sum()
                estimated_years = entity_data['is_estimated'].sum()
                print(f"  Prepared {entity}: {len(entity_data)} years ({actual_years} actual, {estimated_years} estimated)")
        
        if lmdi_data:
            self.lmdi_data = pd.concat(lmdi_data, ignore_index=True)
            total_estimated = (self.lmdi_data['is_estimated'] == True).sum()
            print(f"LMDI data prepared for {self.lmdi_data['country'].nunique()} entities")
            print(f"Total estimated data points: {total_estimated}")
        else:
            self.lmdi_data = pd.DataFrame()
            print("Warning: No valid LMDI data prepared")
        
        return self.lmdi_data
        
    def _merge_historical_data(self) -> pd.DataFrame:
        """Merge all historical datasets into a single DataFrame"""
        if not self.historical_data:
            return pd.DataFrame()
        
        # Start with emissions data
        df = self.historical_data['co2_emissions'].copy()
        df = df.rename(columns={'co2_emissions': 'co2'})
        
        # Merge other datasets
        merge_vars = ['gdp', 'energy_intensity', 'carbon_intensity', 'primary_energy']
        for var_name in merge_vars:
            if var_name in self.historical_data:
                temp_df = self.historical_data[var_name][['year', 'region', var_name]]
                df = df.merge(temp_df, on=['year', 'region'], how='outer')
        
        # Rename region to country and calculate primary energy
        df = df.rename(columns={'region': 'country'})
        # df['primary_energy'] = df['energy_intensity'] * df['gdp'] # Not needed if primary_energy is loaded directly
        
        return df
    
    def add_test_data(self, test_data):
        """
        Add test data for 2025 projections
        Args:
            test_scenarios: Dict with country-specific and year-specific data containing:
                - 'co2': CO2 emissions
                - 'gdp': GDP/activity
                - 'primary_energy': Total primary energy
                
        Example:
            test_scenarios = {
                2025: {
                    'USA': {'co2': 4800, 'gdp': 26000, 'primary_energy': 3380},
                    'CHN': {'co2': 11500, 'gdp': 19000, 'primary_energy': 4370}
                },
                2026: {
                    'USA': {'co2': 4750, 'gdp': 26500, 'primary_energy': 3350},
                    'CHN': {'co2': 11600, 'gdp': 19500, 'primary_energy': 4400}
                }
            }
        """
        if self.merged_data is None or self.merged_data.empty:
            print("No merged data available. Run prepare_lmdi_data() first.")
            return
        
        # Create 2025 test data
        test_data_rows = []
        
        for year, country_data in test_data.items():
            for country, scenario in country_data.items():
                # Calculate intensities from input data
                energy_intensity = scenario['primary_energy'] / scenario['gdp']
                carbon_intensity = scenario['co2'] / scenario['primary_energy']
                
                new_row = {
                'year': year,
                'country': country,
                'co2': scenario['co2'],
                'gdp': scenario['gdp'],
                'primary_energy': scenario['primary_energy'],
                'energy_intensity': energy_intensity,
                'carbon_intensity': carbon_intensity,
                'is_estimated': True
            }
            
            test_data_rows.append(new_row)
        
        # Create DataFrame from test data
        test_df = pd.DataFrame(test_data_rows)
        # Append to merged_data
        self.merged_data = pd.concat([self.merged_data, test_df], ignore_index=True)
        print(f"\nTotal data points now: {len(self.merged_data)}")

        # Re-run LMDI preparation with new data
        self.prepare_lmdi_data(force_merge=False)
        return test_df

## Update to 2025 values and to new structure
    def add_2024_co2_estimates(self):
        """Add 2024 CO2 estimates from GCB to the merged dataset"""
        print("Adding 2024 CO2 estimates from Global Carbon Budget...")
        
        if self.merged_data is None or self.merged_data.empty:
            print("No merged data available to add estimates to")
            return
        
        # 2024 CO2 estimates from GCB (in tonnes to match historical data)
        co2_estimates_2024 = {
            'China': 11957000000.0,        # 11.957 GtCO2 = 11,957,000,000 tonnes
            'India': 3211500000.0,         # 3.2115 GtCO2 = 3,211,500,000 tonnes 
            'United States': 4895840000.0, # 4.89584 GtCO2 = 4,895,840,000 tonnes
            'European Union (27)': 2421540000.0,  # 2.42154 GtCO2 = 2,421,540,000 tonnes
            'World': 37400000000+770000000 # 37.4 Gt minus 0.77 Gt cement carbonation sink (to be consistent with other numbers)
        }
        
        # Add estimated flag column if it doesn't exist
        if 'is_estimated' not in self.merged_data.columns:
            self.merged_data['is_estimated'] = False
        
        # Update CO2 values and estimation flags
        updated_entities = []
        
        for entity, co2_estimate in co2_estimates_2024.items():
            # Find 2024 row for this entity
            mask = (self.merged_data['country'] == entity) & (self.merged_data['year'] == 2024)
            
            if mask.sum() == 1:
                # Update CO2 and flag as estimated
                self.merged_data.loc[mask, 'co2'] = co2_estimate
                self.merged_data.loc[mask, 'is_estimated'] = True
                updated_entities.append(entity)
                print(f"  Updated {entity}: CO2 = {co2_estimate:.1f} tonnes")
            else:
                print(f"  Warning: Could not find unique 2024 row for {entity}")
        
        # DON'T create or modify carbon_intensity column here - let prepare_lmdi_data() handle it
        
        print(f"Successfully updated estimates for: {updated_entities}")
        print(f"Total rows with estimated data: {(self.merged_data['is_estimated'] == True).sum()}")
        
        return self.merged_data
    
    
    def calculate_lmdi_decomposition(self, entity_data):
        """Calculate LMDI decomposition for an entity, preserving estimation flags"""
        results = []
        entity_data = entity_data.sort_values('year').reset_index(drop=True)
        
        for i in range(1, len(entity_data)):
            year = entity_data.loc[i, 'year']
            
            # Get values for base year (t-1) and current year (t)
            A0, A1 = entity_data.loc[i-1, 'activity'], entity_data.loc[i, 'activity']
            I0, I1 = entity_data.loc[i-1, 'energy_intensity'], entity_data.loc[i, 'energy_intensity']
            C0, C1 = entity_data.loc[i-1, 'carbon_intensity'], entity_data.loc[i, 'carbon_intensity']
            E0, E1 = entity_data.loc[i-1, 'emissions'], entity_data.loc[i, 'emissions']
            
            # Check if current year data is estimated
            is_estimated_current = entity_data.loc[i, 'is_estimated'] if 'is_estimated' in entity_data.columns else False
            is_estimated_previous = entity_data.loc[i-1, 'is_estimated'] if 'is_estimated' in entity_data.columns else False
            
            # Skip if any values are invalid
            if any(val <= 0 or np.isnan(val) for val in [A0, A1, I0, I1, C0, C1, E0, E1]):
                continue
            
            # Logarithmic mean for LMDI
            def log_mean(x, y):
                if abs(x - y) < 1e-10:
                    return (x + y) / 2
                return (x - y) / (np.log(x) - np.log(y))
            
            L = log_mean(E1, E0)
            
            # LMDI decomposition effects
            activity_effect = L * np.log(A1 / A0)
            intensity_effect = L * np.log(I1 / I0)  
            carbon_effect = L * np.log(C1 / C0)
            
            total_change = E1 - E0
            computed_change = activity_effect + intensity_effect + carbon_effect
            
            results.append({
                'year': year,
                'activity_effect': activity_effect,
                'intensity_effect': intensity_effect,
                'carbon_effect': carbon_effect,
                'total_change': total_change,
                'computed_change': computed_change,
                'residual': total_change - computed_change,
                'activity_growth': (A1 / A0) - 1,
                'energy_intensity_change': (I1 / I0) - 1,
                'carbon_intensity_change': (C1 / C0) - 1,
                'emissions_change': (E1 / E0) - 1,
                'carbon_intensity': C1,
                'energy_intensity': E1,
                'is_estimated': is_estimated_current or is_estimated_previous  # Flag if either year is estimated
            })
        
        return pd.DataFrame(results)
    

    def analyze_all_entities(self):
        """Run LMDI analysis for all entities"""
        print("Running LMDI decomposition analysis...")
        
        self.lmdi_results = {}
        
        for entity in self.lmdi_data['country'].unique():
            entity_data = self.lmdi_data[self.lmdi_data['country'] == entity]
            
            if len(entity_data) >= 3:
                lmdi_result = self.calculate_lmdi_decomposition(entity_data)
                
                if len(lmdi_result) > 0:
                    self.lmdi_results[entity] = lmdi_result
                    print(f"  Analyzed {entity}: {len(lmdi_result)} years")
        
        return self.lmdi_results
    
    def detect_structural_peaks(self, window_years=3, threshold_years=2):
        """Detect structural peak signals using both raw and smoothed LMDI trends"""
        print("Detecting structural peak signals...")
        
        self.peak_signals = {}
        
        for entity, lmdi_data in self.lmdi_results.items():
            # Calculate forward-looking rolling averages
            lmdi_data = lmdi_data.copy()
            lmdi_data['activity_roll'] = lmdi_data['activity_effect'].rolling(window=window_years, min_periods=2).mean()
            lmdi_data['decoupling_roll'] = (lmdi_data['intensity_effect'] + lmdi_data['carbon_effect']).rolling(window=window_years, min_periods=2).mean()
            
            # Calculate decoupling forces (negative intensity + carbon effects mean emissions reduction)
            lmdi_data['decoupling_forces'] = -(lmdi_data['intensity_effect'] + lmdi_data['carbon_effect'])
            lmdi_data['decoupling_forces_roll'] = -lmdi_data['decoupling_roll']
            
            # Peak signals: when decoupling forces exceed activity growth
            lmdi_data['peak_signal_raw'] = lmdi_data['decoupling_forces'] > lmdi_data['activity_effect']
            lmdi_data['peak_signal_smooth'] = lmdi_data['decoupling_forces_roll'] > lmdi_data['activity_roll']
            
            # Decoupling strength ratios
            lmdi_data['decoupling_strength_raw'] = lmdi_data['decoupling_forces'] / lmdi_data['activity_effect'].where(lmdi_data['activity_effect'] > 0, np.nan)
            lmdi_data['decoupling_strength_smooth'] = lmdi_data['decoupling_forces_roll'] / lmdi_data['activity_roll'].where(lmdi_data['activity_roll'] > 0, np.nan)
            
            # Latest signals (check last 3 years)
            recent_years = 3
            if len(lmdi_data) >= recent_years:
                latest_raw_signal = lmdi_data['peak_signal_raw'].iloc[-recent_years:].sum() >= 2
                latest_smooth_signal = lmdi_data['peak_signal_smooth'].iloc[-recent_years:].sum() >= 2
                latest_raw_strength = lmdi_data['decoupling_strength_raw'].iloc[-1] if len(lmdi_data) > 0 else np.nan
                latest_smooth_strength = lmdi_data['decoupling_strength_smooth'].iloc[-1] if len(lmdi_data) > 0 else np.nan
            else:
                latest_raw_signal = latest_smooth_signal = False
                latest_raw_strength = latest_smooth_strength = np.nan
            
            self.peak_signals[entity] = {
                'lmdi_trends': lmdi_data,
                'latest_raw_signal': latest_raw_signal,
                'latest_smooth_signal': latest_smooth_signal,
                'latest_raw_strength': latest_raw_strength,
                'latest_smooth_strength': latest_smooth_strength,
                'window_years': window_years,
            }
        
        return self.peak_signals
    
    def create_peak_summary(self):
        """Create summary of peak detection results"""
        print("\nStructural Peak Detection Summary")
        print("=" * 50)
        
        summary_data = []
        
        for entity, signals in self.peak_signals.items():
            latest_raw_signal = signals['latest_raw_signal']
            latest_smooth_signal = signals['latest_smooth_signal']
            latest_raw_strength = signals['latest_raw_strength']
            latest_smooth_strength = signals['latest_smooth_strength']
            
            status = self._determine_status(latest_raw_signal, latest_smooth_signal, 
                                          latest_raw_strength, latest_smooth_strength)
            
            summary_data.append({
                'Entity': entity,
                'Latest_Raw_Signal': latest_raw_signal,
                'Latest_Smooth_Signal': latest_smooth_signal,
                'Raw_Strength_2024': round(latest_raw_strength, 2) if not np.isnan(latest_raw_strength) else None,
                'Smooth_Strength_2024': round(latest_smooth_strength, 2) if not np.isnan(latest_smooth_strength) else None,
                'Status': status
            })
            
            print(f"\n{entity}:")
            print(f"  Latest raw signal: {'Active' if latest_raw_signal else 'Inactive'}")
            print(f"  Latest smoothed signal: {'Active' if latest_smooth_signal else 'Inactive'}")
            if not np.isnan(latest_raw_strength):
                print(f"  Current raw strength: {latest_raw_strength:.2f}")
            if not np.isnan(latest_smooth_strength):
                print(f"  Current smoothed strength: {latest_smooth_strength:.2f}")
        
        return pd.DataFrame(summary_data)
    
    def _determine_status(self, raw_signal, smooth_signal, raw_strength, smooth_strength):
        """Determine peak status considering both raw and smoothed signals"""
        if raw_signal and smooth_signal and raw_strength > 1.2 and smooth_strength > 1.1:
            return "Strong Peak Signal (Both)"
        elif (raw_signal and raw_strength > 1.2) or (smooth_signal and smooth_strength > 1.1):
            return "Moderate Peak Signal"
        elif raw_signal and not smooth_signal:
            return "Emerging Peak Signal"
        elif smooth_signal and not raw_signal:
            return "Fading Peak Signal"
        else:
            return "No Clear Signal"
    
    def get_plot_data(self):
        """Extract all data needed for plotting in an easy-to-use format"""
        plot_data = {}
        
        for entity in self.peak_signals.keys():
            lmdi_data = self.peak_signals[entity]['lmdi_trends'].copy()
            
            plot_data[entity] = {
                'data': lmdi_data,
                'status': self._determine_status(
                    self.peak_signals[entity]['latest_raw_signal'],
                    self.peak_signals[entity]['latest_smooth_signal'],
                    self.peak_signals[entity]['latest_raw_strength'],
                    self.peak_signals[entity]['latest_smooth_strength']
                ),
                'has_estimates': ('is_estimated' in lmdi_data.columns and 
                                (lmdi_data['is_estimated'] == True).any())
            }
        
        return plot_data


    def plot_lmdi_trends_with_estimates(self, figsize=(18, 12)):
        """Plot LMDI trends with continuous lines extending to 2024 estimates"""
        entities = list(self.peak_signals.keys())
        n_entities = len(entities)
        
        # Calculate grid dimensions
        cols = 3
        rows = (n_entities + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if n_entities == 1 else axes
        else:
            axes = axes.ravel()
        
        for i, entity in enumerate(entities):
            ax = axes[i]
            
            lmdi_data = self.peak_signals[entity]['lmdi_trends']
            
            # Plot continuous lines for all data (actual + estimated)
            ax.plot(lmdi_data['year'], lmdi_data['activity_effect'], 
                   label='Activity Effect', linewidth=1.5, color='blue', alpha=0.8)
            ax.plot(lmdi_data['year'], lmdi_data['activity_roll'], 
                   label='Activity (3yr avg)', linewidth=2.5, color='blue')
            
            ax.plot(lmdi_data['year'], lmdi_data['decoupling_forces'], 
                   label='Decoupling Forces', linewidth=1.5, color='green', alpha=0.8)
            ax.plot(lmdi_data['year'], lmdi_data['decoupling_forces_roll'], 
                   label='Decoupling (3yr avg)', linewidth=2.5, color='green')
            
            # Highlight estimated portions with different markers and colors
            if 'is_estimated' in lmdi_data.columns:
                estimated_data = lmdi_data[lmdi_data['is_estimated'] == True]
                
                if len(estimated_data) > 0:
                    # Add markers for estimated points
                    ax.scatter(estimated_data['year'], estimated_data['activity_effect'], 
                              color='lightblue', s=80, alpha=0.9, marker='o', 
                              edgecolors='blue', linewidth=2, zorder=6,
                              label='Activity (Estimated)')
                    ax.scatter(estimated_data['year'], estimated_data['decoupling_forces'], 
                              color='lightgreen', s=80, alpha=0.9, marker='o',
                              edgecolors='green', linewidth=2, zorder=6,
                              label='Decoupling (Estimated)')
                    
                    # Add subtle shading or line style change for estimated portion
                    # Draw dashed overlay for estimated segments
                    if len(estimated_data) > 1:
                        ax.plot(estimated_data['year'], estimated_data['activity_effect'], 
                               color='lightblue', linewidth=2, linestyle='--', alpha=0.7, zorder=5)
                        ax.plot(estimated_data['year'], estimated_data['decoupling_forces'], 
                               color='lightgreen', linewidth=2, linestyle='--', alpha=0.7, zorder=5)
                    
                    # Connect last actual point to first estimated point with dotted line
                    if len(lmdi_data) > len(estimated_data):
                        actual_data = lmdi_data[lmdi_data['is_estimated'] != True]
                        if len(actual_data) > 0 and len(estimated_data) > 0:
                            last_actual_year = actual_data['year'].max()
                            first_est_year = estimated_data['year'].min()
                            
                            if last_actual_year < first_est_year:
                                # Get the transition points
                                last_actual = actual_data[actual_data['year'] == last_actual_year].iloc[0]
                                first_est = estimated_data[estimated_data['year'] == first_est_year].iloc[0]
                                
                                # Draw transition lines
                                ax.plot([last_actual_year, first_est_year], 
                                       [last_actual['activity_effect'], first_est['activity_effect']], 
                                       color='blue', linewidth=1, linestyle=':', alpha=0.6, zorder=4)
                                ax.plot([last_actual_year, first_est_year], 
                                       [last_actual['decoupling_forces'], first_est['decoupling_forces']], 
                                       color='green', linewidth=1, linestyle=':', alpha=0.6, zorder=4)
            
            # Zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Peak signal markers (red circles for crossover points)
            crossover_actual = lmdi_data[lmdi_data['peak_signal_raw'] == True]
            if len(crossover_actual) > 0:
                # Distinguish actual vs estimated crossovers
                if 'is_estimated' in crossover_actual.columns:
                    actual_crossovers = crossover_actual[crossover_actual['is_estimated'] != True]
                    estimated_crossovers = crossover_actual[crossover_actual['is_estimated'] == True]
                    
                    if len(actual_crossovers) > 0:
                        ax.scatter(actual_crossovers['year'], 
                                  actual_crossovers['decoupling_forces'],
                                  color='red', s=40, alpha=0.8, marker='o', zorder=7,
                                  label='Peak Signal')
                    
                    if len(estimated_crossovers) > 0:
                        ax.scatter(estimated_crossovers['year'], 
                                  estimated_crossovers['decoupling_forces'],
                                  color='red', s=50, alpha=0.9, marker='*', zorder=8,
                                  label='Peak Signal (Est.)')
                else:
                    ax.scatter(crossover_actual['year'], 
                              crossover_actual['decoupling_forces'],
                              color='red', s=40, alpha=0.8, marker='o', zorder=7)
            
            # Formatting
            ax.set_title(f'{entity} - LMDI Structural Peak Analysis')
            ax.set_xlabel('Year')
            ax.set_ylabel('Effect Magnitude (Mt CO2 eq/year)')
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Status text box
            status = self._determine_status(
                self.peak_signals[entity]['latest_raw_signal'],
                self.peak_signals[entity]['latest_smooth_signal'],
                self.peak_signals[entity]['latest_raw_strength'],
                self.peak_signals[entity]['latest_smooth_strength']
            )
            
            # Check if this entity has 2024 estimates
            has_2024_estimates = ('is_estimated' in lmdi_data.columns and 
                                 (lmdi_data['is_estimated'] == True).any())
            
            status_text = f'Status: {status}'
            if has_2024_estimates:
                status_text += '\n(2024: GCB estimates)'
            
            # ax.text(0.02, 0.98, status_text, 
            #        transform=ax.transAxes, fontsize=8,
            #        verticalalignment='top', 
            #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Extend x-axis slightly to show 2024 clearly
            if len(lmdi_data) > 0:
                x_min, x_max = ax.get_xlim()
                ax.set_xlim(x_min, max(x_max, lmdi_data['year'].max() + 0.5))
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_raw_lmdi_with_emissions(self, figsize=(18, 12)):
        """Plot raw LMDI effects with CO2 emissions (Fig 3 alternative)"""
        entities = list(self.peak_signals.keys())
        n_entities = len(entities)
        
        # Calculate grid dimensions
        cols = 3
        rows = (n_entities + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if n_entities == 1 else axes
        else:
            axes = axes.ravel()
        
        for i, entity in enumerate(entities):
            ax = axes[i]
            
            lmdi_data = self.peak_signals[entity]['lmdi_trends']
            
            # Get original entity data for CO2 emissions
            if hasattr(self, 'lmdi_data'):
                entity_emissions_data = self.lmdi_data[self.lmdi_data['country'] == entity].copy()
                if len(entity_emissions_data) > 0:
                    # Merge emissions data with LMDI trends data on year
                    plot_data = lmdi_data.merge(
                        entity_emissions_data[['year', 'emissions']],
                        on='year', 
                        how='left'
                    )
                else:
                    plot_data = lmdi_data.copy()
                    plot_data['emissions'] = np.nan
            else:
                plot_data = lmdi_data.copy()
                plot_data['emissions'] = np.nan
            
            # Create twin axis for emissions
            ax2 = ax.twinx()
            
            # Plot CO2 emissions on secondary axis (light grey)
            if 'emissions' in plot_data.columns and plot_data['emissions'].notna().any():
                ax2.plot(plot_data['year'], plot_data['emissions'], 
                         color='lightgrey', linewidth=2, alpha=0.7, 
                         label='CO₂ Emissions')
                
                # Highlight estimated emission points
                if 'is_estimated' in plot_data.columns:
                    estimated_emissions = plot_data[plot_data['is_estimated'] == True]
                    if len(estimated_emissions) > 0 and estimated_emissions['emissions'].notna().any():
                        ax2.scatter(estimated_emissions['year'], estimated_emissions['emissions'], 
                                   color='grey', s=60, alpha=0.8, marker='o', 
                                   edgecolors='darkgrey', linewidth=1, zorder=6)
            
            # Plot raw LMDI effects on primary axis
            # Activity Effect (raw only)
            ax.plot(plot_data['year'], plot_data['activity_effect'], 
                   label='Activity Effect (Raw)', linewidth=2, color='blue', alpha=0.8)
            
            # Decoupling Forces (raw only) 
            ax.plot(plot_data['year'], plot_data['decoupling_forces'], 
                   label='Decoupling Forces (Raw)', linewidth=2, color='green', alpha=0.8)
            
            # Highlight estimated portions with markers
            if 'is_estimated' in plot_data.columns:
                estimated_data = plot_data[plot_data['is_estimated'] == True]
                
                if len(estimated_data) > 0:
                    # Add markers for estimated LMDI effects
                    ax.scatter(estimated_data['year'], estimated_data['activity_effect'], 
                              color='lightblue', s=80, alpha=0.9, marker='o', 
                              edgecolors='blue', linewidth=2, zorder=6)
                    ax.scatter(estimated_data['year'], estimated_data['decoupling_forces'], 
                              color='lightgreen', s=80, alpha=0.9, marker='o',
                              edgecolors='green', linewidth=2, zorder=6)
            
            # Zero line for LMDI effects
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Peak signal markers (where decoupling > activity)
            crossover_points = plot_data[plot_data['peak_signal_raw'] == True]
            if len(crossover_points) > 0:
                # Distinguish actual vs estimated crossovers
                if 'is_estimated' in crossover_points.columns:
                    actual_crossovers = crossover_points[crossover_points['is_estimated'] != True]
                    estimated_crossovers = crossover_points[crossover_points['is_estimated'] == True]
                    
                    if len(actual_crossovers) > 0:
                        ax.scatter(actual_crossovers['year'], 
                                  actual_crossovers['decoupling_forces'],
                                  color='red', s=50, alpha=0.8, marker='o', zorder=7,
                                  label='Peak Signal')
                    
                    if len(estimated_crossovers) > 0:
                        ax.scatter(estimated_crossovers['year'], 
                                  estimated_crossovers['decoupling_forces'],
                                  color='red', s=60, alpha=0.9, marker='*', zorder=8,
                                  label='Peak Signal (Est.)')
                else:
                    ax.scatter(crossover_points['year'], 
                              crossover_points['decoupling_forces'],
                              color='red', s=50, alpha=0.8, marker='o', zorder=7)
            
            # Formatting for primary axis (LMDI effects)
            ax.set_xlabel('Year')
            ax.set_ylabel('LMDI Effect Magnitude (Mt CO₂ eq/year)', color='black')
            ax.tick_params(axis='y', labelcolor='black')
            ax.grid(True, alpha=0.3)
            
            # Formatting for secondary axis (emissions)
            ax2.set_ylabel('CO₂ Emissions (Mt CO₂)', color='grey')
            ax2.tick_params(axis='y', labelcolor='grey')
            
            # Title
            ax.set_title(f'{entity} - Raw LMDI Effects & Emissions')
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
            
            # Status text box
            status = self._determine_status(
                self.peak_signals[entity]['latest_raw_signal'],
                self.peak_signals[entity]['latest_smooth_signal'],
                self.peak_signals[entity]['latest_raw_strength'],
                self.peak_signals[entity]['latest_smooth_strength']
            )
            
            # Check if this entity has 2024 estimates
            has_2024_estimates = ('is_estimated' in plot_data.columns and 
                                 (plot_data['is_estimated'] == True).any())
            
            status_text = f'Status: {status}'
            if has_2024_estimates:
                status_text += '\n(2024: GCB estimates)'
            
            # ax.text(0.02, 0.98, status_text, 
            #        transform=ax.transAxes, fontsize=8,
            #        verticalalignment='top', 
            #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Extend x-axis slightly to show 2024 clearly
            if len(plot_data) > 0:
                x_min, x_max = ax.get_xlim()
                ax.set_xlim(x_min, max(x_max, plot_data['year'].max() + 0.5))
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def plot_lmdi_with_emission_changes(self, figsize=(18, 12)):
        """Plot raw LMDI effects with CO2 emission changes as grey bars (Fig 4)"""
        entities = list(self.peak_signals.keys())
        n_entities = len(entities)
        
        # Calculate grid dimensions
        cols = 3
        rows = (n_entities + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if n_entities == 1 else axes
        else:
            axes = axes.ravel()
        
        for i, entity in enumerate(entities):
            ax = axes[i]
            
            lmdi_data = self.peak_signals[entity]['lmdi_trends']
            
            # Calculate CO2 emission changes (year-over-year)
            # The total_change in LMDI data is already the emission change
            plot_data = lmdi_data.copy()
            
            # Plot CO2 emission changes as grey bars
            if 'total_change' in plot_data.columns:
                # Create bars for emission changes
                bars = ax.bar(plot_data['year'], plot_data['total_change'], 
                             color='lightgrey', alpha=0.6, width=0.8, 
                             label='CO₂ Change (Actual)', zorder=1)
                
                # Highlight estimated bars differently
                if 'is_estimated' in plot_data.columns:
                    estimated_mask = plot_data['is_estimated'] == True
                    if estimated_mask.any():
                        # Update color for estimated bars
                        estimated_indices = plot_data.index[estimated_mask].tolist()
                        for idx in estimated_indices:
                            if idx < len(bars):
                                bars[idx].set_color('darkgrey')
                                bars[idx].set_alpha(0.7)
                                bars[idx].set_edgecolor('black')
                                bars[idx].set_linewidth(1)
            
            # Plot raw LMDI effects as lines (on top of bars)
            # Activity Effect (raw)
            ax.plot(plot_data['year'], plot_data['activity_effect'], 
                   label='Activity Effect', linewidth=2.5, color='blue', 
                   alpha=0.9, zorder=5)
            
            # Decoupling Forces (raw) 
            ax.plot(plot_data['year'], plot_data['decoupling_forces'], 
                   label='Decoupling Forces', linewidth=2.5, color='green', 
                   alpha=0.9, zorder=5)
            
            # Highlight estimated line portions with markers
            if 'is_estimated' in plot_data.columns:
                estimated_data = plot_data[plot_data['is_estimated'] == True]
                
                if len(estimated_data) > 0:
                    # Add markers for estimated LMDI effects
                    ax.scatter(estimated_data['year'], estimated_data['activity_effect'], 
                              color='lightblue', s=100, alpha=0.9, marker='o', 
                              edgecolors='blue', linewidth=2, zorder=7,
                              label='Activity (Est.)')
                    ax.scatter(estimated_data['year'], estimated_data['decoupling_forces'], 
                              color='lightgreen', s=100, alpha=0.9, marker='o',
                              edgecolors='green', linewidth=2, zorder=7,
                              label='Decoupling (Est.)')
            
            # Zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=2)
            
            # Peak signal markers (where decoupling > activity)
            crossover_points = plot_data[plot_data['peak_signal_raw'] == True]
            if len(crossover_points) > 0:
                # Distinguish actual vs estimated crossovers
                if 'is_estimated' in crossover_points.columns:
                    actual_crossovers = crossover_points[crossover_points['is_estimated'] != True]
                    estimated_crossovers = crossover_points[crossover_points['is_estimated'] == True]
                    
                    if len(actual_crossovers) > 0:
                        ax.scatter(actual_crossovers['year'], 
                                  actual_crossovers['decoupling_forces'],
                                  color='red', s=60, alpha=0.9, marker='o', 
                                  edgecolors='darkred', linewidth=1, zorder=8,
                                  label='Peak Signal')
                    
                    if len(estimated_crossovers) > 0:
                        ax.scatter(estimated_crossovers['year'], 
                                  estimated_crossovers['decoupling_forces'],
                                  color='red', s=80, alpha=1.0, marker='*', 
                                  edgecolors='darkred', linewidth=1, zorder=9,
                                  label='Peak Signal (Est.)')
                else:
                    ax.scatter(crossover_points['year'], 
                              crossover_points['decoupling_forces'],
                              color='red', s=60, alpha=0.9, marker='o', zorder=8)
            
            # Add a subtle legend entry for estimated bars if they exist
            if 'is_estimated' in plot_data.columns and (plot_data['is_estimated'] == True).any():
                # Add invisible bar for legend
                ax.bar([], [], color='darkgrey', alpha=0.7, 
                       edgecolor='black', linewidth=1, label='CO₂ Change (Est.)')
            
            # Formatting
            ax.set_xlabel('Year')
            ax.set_ylabel('Effect Magnitude (Mt CO₂ eq/year)')
            ax.legend(fontsize=7, loc='lower left')
            ax.grid(True, alpha=0.3, zorder=0)
            
            # Title
            ax.set_title(f'{entity} - LMDI Decomposition & Emission Changes')
            
            # Status text box
            status = self._determine_status(
                self.peak_signals[entity]['latest_raw_signal'],
                self.peak_signals[entity]['latest_smooth_signal'],
                self.peak_signals[entity]['latest_raw_strength'],
                self.peak_signals[entity]['latest_smooth_strength']
            )
            
            # Check if this entity has 2024 estimates
            has_2024_estimates = ('is_estimated' in plot_data.columns and 
                                 (plot_data['is_estimated'] == True).any())
            
            status_text = f'Status: {status}'
            if has_2024_estimates:
                status_text += '\n(2024: GCB estimates)'
            
            # Add LMDI validation info
            if 'residual' in plot_data.columns:
                recent_residual = plot_data['residual'].abs().tail(3).mean()
                if recent_residual > 1:  # Threshold for significant residual
                    status_text += f'\nLMDI residual: {recent_residual:.0f}'
            
            # ax.text(0.02, 0.98, status_text, 
            #        transform=ax.transAxes, fontsize=8,
            #        verticalalignment='top', 
            #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Extend x-axis slightly to show 2024 clearly
            if len(plot_data) > 0:
                x_min, x_max = ax.get_xlim()
                ax.set_xlim(x_min, max(x_max, plot_data['year'].max() + 0.5))
            
            # Set y-axis to show full range of effects and changes
            if 'total_change' in plot_data.columns:
                all_values = pd.concat([
                    plot_data['activity_effect'], 
                    plot_data['decoupling_forces'],
                    plot_data['total_change']
                ])
                y_range = all_values.max() - all_values.min()
                y_margin = y_range * 0.1
                ax.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # At the end of your plotting function, before plt.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        # Remove any duplicate labels
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc='lower left')

        plt.tight_layout()
        plt.show()
        
        return fig
         
   
    
    def get_top_emitters_by_recent_change(self, n_countries=15):
        """Identify top countries by absolute change in emissions over last 2 years"""
        if not hasattr(self, 'merged_data') or self.merged_data.empty:
            return []
        
        # Calculate 2-year emission changes to avoid COVID distortions
        recent_changes = []
        
        for country in self.merged_data['country'].unique():
            country_data = self.merged_data[self.merged_data['country'] == country].sort_values('year')
            
            if len(country_data) >= 2:
                # Get last 2 years of data
                recent_data = country_data.iloc[-2:]
                if len(recent_data) == 2:
                    change = recent_data['co2'].iloc[1] - recent_data['co2'].iloc[0]
                    recent_changes.append({
                        'country': country,
                        'co2_change': change,
                        'abs_change': abs(change)
                    })
        
        if not recent_changes:
            return []
        
        # Sort by absolute change and take top N
        recent_changes_df = pd.DataFrame(recent_changes)
        top_changers = recent_changes_df.nlargest(n_countries, 'abs_change')
        
        return top_changers['country'].tolist()
    
    def analyse_drivers(self, figsize=(16, 10)):
        """Create enhanced driver analysis with more countries and improved bar chart"""
        
        # Get top emitters by recent change for scatter plot
        top_changers = self.get_top_emitters_by_recent_change(15)
        print(f"Top 15 countries by recent emission changes: {top_changers}")
        
        # Calculate driver data for entities with LMDI results
        entities_with_data = list(self.peak_signals.keys())
        print(f"Entities with LMDI data: {entities_with_data}")
        
        # Include all entities that have LMDI data
        # (The top_changers filter was too restrictive since most countries aren't in our target list)
        incl_entities = entities_with_data
        
        # Calculate recent average contributions (last 3 years)
        driver_data = []
        
        for entity in incl_entities:
            if entity not in self.lmdi_results:
                continue
                
            lmdi_data = self.peak_signals[entity]['lmdi_trends']
            
            if len(lmdi_data) >= 3:
                recent_data = lmdi_data.iloc[-3:]
                
                # Calculate average contributions (flip signs to show positive as beneficial)
                avg_intensity = -recent_data['intensity_effect'].mean()
                avg_carbon = -recent_data['carbon_effect'].mean()
                avg_activity = recent_data['activity_effect'].mean()
                
                driver_data.append({
                    'Entity': entity,
                    'Energy_Efficiency': avg_intensity,
                    'Decarbonization': avg_carbon,
                    'Economic_Growth': avg_activity,
                    'Total_Decoupling': avg_intensity + avg_carbon,
                    'Net_Effect': avg_intensity + avg_carbon - avg_activity
                })
        
        if not driver_data:
            print("No data available for driver analysis")
            return None, None
            
        driver_df = pd.DataFrame(driver_data)
        print(f"Driver analysis includes {len(driver_df)} entities")
        
        # For bar chart, exclude World and create stacked components
        bar_entities = driver_df[driver_df['Entity'] != 'World'].copy()
        
        # Create subplot with two charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Chart 1: Stacked bar chart with proper positive/negative stacking
        if len(bar_entities) > 0:
            x_pos = range(len(bar_entities))
            
            # Economic growth (positive, drives emissions up)
            ax1.bar(x_pos, bar_entities['Economic_Growth'], 
                   label='Economic Growth (+emissions)', color='lightcoral', alpha=0.8)
            
            # Energy efficiency (negative effect, reduces emissions)
            efficiency_values = bar_entities['Energy_Efficiency'].values
            ax1.bar(x_pos, -np.maximum(efficiency_values, 0), 
                   label='Energy Efficiency (-emissions)', color='orange', alpha=0.8)
            
            # Decarbonization (negative effect, reduces emissions) - stack below efficiency
            carbon_values = bar_entities['Decarbonization'].values
            efficiency_negative = -np.maximum(efficiency_values, 0)
            ax1.bar(x_pos, -np.maximum(carbon_values, 0), 
                   bottom=efficiency_negative,
                   label='Decarbonization (-emissions)', color='green', alpha=0.8)
            
            # Add net effect line
            net_values = bar_entities['Net_Effect'].values
            ax1.scatter(x_pos, -net_values, color='black', s=50, marker='D', 
                       label='Net Effect', zorder=5)
            
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax1.set_xlabel('Entity')
            ax1.set_ylabel('Effect Magnitude (Mt CO2 eq/year)\n(Positive = More Emissions)')
            ax1.set_title('LMDI Component Effects (Excludes World)\nUpward = Growth Pressure, Downward = Decoupling')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(bar_entities['Entity'], rotation=45, ha='right')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Chart 2: Scatter plot with all available entities
        # Fix color and size logic
        net_effects = driver_df['Net_Effect'].values
        colors = ['green' if net > 0 else 'red' for net in net_effects]
        # Scale bubble sizes more reasonably (20-200 range)
        sizes = [min(200, max(20, abs(net) * 8 + 25)) for net in net_effects]
        
        scatter = ax2.scatter(driver_df['Energy_Efficiency'], driver_df['Decarbonization'], 
                            c=colors, s=sizes, alpha=0.6, edgecolors='black')
        
        # Add entity labels with better positioning - smaller font for more countries
        for idx, row in driver_df.iterrows():
            # Use shorter names for better readability
            name = row['Entity'].replace('European Union (27)', 'EU27')
            ax2.annotate(name, 
                        (row['Energy_Efficiency'], row['Decarbonization']),
                        xytext=(3, 3), textcoords='offset points', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Energy Efficiency Contribution (Mt CO2 eq/year)')
        ax2.set_ylabel('Decarbonization Contribution (Mt CO2 eq/year)')
        ax2.set_title(f'Efficiency vs. Decarbonization Drivers\n({len(driver_df)} entities with LMDI data)')
        ax2.grid(True, alpha=0.3)
        
        # Add quadrant labels with better positioning
        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        
        ax2.text(xlim[1]*0.02, ylim[1]*0.98, 'Efficiency-led\nDecoupling', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontsize=8,
                verticalalignment='top')
        ax2.text(xlim[1]*0.98, ylim[0]*0.02, 'Worsening on\nBoth Fronts', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7), fontsize=8)
        ax2.text(xlim[0]*0.02, ylim[1]*0.98, 'Decarbonization-led\nDecoupling', ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontsize=8)
        ax2.text(xlim[0]*0.02, ylim[0]*0.02, 'Efficiency Gains\nOffset by Carbon', ha='left', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7), fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig, driver_df
    
    def generate_report(self):
        """Generate comprehensive peak detection report"""
        print("\nGenerating Peak Detection Report...")
        
        # Summary statistics
        summary_df = self.create_peak_summary()
        
        # # Create visualizations
        fig1 = self.plot_lmdi_trends_with_estimates()
        fig2, driver_df = self.analyse_drivers()
        
        # Analysis insights
        print("\nKey Findings:")
        status_counts = summary_df['Status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} entities")
        
        # Driver analysis insights
        if driver_df is not None:
            print("\nDriver Analysis (Recent 3-year averages):")
            
            # Efficiency-dominated entities
            efficiency_dominated = driver_df[driver_df['Energy_Efficiency'] > driver_df['Decarbonization'] * 1.5]
            if len(efficiency_dominated) > 0:
                print("  Efficiency-driven decoupling:")
                for _, row in efficiency_dominated.iterrows():
                    print(f"    {row['Entity']}: {row['Energy_Efficiency']:.1f} efficiency vs {row['Decarbonization']:.1f} decarbonization")
            
            # Decarbonization-dominated entities  
            carbon_dominated = driver_df[driver_df['Decarbonization'] > driver_df['Energy_Efficiency'] * 1.5]
            if len(carbon_dominated) > 0:
                print("  Decarbonization-driven decoupling:")
                for _, row in carbon_dominated.iterrows():
                    print(f"    {row['Entity']}: {row['Decarbonization']:.1f} decarbonization vs {row['Energy_Efficiency']:.1f} efficiency")
            
            # Balanced approaches
            balanced = driver_df[~driver_df.index.isin(efficiency_dominated.index) & ~driver_df.index.isin(carbon_dominated.index)]
            if len(balanced) > 0:
                print("  Balanced approaches:")
                for _, row in balanced.iterrows():
                    print(f"    {row['Entity']}: {row['Energy_Efficiency']:.1f} efficiency + {row['Decarbonization']:.1f} decarbonization")
        
        # Recent signals
        recent_raw_signals = summary_df[summary_df['Latest_Raw_Signal'] == True]
        recent_smooth_signals = summary_df[summary_df['Latest_Smooth_Signal'] == True]
        
        if len(recent_raw_signals) > 0:
            print(f"\nEntities with active raw peak signals:")
            for _, row in recent_raw_signals.iterrows():
                strength = row['Raw_Strength_2024']
                print(f"  {row['Entity']}: Raw strength = {strength}")
        
        if len(recent_smooth_signals) > 0:
            print(f"\nEntities with active smoothed peak signals:")
            for _, row in recent_smooth_signals.iterrows():
                strength = row['Smooth_Strength_2024']
                print(f"  {row['Entity']}: Smoothed strength = {strength}")
        
        return {
            'summary': summary_df,
            'driver_analysis': driver_df,
            'detailed_results': self.peak_signals,
            'lmdi_decomposition': self.lmdi_results,
            'figures': [fig1, fig2]
        }
    
    def run_complete_analysis(self, 
                              regions=None,
                              report = True):
        """Run the complete LMDI peak detection analysis with 2024 estimates"""
        print("Starting Complete LMDI Peak Detection Analysis")
        print("=" * 60)
        
        # OWID version
        # Load all datasets
        # self.load_gdp_data()
        # self.load_owid_data()
        # self.prepare_datasets()
        # self.merge_datasets()

        # Processed data version
        self.load_all_data(regions=regions)
        
        # Add 2024 CO2 estimates from GCB
        # self.add_2024_co2_estimates()
        
        # Continue with analysis
        self.prepare_lmdi_data()
        self.analyze_all_entities()
        self.detect_structural_peaks()
        
        # Generate report with estimated data visualization
        if report:
            results = self.generate_report()
            return self, results
        
        return self, None

    
    def fix_entity_name_mapping(self):
        """Apply entity name mapping to ensure consistency"""
        print("\n=== Applying Entity Name Fixes ===")
        
        # Common entity name mappings
        entity_mappings = {
            # World Bank to standard names
            'United States': 'United States',
            'European Union': 'European Union (27)',
            # Add other mappings as needed based on debug output
        }
        
        # Apply mappings to GDP data
        if hasattr(self, 'gdp_data') and self.gdp_data is not None:
            original_entities = self.gdp_data['Entity'].unique()
            self.gdp_data['Entity'] = self.gdp_data['Entity'].map(entity_mappings).fillna(self.gdp_data['Entity'])
            mapped_entities = self.gdp_data['Entity'].unique()
            
            changes = set(original_entities) - set(mapped_entities)
            if changes:
                print(f"  GDP entity names mapped: {changes}")
        
        return entity_mappings


# Usage example
if __name__ == "__main__":
    analyzer = LMDIPeakAnalyzer()
    analyzer.load_all_data()
    analyzer.prepare_lmdi_data()
    analyzer.analyze_all_entities()
    analyzer.detect_structural_peaks()
    # Generate report with estimated data visualization
    report = analyzer.generate_report()
    # analyzer.load_gdp_data()
    # analyzer.load_owid_data()
        
    #     # Prepare and merge data
    # analyzer.prepare_datasets()
    # analyzer.merge_datasets()
    # report = analyzer.run_complete_analysis()
