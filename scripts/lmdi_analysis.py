import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LMDIPeakAnalyzer:
    def __init__(self):
        """Initialize LMDI Peak Detection Analyzer using OWID data"""
        self.target_entities = [
            'China', 'United States', 'India', 'European Union (27)', 'World',
            'Indonesia', 'Brazil', 'Mexico'
        ]
        self.data = {}
        self.merged_data = None
        self.lmdi_results = {}
        self.peak_signals = {}
    
    def load_gdp_data(self):
        """Load GDP PPP data from World Bank API (2021 international dollars)"""
        print("Loading GDP data from World Bank...")
        
        # World Bank API endpoint for GDP PPP (constant 2021 international $)
        wb_countries = {
            'China': 'CN',
            'United States': 'US', 
            'India': 'IN',
            'European Union (27)': 'EU',  # World Bank has EU aggregate
            'Indonesia': 'ID',
            'Brazil': 'BR',
            'Mexico': 'MX'
        }
        
        country_codes = ';'.join(wb_countries.values()) + ';1W'  # Add World (1W)
        
        wb_url = f"https://api.worldbank.org/v2/country/{country_codes}/indicator/NY.GDP.MKTP.PP.KD"
        params = {
            'date': '2000:2024',
            'format': 'json', 
            'per_page': 1000
        }
        
        try:
            response = requests.get(wb_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    
                    # Convert to DataFrame
                    gdp_records = []
                    for record in data[1]:
                        if record['value'] is not None:
                            # Map World Bank country names to our standard names
                            country_name_map = {
                                'China': 'China',
                                'United States': 'United States',
                                'India': 'India', 
                                'European Union': 'European Union (27)',
                                'Indonesia': 'Indonesia',
                                'Brazil': 'Brazil',
                                'Mexico': 'Mexico',
                                'World': 'World'
                            }
                            
                            wb_country = record['country']['value']
                            std_country = country_name_map.get(wb_country, wb_country)
                            
                            gdp_records.append({
                                'Entity': std_country,
                                'Year': int(record['date']),
                                'GDP_PPP_2021': record['value']  # GDP PPP constant 2021 international $
                            })
                    
                    self.gdp_data = pd.DataFrame(gdp_records)
                    print(f"GDP data shape: {self.gdp_data.shape}")
                    print(f"GDP columns: {list(self.gdp_data.columns)}")
                    print(f"Countries: {sorted(self.gdp_data['Entity'].unique())}")
                    
                    # Check coverage
                    for entity in self.gdp_data['Entity'].unique():
                        entity_data = self.gdp_data[self.gdp_data['Entity'] == entity]
                        year_range = f"{entity_data['Year'].min()}-{entity_data['Year'].max()}"
                        print(f"  {entity}: {len(entity_data)} years ({year_range})")
                else:
                    print("No GDP data returned from World Bank")
                    self.gdp_data = None
            else:
                print(f"World Bank API error: {response.status_code}")
                self.gdp_data = None
                
        except Exception as e:
            print(f"Error loading World Bank GDP data: {e}")
            self.gdp_data = None
            
        return self.gdp_data
        
    def load_owid_data(self):
        """Load required datasets from OWID and return the loaded data"""
        print("Loading data from Our World in Data...")
        
        datasets = {
            'co2': "https://ourworldindata.org/grapher/annual-co2-emissions-per-country.csv?v=1&csvType=full&useColumnShortNames=true",
            'primary_energy': "https://ourworldindata.org/grapher/primary-energy-cons.csv?v=1&csvType=full&useColumnShortNames=true"
        }
        
        loaded_data = {}
        
        for name, url in datasets.items():
            try:
                print(f"Loading {name} data...")
                
                # Try simple pandas read_csv first - it often works without custom headers
                try:
                    df = pd.read_csv(url)
                except Exception as e1:
                    print(f"  Simple CSV read failed, trying with requests: {e1}")
                    # Fall back to requests + pandas approach
                    import requests
                    from io import StringIO
                    
                    headers = {'User-Agent': 'Mozilla/5.0 (compatible; research/1.0)'}
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    df = pd.read_csv(StringIO(response.text))
                
                # Validate the loaded data
                if df.empty:
                    print(f"  Warning: {name} dataset is empty")
                    setattr(self, f'{name}_data', None)
                    loaded_data[name] = None
                    continue
                    
                # Check for required columns
                required_cols = ['Entity', 'Year']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"  Warning: {name} dataset missing required columns: {missing_cols}")
                    print(f"  Available columns: {list(df.columns)}")
                
                # Store as instance attribute
                setattr(self, f'{name}_data', df)
                loaded_data[name] = df
                
                print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")
                
                # Show data coverage
                if 'Entity' in df.columns and 'Year' in df.columns:
                    entities_in_targets = df[df['Entity'].isin(self.target_entities)]['Entity'].nunique()
                    year_range = f"{df['Year'].min()}-{df['Year'].max()}"
                    total_entities = df['Entity'].nunique()
                    print(f"  Coverage: {entities_in_targets}/{len(self.target_entities)} target entities, years {year_range}")
                    print(f"  Total entities in dataset: {total_entities}")
                
            except Exception as e:
                print(f"  Error loading {name}: {e}")
                setattr(self, f'{name}_data', None)
                loaded_data[name] = None
        
        # Summary
        successful_loads = sum(1 for data in loaded_data.values() if data is not None)
        print(f"Successfully loaded {successful_loads}/{len(datasets)} datasets")
        
        return loaded_data
    
    def prepare_datasets(self):
        """Clean and standardize all datasets"""
        print("Preparing datasets...")
        
        clean_data = {}
        
        # Handle World Bank GDP data separately (different structure)
        if hasattr(self, 'gdp_data') and self.gdp_data is not None:
            gdp_clean = self.gdp_data[['Entity', 'Year', 'GDP_PPP_2021']].rename(columns={
                'Entity': 'country',
                'Year': 'year',
                'GDP_PPP_2021': 'gdp'
            })
            
            # Filter for target entities
            gdp_clean = gdp_clean[gdp_clean['country'].isin(self.target_entities)]
            gdp_clean = gdp_clean[gdp_clean['gdp'].notna() & (gdp_clean['gdp'] > 0)]
            
            clean_data['gdp'] = gdp_clean
            print(f"  GDP (World Bank): {len(gdp_clean)} rows for target entities")
            
            # Show coverage
            coverage = gdp_clean.groupby('country')['year'].agg(['min', 'max', 'count'])
            for entity in self.target_entities:
                if entity in coverage.index:
                    row = coverage.loc[entity]
                    print(f"    {entity}: {row['count']} years ({row['min']}-{row['max']})")
                else:
                    print(f"    {entity}: No data")
        
        # Handle OWID datasets 
        owid_datasets = ['co2', 'primary_energy']
        
        for name in owid_datasets:
            # Check if the attribute exists and is not None
            if hasattr(self, f'{name}_data') and getattr(self, f'{name}_data') is not None:
                df = getattr(self, f'{name}_data')
                
                # Standardize column names
                cols = df.columns.tolist()
                entity_col = 'Entity' if 'Entity' in cols else 'Country'
                year_col = 'Year' if 'Year' in cols else 'year'
                
                # Find value column (exclude Entity, Year, Code)
                value_cols = [col for col in cols if col not in [entity_col, year_col, 'Code']]
                
                if not value_cols:
                    print(f"  Warning: No value columns found in {name}")
                    continue
                    
                # Take the main value column (usually the last one)
                value_col = value_cols[0] if len(value_cols) == 1 else value_cols[-1]
                
                # Create clean dataset
                clean_df = df[[entity_col, year_col, value_col]].rename(columns={
                    entity_col: 'country',
                    year_col: 'year',
                    value_col: name
                })
                
                # Filter for target entities and valid data
                clean_df = clean_df[clean_df['country'].isin(self.target_entities)]
                clean_df = clean_df[clean_df[name].notna() & (clean_df[name] > 0)]
                
                clean_data[name] = clean_df
                print(f"  {name}: {len(clean_df)} rows for target entities")
                
                # Show data coverage by entity
                coverage = clean_df.groupby('country')['year'].agg(['min', 'max', 'count'])
                for entity in self.target_entities:
                    if entity in coverage.index:
                        row = coverage.loc[entity]
                        print(f"    {entity}: {row['count']} years ({row['min']}-{row['max']})")
                    else:
                        print(f"    {entity}: No data")
            else:
                print(f"  Skipping {name}: No data loaded")
        
        self.clean_data = clean_data
        return clean_data
    
    def merge_datasets(self):
        """Merge all datasets for LMDI analysis"""
        print("Merging datasets...")
        
        if not hasattr(self, 'clean_data') or not self.clean_data:
            print("Error: No clean_data available to merge")
            self.merged_data = pd.DataFrame()
            return self.merged_data
        
        # Start with the first available dataset
        merged = None
        
        for name, df in self.clean_data.items():
            if df is None or df.empty:
                print(f"  Skipping {name}: No data")
                continue
                
            print(f"  Merging {name}: {len(df)} rows")
            
            if merged is None:
                merged = df.copy()
                print(f"    Starting with {name}: {merged.shape}")
            else:
                before_merge = len(merged)
                merged = merged.merge(df, on=['country', 'year'], how='outer')
                print(f"    After merging {name}: {len(merged)} rows (was {before_merge})")
        
        if merged is not None and not merged.empty:
            print(f"Initial merged dataset: {merged.shape[0]} total rows")
            print(f"Columns: {list(merged.columns)}")
            print(f"Year range: {merged['year'].min()}-{merged['year'].max()}")
            
            # Check 2024 data specifically
            data_2024 = merged[merged['year'] == 2024]
            print(f"2024 data in merged: {len(data_2024)} rows")
            if len(data_2024) > 0:
                print(f"  2024 entities: {list(data_2024['country'].unique())}")
                
                # Check completeness
                for col in ['gdp', 'primary_energy', 'co2']:
                    if col in data_2024.columns:
                        complete = data_2024[col].notna().sum()
                        print(f"  2024 {col} complete: {complete}/{len(data_2024)}")
            
            # Filter for complete data - but be more flexible about CO2
            required_cols = ['gdp', 'primary_energy', 'co2']
            available_required = [col for col in required_cols if col in merged.columns]
            
            print(f"Required columns available: {available_required}")
            
            if len(available_required) >= 2:  # Changed from 3 to 2
                # For 2024, we'll add CO2 estimates later, so only require GDP + energy
                if len(available_required) == 3:
                    # We have all three columns
                    # For years before 2024, require all three
                    pre_2024_complete = merged[
                        (merged['year'] < 2024) & 
                        merged[available_required].notna().all(axis=1) & 
                        (merged[available_required] > 0).all(axis=1)
                    ]
                    
                    # For 2024, only require GDP and energy (CO2 will be added as estimates)
                    year_2024_partial = merged[
                        (merged['year'] == 2024) & 
                        merged[['gdp', 'primary_energy']].notna().all(axis=1) & 
                        (merged[['gdp', 'primary_energy']] > 0).all(axis=1)
                    ]
                    
                    complete_data = pd.concat([pre_2024_complete, year_2024_partial], ignore_index=True)
                else:
                    # We only have GDP and energy - this is fine
                    complete_data = merged[
                        merged[available_required].notna().all(axis=1) & 
                        (merged[available_required] > 0).all(axis=1)
                    ]
                
                print(f"Complete data after filtering: {complete_data.shape[0]} rows")
                print(f"Entities with complete data: {complete_data['country'].nunique()}")
                print(f"Year range in complete data: {complete_data['year'].min()}-{complete_data['year'].max()}")
                
                # Check 2024 specifically
                complete_2024 = complete_data[complete_data['year'] == 2024]
                print(f"Complete 2024 data: {len(complete_2024)} rows")
                if len(complete_2024) > 0:
                    print(f"  2024 entities in final data: {list(complete_2024['country'].unique())}")
                
                self.merged_data = complete_data
            else:
                print("Error: Not enough required datasets available")
                print(f"Available: {available_required}, Need at least: {required_cols[:2]}")
                self.merged_data = pd.DataFrame()
        else:
            print("Error: Merge operation failed - no data to merge")
            self.merged_data = pd.DataFrame()
        
        return self.merged_data

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
    
    def prepare_lmdi_data(self):
        """Prepare data for LMDI decomposition, always calculating carbon intensity from scratch"""
        print("Preparing LMDI variables...")
        
        if self.merged_data.empty:
            print("No data available for LMDI analysis")
            return pd.DataFrame()
        
        df = self.merged_data.copy()
        
        # Create LMDI variables for each entity
        lmdi_data = []
        
        for entity in df['country'].unique():
            entity_data = df[df['country'] == entity].sort_values('year').copy()
            
            # Need at least 3 years for analysis
            if len(entity_data) < 3:
                continue
            
            # LMDI factors: CO2 = Activity × (Energy/Activity) × (CO2/Energy)
            entity_data['activity'] = entity_data['gdp']
            entity_data['energy_intensity'] = entity_data['primary_energy'] / entity_data['gdp']
            
            # ALWAYS calculate carbon intensity from CO2/energy - ignore any existing carbon_intensity column
            entity_data['carbon_intensity_final'] = entity_data['co2'] / entity_data['primary_energy']
            
            entity_data['emissions'] = entity_data['co2']
            
            # Preserve estimation flag if it exists
            if 'is_estimated' not in entity_data.columns:
                entity_data['is_estimated'] = False
            
            # Clean data - remove invalid values
            required_vars = ['activity', 'energy_intensity', 'carbon_intensity_final', 'emissions']
            entity_data = entity_data.replace([np.inf, -np.inf], np.nan)
            valid_mask = (entity_data[required_vars] > 0).all(axis=1)
            entity_data = entity_data[valid_mask]
            
            if len(entity_data) >= 3:
                lmdi_data.append(entity_data)
                actual_years = (~entity_data['is_estimated']).sum()
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
    
    def calculate_lmdi_decomposition(self, entity_data):
        """Calculate LMDI decomposition for an entity, preserving estimation flags"""
        results = []
        entity_data = entity_data.sort_values('year').reset_index(drop=True)
        
        for i in range(1, len(entity_data)):
            year = entity_data.loc[i, 'year']
            
            # Get values for base year (t-1) and current year (t)
            A0, A1 = entity_data.loc[i-1, 'activity'], entity_data.loc[i, 'activity']
            I0, I1 = entity_data.loc[i-1, 'energy_intensity'], entity_data.loc[i, 'energy_intensity']
            C0, C1 = entity_data.loc[i-1, 'carbon_intensity_final'], entity_data.loc[i, 'carbon_intensity_final']
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
                'window_years': window_years
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
         
    def plot_lmdi_alldrivers_with_emission_changes(self, figsize=(18, 16)):
        """Plot raw LMDI effects - with decoupling forces separated out - with CO2 emission changes as grey bars (Fig 4)"""
        entities = list(self.peak_signals.keys())
        n_entities = len(entities)
        
        # Calculate grid dimensions
        cols = 3
        rows = (n_entities + cols - 1) // cols
        
        # fig, axes = plt.subplots(rows, cols, figsize=figsize)
        # if rows == 1:
        #     axes = [axes] if n_entities == 1 else axes
        # else:
        #     axes = axes.ravel()
        fig = plt.figure(figsize=figsize)
        
        for i, entity in enumerate(entities):
            # ax = axes[i]
            # Create two subplots vertically stacked for each entity
            ax_top = plt.subplot(rows * 2, cols, (i % cols) + 1 + (i // cols) * cols * 2)
            ax_bottom = plt.subplot(rows * 2, cols, (i % cols) + 1 + cols + (i // cols) * cols * 2, sharex=ax_top)
            
            lmdi_data = self.peak_signals[entity]['lmdi_trends']
            
            # Calculate CO2 emission changes (year-over-year)
            # The total_change in LMDI data is already the emission change
            plot_data = lmdi_data.copy()
            
            # # Plot CO2 emission changes as grey bars
            # if 'total_change' in plot_data.columns:
            #     # Create bars for emission changes
            #     bars = ax.bar(plot_data['year'], plot_data['total_change'], 
            #                  color='lightgrey', alpha=0.6, width=0.8, 
            #                  label='CO₂ Change (Actual)', zorder=1)

            # Plot CO2 emission changes as grey bars
            if 'total_change' in plot_data.columns:
                # Create bars with colors based on positive/negative values
                bar_colors = ['darkgrey' if val < 0 else 'lightgrey' 
                              for val in plot_data['total_change']]
                bars = ax_top.bar(plot_data['year'], plot_data['total_change'], 
                             color=bar_colors, alpha=0.6, width=0.8, 
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
            
            # plot carbon intensity
            ax_top2 = ax_top.twinx()
            ax_top2.plot(plot_data['year'], plot_data['carbon_intensity'],
                        label = 'Carbon intensity', linewidth = 2, color = 'purple', zorder=5)

            # Format the secondary axis
            ax_top2.set_ylabel('Carbon Intensity\n(tCO₂/unit)', fontsize=9, color='purple')
            ax_top2.tick_params(axis='y', labelcolor='purple')
            
            # Ensure the line is visible above bars
            ax_top2.set_zorder(ax_top.get_zorder() + 1)
            ax_top.patch.set_visible(False)  # Makes the bars visible under the line
            
            # Plot raw LMDI effects as lines (on top of bars)
            # Activity Effect (raw)
            ax_bottom.plot(plot_data['year'], plot_data['activity_effect'], 
                   label='Activity Effect', linewidth=2.5, color='deeppink', 
                   alpha=0.9, zorder=5)
            
            # Decoupling Forces (raw) 
            ax_bottom.plot(plot_data['year'], plot_data['intensity_effect'], 
                   label='Energy intensity effect', linewidth=1.5, color='orange', 
                   alpha=0.9, zorder=5)
            
            ax_bottom.plot(plot_data['year'], plot_data['carbon_effect'], 
                    label='Carbon intensity effect', linewidth=1.5, color='darkturquoise', 
                    alpha=0.9, zorder=5)

            ax_bottom.plot(plot_data['year'], -1*plot_data['decoupling_forces'], 
                    label='Decoupling forces', linewidth=2.5, color='green', 
                    alpha=0.7, zorder=5)
            
            # Highlight estimated line portions with markers
            if 'is_estimated' in plot_data.columns:
                estimated_data = plot_data[plot_data['is_estimated'] == True]
                
                if len(estimated_data) > 0:
                    # Add markers for estimated LMDI effects
                    ax_bottom.scatter(estimated_data['year'], estimated_data['activity_effect'], 
                              color='lightpink', s=100, alpha=0.9, marker='o', 
                              edgecolors='deeppink', linewidth=2, zorder=7,
                              label='Activity (Est.)')
                    ax_bottom.scatter(estimated_data['year'], estimated_data['intensity_effect'], 
                              color='gold', s=100, alpha=0.9, marker='o',
                              edgecolors='orange', linewidth=2, zorder=7,
                              label='Energy intensity (Est.)')
                    ax_bottom.scatter(estimated_data['year'], estimated_data['carbon_effect'], 
                              color='paleturquoise', s=100, alpha=0.9, marker='o',
                              edgecolors='darkturquoise', linewidth=2, zorder=7,
                              label='Carbon intensity (Est.)')
                    ax_bottom.scatter(estimated_data['year'], -1*estimated_data['decoupling_forces'], 
                              color='lightgreen', s=100, alpha=0.9, marker='o',
                              edgecolors='green', linewidth=2, zorder=7,
                              label='Decoupling forces (Est.)')
            
            
            # Zero line
            ax_bottom.axhline(y=0, color='black', linestyle='--', alpha=0.5, zorder=2)
            
            # # Peak signal markers (where decoupling > activity)
            # crossover_points = plot_data[plot_data['peak_signal_raw'] == True]
            # if len(crossover_points) > 0:
            #     # Distinguish actual vs estimated crossovers
            #     if 'is_estimated' in crossover_points.columns:
            #         actual_crossovers = crossover_points[crossover_points['is_estimated'] != True]
            #         estimated_crossovers = crossover_points[crossover_points['is_estimated'] == True]
                    
            #         if len(actual_crossovers) > 0:
            #             ax.scatter(actual_crossovers['year'], 
            #                       actual_crossovers['decoupling_forces'],
            #                       color='red', s=60, alpha=0.9, marker='o', 
            #                       edgecolors='darkred', linewidth=1, zorder=8,
            #                       label='Peak Signal')
                    
            #         if len(estimated_crossovers) > 0:
            #             ax.scatter(estimated_crossovers['year'], 
            #                       estimated_crossovers['decoupling_forces'],
            #                       color='red', s=80, alpha=1.0, marker='*', 
            #                       edgecolors='darkred', linewidth=1, zorder=9,
            #                       label='Peak Signal (Est.)')
            #     else:
            #         ax.scatter(crossover_points['year'], 
            #                   crossover_points['decoupling_forces'],
            #                   color='red', s=60, alpha=0.9, marker='o', zorder=8)
            
            # Add a subtle legend entry for estimated bars if they exist
            if 'is_estimated' in plot_data.columns and (plot_data['is_estimated'] == True).any():
                # Add invisible bar for legend
                ax_top.bar([], [], color='darkgrey', alpha=0.7, 
                       edgecolor='black', linewidth=1, label='CO₂ Change (Est.)')
            
            # Formatting
            ax_bottom.set_xlabel('Year')
            ax_bottom.set_ylabel('Effect Magnitude (Mt CO₂ eq/year)')
            # ax_bottom.legend(fontsize=7, loc='lower left')
            ax_bottom.grid(True, alpha=0.3, zorder=0)
            
            # Title
            ax_top.set_title(f'{entity} - LMDI Decomposition & Emission Changes')
            
            # # Status text box
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
                x_min, x_max = ax_bottom.get_xlim()
                ax_bottom.set_xlim(x_min, max(x_max, plot_data['year'].max() + 0.5))
            
            # Set y-axis to show full range of effects and changes
            if 'total_change' in plot_data.columns:
                all_values = pd.concat([
                    plot_data['activity_effect'], 
                    plot_data['intensity_effect'],
                    plot_data['carbon_effect'],
                    plot_data['total_change']
                ])
                y_range = all_values.max() - all_values.min()
                y_margin = y_range * 0.1
                ax_bottom.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)
                ax_top.set_ylim(plot_data['total_change'].min(), plot_data['total_change'].max())
        
        
            # Add shading when emissions decrease AND decoupling > activity
            if 'total_change' in plot_data.columns:
                # Create mask for conditions
                mask = (plot_data['total_change'] < 0) & (plot_data['decoupling_forces'].abs() > plot_data['activity_effect'])
                
                # Get y-axis limits for shading
                y_min, y_max = ax_bottom.get_ylim()
                
                # Shade each year where conditions are met
                for idx, row in plot_data[mask].iterrows():
                    year = row['year']
                    # Shade from year-0.5 to year+0.5 to cover the full bar width
                    ax_bottom.axvspan(year - 0.5, year + 0.5, 
                                      color='lightgreen', alpha=0.2, zorder=0,
                                      label='Successful decoupling' if idx == plot_data[mask].index[0] else "")
    
        # At the end of your plotting function, before plt.tight_layout()
        handles, labels = ax_bottom.get_legend_handles_labels()
        # Remove any duplicate labels
        by_label = dict(zip(labels, handles))
        # ax_bottom.legend(by_label.values(), by_label.keys(), fontsize=7, loc='lower left')
        fig.legend(by_label.values(), by_label.keys(),
          loc='center left',
          bbox_to_anchor=(1.02, 0.5),
          fontsize=8,
          frameon=True)
        # Add more vertical spacing between subplot pairs
        plt.subplots_adjust(hspace=0.5, wspace=0.3, right = 0.85)  # Increase hspace for vertical spacing
        # plt.tight_layout()
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
    
    def plot_driver_analysis(self, figsize=(16, 10)):
        """Create enhanced driver analysis with more countries and improved bar chart"""
        
        # Get top emitters by recent change for scatter plot
        top_changers = self.get_top_emitters_by_recent_change(15)
        print(f"Top 15 countries by recent emission changes: {top_changers}")
        
        # Calculate driver data for entities with LMDI results
        entities_with_data = list(self.peak_signals.keys())
        print(f"Entities with LMDI data: {entities_with_data}")
        
        # For scatter plot, include all entities that have LMDI data
        # (The top_changers filter was too restrictive since most countries aren't in our target list)
        scatter_entities = entities_with_data
        
        # Calculate recent average contributions (last 3 years)
        driver_data = []
        
        for entity in scatter_entities:
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
        
        # Create visualizations
        fig1 = self.plot_lmdi_trends_with_estimates()
        fig2, driver_df = self.plot_driver_analysis()
        
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
    
    def run_complete_analysis(self):
        """Run the complete LMDI peak detection analysis with 2024 estimates"""
        print("Starting Complete LMDI Peak Detection Analysis")
        print("=" * 60)
        
        # Load all datasets
        self.load_gdp_data()
        self.load_owid_data()
        
        # Prepare and merge data
        self.prepare_datasets()
        self.merge_datasets()
        
        # Add 2024 CO2 estimates from GCB
        self.add_2024_co2_estimates()
        
        # Continue with analysis
        self.prepare_lmdi_data()
        self.analyze_all_entities()
        self.detect_structural_peaks()
        
        # Generate report with estimated data visualization
        report = self.generate_report()
        
        return report

    
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
    report = analyzer.run_complete_analysis()