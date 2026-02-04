

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



### OLD DATA IMPORT VERSIONS BELOW ###

    # def _validate_historical_data(self) -> None:
    #     """Validate the loaded historical data."""
    #     if self.historical_data is None:
    #         raise ValueError("No historical data loaded")

    #     if len(self.historical_data) < 10:
    #         raise ValueError("Need at least 10 years of historical data")

    #     if self.historical_data["emissions"].isna().any():
    #         raise ValueError("Historical data contains missing values")

    #     if (self.historical_data["emissions"] < 0).any():
    #         raise ValueError("Historical emissions cannot be negative")


    #     ###
    
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


   # def prepare_lmdi_data(self):
    #     """Prepare data for LMDI decomposition, always calculating carbon intensity from scratch"""
    #     print("Preparing LMDI variables...")
        
    #     if self.merged_data.empty:
    #         print("No data available for LMDI analysis")
    #         return pd.DataFrame()
        
        # df = self.merged_data.copy()
        
        # # Create LMDI variables for each entity
        # lmdi_data = []
        
        # for entity in df['country'].unique():
        #     entity_data = df[df['country'] == entity].sort_values('year').copy()
            
        #     # Need at least 3 years for analysis
        #     if len(entity_data) < 3:
        #         continue
            
        #     # LMDI factors: CO2 = Activity × (Energy/Activity) × (CO2/Energy)
        #     entity_data['activity'] = entity_data['gdp']
        #     entity_data['energy_intensity'] = entity_data['primary_energy'] / entity_data['gdp']
            
        #     # ALWAYS calculate carbon intensity from CO2/energy - ignore any existing carbon_intensity column
        #     entity_data['carbon_intensity_final'] = entity_data['co2'] / entity_data['primary_energy']
            
        #     entity_data['emissions'] = entity_data['co2']
            
        #     # Preserve estimation flag if it exists
        #     if 'is_estimated' not in entity_data.columns:
        #         entity_data['is_estimated'] = False
            
        #     # Clean data - remove invalid values
        #     required_vars = ['activity', 'energy_intensity', 'carbon_intensity_final', 'emissions']
        #     entity_data = entity_data.replace([np.inf, -np.inf], np.nan)
        #     valid_mask = (entity_data[required_vars] > 0).all(axis=1)
        #     entity_data = entity_data[valid_mask]
            
        #     if len(entity_data) >= 3:
        #         lmdi_data.append(entity_data)
        #         actual_years = (~entity_data['is_estimated']).sum()
        #         estimated_years = entity_data['is_estimated'].sum()
        #         print(f"  Prepared {entity}: {len(entity_data)} years ({actual_years} actual, {estimated_years} estimated)")
        
        # if lmdi_data:
        #     self.lmdi_data = pd.concat(lmdi_data, ignore_index=True)
        #     total_estimated = (self.lmdi_data['is_estimated'] == True).sum()
        #     print(f"LMDI data prepared for {self.lmdi_data['country'].nunique()} entities")
        #     print(f"Total estimated data points: {total_estimated}")
        # else:
        #     self.lmdi_data = pd.DataFrame()
        #     print("Warning: No valid LMDI data prepared")
        
        # return self.lmdi_data