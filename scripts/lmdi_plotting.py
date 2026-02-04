# lmdi_plotting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def setup_plot_style():
    """Set matplotlib style once"""
    plt.style.use('seaborn-v0_8-darkgrid')  # or your preferred style
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10

def plot_single_entity(entity_name, plot_data, figsize=(10, 8)):
    """Plot a single entity - easy to edit and test"""
    data = plot_data[entity_name]['data']
    
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize, 
                                             sharex=True, 
                                             gridspec_kw={'height_ratios': [1, 2]})
    
    # Top panel: CO2 changes and carbon intensity
    bar_colors = ['darkgrey' if val < 0 else 'lightgrey' 
                  for val in data['total_change']]
    ax_top.bar(data['year'], data['total_change'], 
               color=bar_colors, alpha=0.6, width=0.8)
    
    ax_top2 = ax_top.twinx()
    ax_top2.plot(data['year'], data['carbon_intensity'],
                 label='Carbon intensity', linewidth=2, color='purple')
    ax_top2.set_ylabel('Carbon Intensity', color='purple')
    
    # Bottom panel: LMDI effects
    ax_bottom.plot(data['year'], data['activity_effect'], 
                   label='Activity', linewidth=2.5, color='deeppink')
    ax_bottom.plot(data['year'], data['intensity_effect'], 
                   label='Energy intensity (EI)', linewidth=1.5, color='lightgreen')
    ax_bottom.plot(data['year'], data['carbon_effect'], 
                   label='Carbon intensity (CI)', linewidth=1.5, color='turquoise')
    ax_bottom.plot(data['year'], -1*data['decoupling_forces'], 
                   label='Decoupling effects (CI + EI)', linewidth=2.5, color='navy')
    
    ax_bottom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_bottom.legend(loc='best')
    ax_bottom.set_xlabel('Year')
    ax_bottom.set_ylabel('Effect (Mt CO2/yr)')
    ax_top.set_title(f'{entity_name}')
    
    plt.tight_layout()
    return fig

def plot_all_entities(plot_data, hist_data, entities=None, save_path = None):
    """Plot all entities in a grid"""
    if entities is None:
        entities = list(plot_data.keys())
    
    n_entities = len(entities)
    cols = 3
    rows = (n_entities + cols - 1) // cols
    
    fig = plt.figure(figsize=(18, 6*rows))
    
    # Add overall title
    fig.suptitle('LMDI Decomposition Analysis', 
                 fontsize=16, 
                 fontweight='bold',
                 y=0.98)  # Adjust vertical position (0-1)
    
    for i, entity in enumerate(entities):
        # Your grid plotting logic here
            # Create two subplots vertically stacked for each entity
            ax_top = plt.subplot(rows * 2, cols, (i % cols) + 1 + (i // cols) * cols * 2)
            ax_bottom = plt.subplot(rows * 2, cols, (i % cols) + 1 + cols + (i // cols) * cols * 2, sharex=ax_top)
            data = plot_data[entity]['data']

            # Plot CO2 emission changes as grey bars
            if 'total_change' in data.columns:
                # Create bars with colors based on positive/negative values
                bar_colors = ['darkgrey' if val < 0 else 'lightgrey' 
                              for val in data['total_change']]
                bars = ax_top.bar(data['year'], data['total_change'], 
                             color=bar_colors, alpha=0.6, width=0.8, 
                             label='CO2 emissions change', zorder=2)
                
                # Highlight estimated bars differently
                if 'is_estimated' in data.columns:
                    estimated_mask = data['is_estimated'] == True
                    if estimated_mask.any():
                        # Update color for estimated bars
                        estimated_indices = data.index[estimated_mask].tolist()
                        for idx in estimated_indices:
                            if idx < len(bars):
                                bars[idx].set_color('darkgrey')
                                bars[idx].set_alpha(0.7)
                                bars[idx].set_edgecolor('black')
                                bars[idx].set_linewidth(1)
            
            ax_top.set_ylabel('CO2 Emissions Change (Mt CO2/yr)', fontsize=9)

            # plot carbon intensity (GDP)
            ax_top2 = ax_top.twinx()
            hist_ci = hist_data['carbon_intensity_gdp']
            ci_data = hist_ci.loc[hist_ci['region'] == entity, ['year', 'carbon_intensity_gdp']].copy()
            ax_top2.plot(ci_data['year'], ci_data['carbon_intensity_gdp'],
                        label = 'Carbon intensity (gdp)', linewidth = 2, color = 'purple', zorder=5)

            # Format the secondary axis
            ax_top2.set_ylabel('Carbon Intensity (tCO2/USD)', fontsize=9, color='purple')
            ax_top2.tick_params(axis='y', labelcolor='purple')
            
            # Ensure the line is visible above bars
            # ax_top2.set_zorder(ax_top.get_zorder() + 1)
            ax_top.patch.set_visible(False)  # Makes the bars visible under the line
            
            # Plot raw LMDI effects as lines (on top of bars)
            # Activity Effect (raw)
            ax_bottom.plot(data['year'], data['activity_effect'], 
                   label='Activity Effect (GDP)', linewidth=2.5, color='deeppink', 
                   alpha=0.9, zorder=5)
            
            # Decoupling Forces (raw) 
            ax_bottom.plot(data['year'], data['intensity_effect'], 
                   label='Energy intensity effect', linewidth=1.5, color='turquoise', 
                   alpha=0.9, zorder=5)
            
            ax_bottom.plot(data['year'], data['carbon_effect'], 
                    label='Carbon intensity effect', linewidth=1.5, color='lightgreen', 
                    alpha=0.9, zorder=5)

            ax_bottom.plot(data['year'], -1*data['decoupling_forces'], 
                    label='Decoupling effects (CI + EI)', linewidth=2.5, color='navy', 
                    alpha=0.7, zorder=5)
            
            # Highlight estimated line portions with markers
            if 'is_estimated' in data.columns:
                estimated_data = data[data['is_estimated'] == True]
                
                if len(estimated_data) > 0:
                    # Add markers for estimated LMDI effects
                    ax_bottom.scatter(estimated_data['year'], estimated_data['activity_effect'], 
                              color='lightpink', s=100, alpha=0.9, marker='o', 
                              edgecolors='deeppink', linewidth=2, zorder=7,
                              label='Activity (Est.)')
                    ax_bottom.scatter(estimated_data['year'], estimated_data['intensity_effect'], 
                              color='darkturquoise', s=100, alpha=0.9, marker='o',
                              edgecolors='turquoise', linewidth=2, zorder=7,
                              label='Energy intensity (Est.)')
                    ax_bottom.scatter(estimated_data['year'], estimated_data['carbon_effect'], 
                              color='lightgreen', s=100, alpha=0.9, marker='o',
                              edgecolors='green', linewidth=2, zorder=7,
                              label='Carbon intensity (Est.)')
                    ax_bottom.scatter(estimated_data['year'], -1*estimated_data['decoupling_forces'], 
                              color='blue', s=100, alpha=0.9, marker='o',
                              edgecolors='navy', linewidth=2, zorder=7,
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
            
            # Add a legend entry for estimated bars if they exist
            if 'is_estimated' in data.columns and (data['is_estimated'] == True).any():
                # Add invisible bar for legend
                ax_top.bar([], [], color='darkgrey', alpha=0.7, 
                       edgecolor='black', linewidth=1, label='CO₂ Change (Est.)')
            
            # Formatting
            ax_bottom.set_xlabel('Year')
            ax_bottom.set_ylabel('Effect Magnitude (Mt CO2 eq/year)')
            # ax_bottom.legend(fontsize=7, loc='lower left')
            ax_bottom.grid(True, alpha=0.3, zorder=0)
            ax_top.grid(False)
            ax_top2.grid(False)
            
            # Title
            ax_top.set_title(f'{entity}')
            
            # # Status text box
            status = plot_data[entity]['status']
            
            # Check if this entity has 2024 estimates
            has_2024_estimates = data.get('is_estimated', pd.Series([])).any()
            
            status_text = f'Status: {status}'
            if has_2024_estimates:
                status_text += '\n(2024: GCB estimates)'
        
            
            # Extend x-axis slightly to show 2024 clearly
            if len(data) > 0:
                x_min, x_max = ax_bottom.get_xlim()
                ax_bottom.set_xlim(x_min, max(x_max, data['year'].max() + 0.5))
            
            # Set y-axis to show full range of effects and changes
            if 'total_change' in data.columns:
                all_values = pd.concat([
                    data['activity_effect'], 
                    data['intensity_effect'],
                    data['carbon_effect'],
                    data['total_change']
                ])
                y_range = all_values.max() - all_values.min()
                y_margin = y_range * 0.1
                ax_bottom.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)
                ax_top.set_ylim(data['total_change'].min(), data['total_change'].max())
        
        
            # Add shading when emissions decrease AND decoupling > activity
            if 'total_change' in data.columns:
                # Create mask for conditions
                mask = (data['total_change'] < 0) & (data['decoupling_forces'].abs() > data['activity_effect'])
                
                # Get y-axis limits for shading
                y_min, y_max = ax_bottom.get_ylim()
                
                # Shade each year where conditions are met
                for idx, row in data[mask].iterrows():
                    year = row['year']
                    # Shade from year-0.5 to year+0.5 to cover the full bar width
                    ax_bottom.axvspan(year - 0.5, year + 0.5, 
                                      color='yellow', alpha=0.1, zorder=0,
                                      label='Decoupling effect > activity effect' if idx == data[mask].index[0] else "")
    
    # At the end of your plotting function, before plt.tight_layout()
    handles, labels = ax_bottom.get_legend_handles_labels()
    # Remove any duplicate labels
    by_label = dict(zip(labels, handles))
    # ax_bottom.legend(by_label.values(), by_label.keys(), fontsize=7, loc='lower left')
    fig.legend(by_label.values(), by_label.keys(),
          loc='center left',
          bbox_to_anchor=(0.87, 0.8),
          fontsize=8,
          frameon=True,
          fancybox=True)
    # Add more vertical spacing between subplot pairs
    plt.subplots_adjust(hspace=0.5, wspace=0.3, left = 0.05, right=0.82)  # Give more room for legend
    # plt.tight_layout()
    # plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig




# Usage in interactive session or notebook:
# %%
# import importlib

# importlib.reload(lmdi_analysis)
from lmdi_analysis import LMDIPeakAnalyzer
# from lmdi_plotting import setup_plot_style, plot_single_entity, plot_all_entities

# # Run analysis once
analyzer = LMDIPeakAnalyzer()

# analyzer.load_all_data(regions = ['WLD','CHN', 'IND', 'DEU', 'JPN', 'BRA', 'ZAF'])
# analyzer.run_complete_analysis(regions = ['CHN', 'DEU', 'WLD'],
#                                report = False)

# # # Get plot data
# plot_data = analyzer.get_plot_data()
# hist_data = analyzer.historical_data
# # # Now experiment with plots
# setup_plot_style()

# # # %% Try different entities
# # fig = plot_single_entity('USA', plot_data)
# # plt.show()

# # # %% Try different colors/styles - just edit plot_single_entity() and rerun
# # fig = plot_single_entity('CHN', plot_data, figsize=(12, 10))
# # plt.show()

# # # %% Plot all at once
# fig = plot_all_entities(plot_data = plot_data, hist_data = hist_data, save_path = './outputs/figures/lmdi_decomposition_plots_small.png')

# plt.show()

analyzer.load_all_data(regions = ['WLD'])
analyzer.prepare_lmdi_data()

gdp_2025 = analyzer.historical_data['gdp'].loc[(analyzer.historical_data['gdp']['year'] == 2024) &
                                               (analyzer.historical_data['gdp']['region'] == 'WLD'), 'gdp'].values[0] * 1.03  # Assume 3% growth

pe_2025 = analyzer.historical_data['primary_energy'].loc[(analyzer.historical_data['primary_energy']['year'] == 2024) &
                                        (analyzer.historical_data['primary_energy']['region'] == 'WLD'), 'primary_energy'].values[0] * 1.02  # Assume 2% growth


test_data = {
    2025: {
        'WLD': {
            'co2': 38980,  # Example CO2 emissions in Mt
            'gdp': gdp_2025,
            'primary_energy': pe_2025  # Example primary energy in Mtoe
        }
    }
}   

# analyzer.add_test_data(test_data)
analyzer.analyze_all_entities()
analyzer.detect_structural_peaks()
plot_data = analyzer.get_plot_data()
setup_plot_style()
fig = plot_single_entity('WLD', plot_data, figsize=(12, 10))
plt.show()