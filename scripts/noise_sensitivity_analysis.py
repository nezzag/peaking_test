from peak_tester import EmissionsPeakTest
import pandas as pd
import numpy as np
from helper_functions import HiddenPrints
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from typing import Optional, Dict
from statsmodels.nonparametric.smoothers_lowess import lowess

test_peaker = EmissionsPeakTest()

if Config.sensitivity_analyses['emissions']:
    title_str = 'emissions'
    hist_data = "gcb_hist_co2.csv"
    null_hypothesis = 'zero_trend'
    print("Running sensitivity analysis on emissions data")
    test_data = [
        (2025, 37700),
        (2026, 37580), 
        (2027, 37460)]


elif Config.sensitivity_analyses['carbon_intensity']:
    title_str = "carbon_intensity"
    hist_data = "carbon_intensity_gdp.csv"
    null_hypothesis = '2pc_decline'
    print("Running sensitivity analysis on carbon intensity data with null hypothesis of 2% per year decline")   
    test_data = [
        (2025, 0.399),
        (2026, 0.391), 
        (2027, 0.383)]

test_peaker.load_historical_data(
    hist_data, region="WLD", year_range=range(1970, 2025)
)

# -------------------------------------------
# Test 1: How do different methods provide 
# different noise characterisations
# -------------------------------------------

if Config.sensitivity_analyses['method_test']:

    residuals = pd.DataFrame(columns=test_peaker.historical_data.year)
    trend = pd.DataFrame(columns=test_peaker.historical_data.year)
    autocorr = pd.DataFrame(columns = ['has_autocorrelation'])
    peaking_likelihood = pd.DataFrame(columns=['likelihood_of_peaking'])
    threshold_90_trend = pd.DataFrame(columns=['90th percentile negative trend threshold'])
  
    f, ax = plt.subplots()

    for method in [
        "lowess",
        "linear",
        "linear_w_autocorrelation",
        "broken_trend",
        "hp",
        "hamilton",
        "spline",
    ]:
        with HiddenPrints():
            test_peaker.characterize_noise(method=method, noise_type="normal")
            residuals.loc[method] = test_peaker.residuals
            trend.loc[method] = test_peaker.trend
            autocorr.loc[method] = test_peaker.autocorr_params['has_autocorr']
            
            test_peaker.create_noise_generator()
            test_peaker.set_test_data(test_data).run_complete_bootstrap_test(bootstrap_method = 'ar_bootstrap', null_hypothesis=null_hypothesis)
            peaking_likelihood.loc[method] = 1-test_peaker.bootstrap_results['p_value_one_tail']
            threshold_90_trend.loc[method] = np.percentile(test_peaker.bootstrap_results['bootstrap_slopes'], 10)
            
            sns.kdeplot(test_peaker.bootstrap_results["bootstrap_slopes"],ax=ax,label=method)
    
    ax.legend()
    ax.set_title(f'Boostrapped slopes: {title_str}: different noise methods')
    plt.savefig('./outputs/figures/bootstrap_slopes.png',dpi=300)

    print('-'*50 + '\n standard deviation in residuals:\n' + '-'*50)
    print(residuals.std(axis=1))
    print('-'*50 + '\n average absolute size of residuals:\n' + '-'*50)
    print(residuals.abs().mean(axis=1))
    print('-'*50 + '\n Presence of autocorrelation in residuals:\n' + '-'*50)
    print(autocorr)
    print('-'*50 + '\n Likelihood of peaking:\n' + '-'*50)
    print(peaking_likelihood)
    print('-'*50 + '\n 90th percentile for negative trend:\n' + '-'*50)
    print(threshold_90_trend)

# -------------------------------------------
# Test 2: How do different paramterisations provide 
# different noise characterisations
# -------------------------------------------

if Config.sensitivity_analyses['lowess_fraction_test']:
    noise_params = []
    peaking_likelihood = []

    print('-'*50 + '\n Testing how different LOWESS fractions impact on results')
    with HiddenPrints():
        for frac in np.linspace(0.05,0.45,8):
            (
                test_peaker
                .characterize_noise(method='lowess',fraction=frac)
                .create_noise_generator()
            )
            (    test_peaker
                .set_test_data([
                    (2025, 37700),
                    (2026, 37580), 
                    (2027, 37460),
                ]).run_complete_bootstrap_test(bootstrap_method='ar_bootstrap')
            )


            noise_params.append(test_peaker.autocorr_params)
            peaking_likelihood.append(test_peaker.bootstrap_results['p_value_one_tail'])

    f, ax = plt.subplots()
    ax.plot(np.linspace(0.05,0.45,8),[1-s for s in peaking_likelihood])
    ax.set_ylabel('Likelihood that emissions have peaked')
    ax.set_xlabel('LOWESS fraction')
    ax.axhline(y=0.66,color='k',ls='--')
    ax.set_title('The impact of increasing fractions on the \nlikelihood that emissions have peaked')
    plt.show()

# -------------------------------------------
# Test 3: How do different noise distributions
# in the noise_generator create different results
# -------------------------------------------


if Config.sensitivity_analyses['noise_distribution_test']:
    noise_params = []
    peaking_likelihood = pd.DataFrame(columns=['likelihood_of_peaking'])

    with HiddenPrints():
        for noise_type in ["normal", "t-dist", "empirical"]:
            (
                test_peaker
                .characterize_noise(method='lowess',noise_type=noise_type)
                .create_noise_generator()
            )
            (    test_peaker
                .set_test_data([
                    (2025, 37700),
                    (2026, 37580), 
                    (2027, 37460)
                ]).run_complete_bootstrap_test(bootstrap_method='ar_bootstrap')
            )

            noise_params.append(test_peaker.autocorr_params)
            peaking_likelihood.loc[noise_type] = 1 - test_peaker.bootstrap_results['p_value_one_tail']
            
    print(peaking_likelihood)


    # --------------------------------------------
    # Test 4: Comparing methods using AIC/BIC
    # --------------------------------------------

if Config.sensitivity_analyses['aic_bic_comparison']:

    methods_to_test = {
        'linear': [{}],
        'linear_w_autocorrelation': [{}],
        'lowess': [{'fraction': f} for f in [0.1, 0.2, 0.3, 0.4, 0.5]],
        'hp': [{}], #[{'lamb': lamb} for lamb in [100, 1600, 6400]],
        'broken_trend': [{}],  # [{'n_segments': n} for n in [2, 3, 4]],
        'hamilton': [{}],
        'spline': [{'n_knots': s} for s in [8, 10, 15, 25, len(test_peaker.historical_data.values)]],
    }
    
    def get_effective_parameters(method: str, n: int, params: Dict, trend_info: Dict, years:  Optional[np.ndarray] = None) -> int:
        """
        Determine effective number of degrees of freedom for a given method (k).
        
        Args:
            method: Name of the trend extraction method
            n: Number of observations
            years: Array of time points
            trend_info: Information returned by _calculate_residuals
            years: Optional, array of time points
            
        Returns:
            Effective number of degrees of freedom / parameters (k)
        """
        if method == 'linear':
            return 2  # slope + intercept
        
        elif method == 'linear_w_autocorrelation':
            return 3  # slope + intercept + AR(1) coefficient
        
        elif method == 'lowess':
            # Effective degrees of freedom ≈ n × frac
            frac = params.get('fraction', 0.3)

            def compute_lowess_hat_trace(years, frac):
                """
                Compute the trace of the LOWESS hat matrix.
                Args:
                    years: array of x values (time points)
                    frac: LOWESS smoothing parameter
                Returns:
                    Trace of the hat matrix (effective degrees of freedom)
                """
                n = len(years)
                # Create identity matrix - each column is a unit vector
                identity = np.eye(n)
                
                # Apply LOWESS to each column - each result gives one column of the hat matrix
                hat_diagonal = np.zeros(n)
                
                for i in range(n):
                    # Smooth the i-th unit vector
                    smoothed = lowess(identity[:, i], years, frac=frac, return_sorted=False)
                    hat_diagonal[i] = smoothed[i]
                
                # Trace = sum of diagonal elements = effective df
                trace = np.sum(hat_diagonal)
                
                return trace
            
            trace = compute_lowess_hat_trace(years, frac)
            return int(np.ceil(trace)) 
        
        elif method == 'hp':
            # HP filter complexity depends on lambda
            # Smaller lambda = more flexible = more parameters
            # This is approximate - could compute trace of smoother matrix
            lamb = params.get('lamb', 1600)
            if lamb <= 100:
                return int(n * 0.5)  # Very flexible
            elif lamb <= 1600:
                return int(n * 0.3)  # Moderate
            else:
                return int(n * 0.1)  # Very smooth
        
        elif method == 'spline':
            # Get number of knots from trend_info if available
            knots = trend_info.get('number_of_knots', None)
            k_degree = 3  # Cubic spline default
            if knots is not None:
                return knots + k_degree
            else:
                s = params.get('s', n)
        
                # Approximate effective df based on smoothing
                if s < n * 0.5:
                    return int(n * 0.7)  # Light smoothing, many parameters
                elif s < n * 1.0:
                    return int(n * 0.5)  # Moderate
                elif s < n * 2.0:
                    return int(n * 0.3)  # Heavy smoothing
                else:
                    return int(n * 0.2)  # Very smooth, few parameters 
            
        
        elif method == 'broken_trend':
            # Number of segments
            if 'number_of_breakpoints' in trend_info:   
                n_segments = trend_info['number_of_breakpoints']
                return n_segments * 2  # slope + intercept per segment
            else:
                print('used default 4 segments for broken trend k calculation')
                return 4  # Default assumption

        elif method == 'hamilton':
            # Hamilton filter uses 4-year moving average
            return 5  # Approximate as 4 parameters 
        
        else:
            # Default: assume moderate complexity
            print("using default value of k = 20% of n for method:", method)
            return int(n * 0.2)

    def print_comparison_summary(results_df: pd.DataFrame) -> None:
        """
        Args:
            results_df: DataFrame returned by compare_trend_methods()
        """
        print("\n" + "="*80)
        print("TREND METHOD COMPARISON")
        print("="*80)
        
        print("\nTop 5 methods by AIC:")
        print("-"*80)
        top5 = results_df.head()
        for idx, row in top5.iterrows():
            print(f"\n{idx+1}. {row['method']:30s} ({row['parameters']})")
            print(f"   AIC: {row['AIC']:8.2f} (Δ={row['delta_AIC']:6.2f})  |  "
                f"BIC: {row['BIC']:8.2f} (Δ={row['delta_BIC']:6.2f})")
            print(f"   R²: {row['R_squared']:6.3f}  |  σ: {row['sigma']:8.1f}  |  "
                f"k: {row['k_params']:3d}  |  Autocorr: {row['has_autocorr']}")
        
        print("\n" + "="*80)
        print("\nInterpretation guide:")
        print("  • Lower AIC/BIC = better")
        print("  • Δ < 2: essentially equivalent")
        print("  • Δ = 2-7: moderate evidence")  
        print("  • Δ > 10: strong evidence")
        print("  • 'has_autocorr' should be False for valid residuals")
        print("="*80)

        return


    results = []
    
    for method_name, param_configs in methods_to_test.items():
        for params in param_configs:
            try:
                # Calculate residuals using this method
                residuals, trend, trend_info = test_peaker._calculate_residuals(method_name, **params)
                years_to_use = residuals.index.values
                # Calculate goodness of fit metrics
                n = len(residuals)
                RSS = np.sum(residuals**2)
                sigma_squared = RSS / n
                
                # Determine effective parameters
                k = get_effective_parameters(method_name, n, params, trend_info, years_to_use)
                
                # Calculate information criteria
                # Neil: This AIC model works for a linear fit, need to look further into this
                AIC = n * np.log(sigma_squared) + 2 * k
                BIC = n * np.log(sigma_squared) + k * np.log(n)
                
                # Additional metrics
                r_squared = 1 - (RSS / np.sum((test_peaker.historical_data['emissions'].values[-n:] - 
                                            test_peaker.historical_data['emissions'].values[-n:].mean())**2))
                
                # # Test for remaining autocorrelation
                # acf_values = acf(residuals, nlags=1, fft=False)
                # has_autocorr = np.abs(acf_values[1]) > (1.96 / np.sqrt(n))
                
                # Store results
                param_str = ', '.join([f"{k}={v}" for k, v in params.items()]) if params else '-'
                
                results.append({
                    'method': method_name,
                    'parameters': param_str,
                    'n_obs': n,
                    'k_params': k,
                    'RSS': RSS,
                    'sigma': np.sqrt(sigma_squared),
                    'R_squared': r_squared,
                    'AIC': AIC,
                    'BIC': BIC,
                    'delta_AIC': 0,  # Will calculate after loop
                    'delta_BIC': 0,  # Will calculate after loop
                    # 'has_autocorr': has_autocorr,
                    # 'acf_lag1': acf_values[1],
                })
                
            except Exception as e:
                print(f"  Failed for {method_name} with {params}: {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate delta AIC/BIC (difference from best model)
    best_AIC = results_df['AIC'].min()
    best_BIC = results_df['BIC'].min()
    results_df['delta_AIC'] = results_df['AIC'] - best_AIC
    results_df['delta_BIC'] = results_df['BIC'] - best_BIC
    
    # Sort by AIC
    results_df = results_df.reset_index(drop=True)
    print('-'*50 + '\n: Ranking models via AIC\n' + '-'*50)
    print(results_df.sort_values('AIC')[['method', 'parameters', 'AIC', 'BIC', 'R_squared', 'sigma', 'k_params']])

    print('-'*50 + '\n: Ranking models via BIC\n' + '-'*50)
    print(results_df.sort_values('BIC')[['method', 'parameters', 'AIC', 'BIC', 'R_squared', 'sigma', 'k_params']])
