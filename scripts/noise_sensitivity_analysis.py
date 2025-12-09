from peak_tester import EmissionsPeakTest
import pandas as pd
import numpy as np
from helper_functions import HiddenPrints
import matplotlib.pyplot as plt
from config import Config

global_co2_peaker = EmissionsPeakTest()

if Config.sensitivity_analyses['emissions']:
    hist_data = "gcb_hist_co2.csv"
    null_hypothesis = 'zero_trend'
    print("Running sensitivity analysis on emissions data")
    test_data = [
        (2025, 37700),
        (2026, 37580), 
        (2027, 37460)]


elif Config.sensitivity_analyses['carbon_intensity']:
    hist_data = "carbon_intensity_gdp.csv"
    null_hypothesis = '2pc_decline'
    print("Running sensitivity analysis on carbon intensity data with null hypothesis of 2% per year decline")   
    test_data = [
        (2025, 0.399),
        (2026, 0.391), 
        (2027, 0.383)]

global_co2_peaker.load_historical_data(
    hist_data, region="WLD", year_range=range(1970, 2025)
)

# -------------------------------------------
# Test 1: How do different methods provide 
# different noise characterisations
# -------------------------------------------

if Config.sensitivity_analyses['method_test']:

    residuals = pd.DataFrame(columns=global_co2_peaker.historical_data.year)
    trend = pd.DataFrame(columns=global_co2_peaker.historical_data.year)
    autocorr = pd.DataFrame(columns = ['has_autocorrelation'])
    peaking_likelihood = pd.DataFrame(columns=['likelihood_of_peaking'])
    threshold_90_trend = pd.DataFrame(columns=['90th percentile negative trend threshold'])

    for method in [
        "loess",
        "linear",
        "linear_w_autocorrelation",
        "broken_trend",
        "hp",
        "hamilton",
        "spline",
    ]:
        with HiddenPrints():
            global_co2_peaker.characterize_noise(method=method, noise_type="normal")
            residuals.loc[method] = global_co2_peaker.residuals
            trend.loc[method] = global_co2_peaker.trend
            autocorr.loc[method] = global_co2_peaker.autocorr_params['has_autocorr']
            global_co2_peaker.create_noise_generator()
            global_co2_peaker.set_test_data(test_data).run_complete_bootstrap_test(bootstrap_method = 'white_noise_bootstrap', null_hypothesis=null_hypothesis)
            # ]).run_complete_bootstrap_test(bootstrap_method = 'white_noise_bootstrap' if not global_co2_peaker.autocorr_params['has_autocorr'] else 'ar_bootstrap')
            peaking_likelihood.loc[method] = 1-global_co2_peaker.bootstrap_results['p_value_one_tail']
            threshold_90_trend.loc[method] = np.percentile(global_co2_peaker.bootstrap_results['bootstrap_slopes'], 10)


    print('-'*50 + '\n standard deviation in residuals: ')
    print(residuals.std(axis=1))
    print('-'*50 + '\n average absolute size of residuals: ')
    print(residuals.abs().mean(axis=1))
    print('-'*50 + '\n Presence of autocorrelation in residuals: ')
    print(autocorr)
    print('-'*50 + '\n 90th percentile for negative trend: ')
    print(threshold_90_trend)

# -------------------------------------------
# Test 2: How do different paramterisations provide 
# different noise characterisations
# -------------------------------------------

if Config.sensitivity_analyses['loess_fraction_test']:
    noise_params = []
    peaking_likelihood = []

    print('-'*50 + '\n Testing how different LOESS fractions impact on results')
    with HiddenPrints():
        for frac in np.linspace(0.05,0.45,8):
            (
                global_co2_peaker
                .characterize_noise(method='loess',fraction=frac)
                .create_noise_generator()
            )
            (    global_co2_peaker
                .set_test_data([
                    (2025, 37700),
                    (2026, 37580), 
                    (2027, 37460),
                ]).run_complete_bootstrap_test(bootstrap_method='ar_bootstrap')
            )


            noise_params.append(global_co2_peaker.autocorr_params)
            peaking_likelihood.append(global_co2_peaker.bootstrap_results['p_value_one_tail'])

    f, ax = plt.subplots()
    ax.plot(np.linspace(0.05,0.45,8),[1-s for s in peaking_likelihood])
    ax.set_ylabel('Likelihood that emissions have peaked')
    ax.set_xlabel('LOESS fraction')
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
                global_co2_peaker
                .characterize_noise(method='loess',noise_type=noise_type)
                .create_noise_generator()
            )
            (    global_co2_peaker
                .set_test_data([
                    (2025, 37700),
                    (2026, 37580), 
                    (2027, 37460)
                ]).run_complete_bootstrap_test(bootstrap_method='ar_bootstrap')
            )

            noise_params.append(global_co2_peaker.autocorr_params)
            peaking_likelihood.loc[noise_type] = 1 - global_co2_peaker.bootstrap_results['p_value_one_tail']
            
    print(peaking_likelihood)
