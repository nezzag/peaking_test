from peak_tester import EmissionsPeakTest
import pandas as pd
import numpy as np

global_co2_peaker = EmissionsPeakTest()
global_co2_peaker.load_historical_data(
    'gcb_hist_co2.csv', region='WLD', year_range = range(1970,2025))

residuals = pd.DataFrame(columns=global_co2_peaker.historical_data.year)
trend = pd.DataFrame(columns=global_co2_peaker.historical_data.year)

for method in ['loess','linear','linear_w_autocorrelation','hp','hamilton','spline']:
    global_co2_peaker.characterize_noise(method=method,noise_type = 't-dist')
    residuals.loc[method] = global_co2_peaker.residuals
    trend.loc[method] = global_co2_peaker.trend


print(residuals.abs().mean(axis=1))