"""data_clean.py
Loads raw data from data/raw and processes it to produce data for analysis in data/processed
"""

import pandas as pd
import pandas_indexing as pix
from helper_functions import iso_to_name, name_to_iso

MtC_TO_MtCO2 = 3.664
IX_ORDER = ['region','region_name','variable','unit']

#–-------------------------------------------------------
# CO2 emissions -----------------------------------------
#–-------------------------------------------------------

fossil_co2 = pd.read_excel(
    'data/raw/National_Fossil_Carbon_Emissions_2024v1.0.xlsx',
    sheet_name='Territorial Emissions',
    skiprows=11,
    index_col=0
).T * MtC_TO_MtCO2

fossil_co2 = fossil_co2.pix.assign(variable='Emissions|CO2|Fossil',unit='Mt CO2')
fossil_co2.index.names = ['region_name','variable','unit']
fossil_co2 = fossil_co2.pix.assign(region=[name_to_iso(r) for r in fossil_co2.pix.project('region_name').index])
fossil_co2 = fossil_co2.reorder_levels(IX_ORDER)
fossil_co2.to_csv('data/processed/gcb_hist_co2.csv')

#–-------------------------------------------------------
# GDP -----------------------------------------
#–-------------------------------------------------------

gdp = pd.read_csv(
    'data/raw/WorldBank_GDP.csv',
    skiprows=4,
    index_col=[0,1,2,3]
)
gdp.index.names = ['region_name','region','variable','id']
gdp = gdp.droplevel(['region_name','id'])
gdp = gdp.pix.assign(
    region_name = [iso_to_name(iso) for iso in gdp.pix.project('region').index],
    variable='GDP',
    unit = 'USD2015'
)
gdp = gdp.reorder_levels(IX_ORDER)
gdp = gdp.dropna(how='all',axis=1)
gdp.columns = gdp.columns.astype(int)
gdp.to_csv('data/processed/wdi_gdp.csv')


#–-------------------------------------------------------
# Carbon Intensity --------------------------------------
#–-------------------------------------------------------

ci = (fossil_co2 / gdp.droplevel(['variable','unit'])).dropna(how='all').dropna(how='all',axis=1) * 1e9
ci = ci.pix.assign(
    variable='Carbon Intensity',
    unit = 'kg CO2 / $'
)
ci.to_csv('data/processed/carbon_intensity.csv')