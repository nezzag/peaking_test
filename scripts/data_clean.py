"""data_clean.py
Loads raw data from data/raw and processes it to produce data for analysis in data/processed
"""

import pandas as pd
import pandas_indexing as pix
from pandas_indexing import isin
from helper_functions import iso_to_name, name_to_iso

MtC_TO_MtCO2 = 3.664
IX_ORDER = ["region", "region_name", "variable", "unit"]

# –-------------------------------------------------------
# CO2 emissions
# –-------------------------------------------------------

fossil_co2 = (
    pd.read_excel(
        "data/raw/National_Fossil_Carbon_Emissions_2025_v0.3.xlsx",
        sheet_name="Territorial Emissions",
        skiprows=11,
        index_col=0,
    ).T
    * MtC_TO_MtCO2
)

fossil_co2 = fossil_co2.pix.assign(variable="Emissions|CO2|Fossil", unit="Mt CO2")
fossil_co2.index.names = ["region_name", "variable", "unit"]
fossil_co2 = fossil_co2.pix.assign(
    region=[name_to_iso(r) for r in fossil_co2.pix.project("region_name").index]
)
fossil_co2 = fossil_co2.reorder_levels(IX_ORDER)
fossil_co2.to_csv("data/processed/gcb_hist_co2.csv")

# –-------------------------------------------------------
# GDP
# –-------------------------------------------------------

gdp = pd.read_csv("data/raw/WorldBank_GDP.csv", skiprows=4, index_col=[0, 1, 2, 3])
gdp.index.names = ["region_name", "region", "variable", "id"]
gdp = gdp.droplevel(["region_name", "id"])

#Standardise to Fossil CO2 regions and names
gdp = gdp.loc[isin(region=fossil_co2.pix.project('region').index)]
gdp = gdp.pix.assign(
    region_name=(
        gdp.pix.project('region').index
        .map(dict(fossil_co2.pix.project(['region','region_name']).index))),
    variable="GDP",
    unit="trillion USD2015",
)/1e12

gdp = gdp.reorder_levels(IX_ORDER)
gdp = gdp.dropna(how="all", axis=1)
gdp.columns = gdp.columns.astype(int)
gdp.to_csv("data/processed/wdi_gdp.csv")


# –-------------------------------------------------------
# Primary Energy
# –-------------------------------------------------------

pe = pd.read_excel("data/raw/EI_primaryenergy.xlsx",skiprows=2,index_col=0,skipfooter=13,usecols=range(61)).dropna(how='all')
pe.index.names = ['region_name']
pe.rename(index=lambda s:s.replace('Total World','World'),inplace=True)

pe = pe.pix.assign(
    region=[name_to_iso(r) for r in pe.pix.project("region_name").index],
    variable='Primary Energy',
    unit = 'EJ / yr'
)
#Standardise to Fossil CO2 regions and names
pe = pe.loc[isin(region=fossil_co2.pix.project('region').index)]
pe = pe.pix.assign(
    region_name = (
        pe.pix.project('region').index
        .map(dict(fossil_co2.pix.project(['region','region_name']).index))),
)
pe = pe.reorder_levels(IX_ORDER)
pe.columns = pe.columns.astype(int)
pe = pe.loc[isin(region = list(filter(lambda s:len(s)==3, pe.pix.project('region').index)))]
pe.to_csv("data/processed/primary_energy.csv")


# –-------------------------------------------------------
# Intensities
# –-------------------------------------------------------

ci_gdp = (fossil_co2 / gdp.droplevel(["variable", "unit"])).dropna(how="all").dropna(
    how="all", axis=1
)
ci_gdp = ci_gdp.pix.assign(variable="Carbon Intensity", unit="g CO2 / $")
ci_gdp.to_csv("data/processed/carbon_intensity_gdp.csv")


ei_gdp = (pe / gdp.droplevel(["variable", "unit"])).dropna(how="all").dropna(
    how="all", axis=1
)
ei_gdp = ei_gdp.pix.assign(variable="Energy Intensity", unit="MJ / $")
ei_gdp.to_csv("data/processed/energy_intensity_gdp.csv")

ci_energy = (fossil_co2 / pe.droplevel(["variable", "unit"])).dropna(how="all").dropna(
    how="all", axis=1
)
ci_energy = ci_energy.pix.assign(variable="Carbon Intensity", unit="g CO2 / MJ")
ci_energy.to_csv("data/processed/carbon_intensity_energy.csv")

# –-------------------------------------------------------
# GCB 2026 fossil CO2 (new flat-format release)
# Added alongside gcb_hist_co2.csv, which stays based on the 2025 Excel release.
# Source: Global Carbon Budget 2026 (2026v11), Zenodo DOI 10.5281/zenodo.23667912
# Flat CSV is already in Mt CO2, one row per country-year, 1750-2025.
# –-------------------------------------------------------

gcb26 = pd.read_csv("data/raw/GCB2026v11_MtCO2_flat.csv")

# Keep only standard ISO3 country codes (drops entities such as "Kuwaiti Oil
# Fires", "Ryukyu Islands" that have no ISO code).
iso = gcb26["ISO 3166-1 alpha-3"]
gcb26 = gcb26[iso.notna() & (iso.str.len() == 3)].copy()
gcb26 = gcb26.rename(columns={"ISO 3166-1 alpha-3": "region", "Country": "region_name"})

# Map source columns -> (variable, unit)
GCB26_COMPONENTS = {
    "Total":      ("Emissions|CO2|Fossil",         "Mt CO2"),
    "Coal":       ("Emissions|CO2|Fossil|Coal",    "Mt CO2"),
    "Oil":        ("Emissions|CO2|Fossil|Oil",     "Mt CO2"),
    "Gas":        ("Emissions|CO2|Fossil|Gas",     "Mt CO2"),
    "Cement":     ("Emissions|CO2|Fossil|Cement",  "Mt CO2"),
    "Flaring":    ("Emissions|CO2|Fossil|Flaring", "Mt CO2"),
    "Other":      ("Emissions|CO2|Fossil|Other",   "Mt CO2"),
    "Per Capita": ("Emissions|CO2|Fossil|Per Capita", "t CO2/capita"),
}

gcb26_long = gcb26.melt(
    id_vars=["region", "region_name", "Year"],
    value_vars=list(GCB26_COMPONENTS),
    var_name="component",
    value_name="value",
)
gcb26_long["variable"] = gcb26_long["component"].map({k: v[0] for k, v in GCB26_COMPONENTS.items()})
gcb26_long["unit"] = gcb26_long["component"].map({k: v[1] for k, v in GCB26_COMPONENTS.items()})

gcb26_wide = gcb26_long.pivot_table(
    index=IX_ORDER,
    columns="Year",
    values="value",
    aggfunc="first",
)
gcb26_wide.columns = [int(c) for c in gcb26_wide.columns]
gcb26_wide = gcb26_wide.sort_index()
gcb26_wide.to_csv("data/processed/gcb_hist_co2_2026v11.csv")

# –-------------------------------------------------------
# LULUCF Data Hub (EU Forest Observatory), version 4.1.0
# Source: Zenodo DOI 10.5281/zenodo.23559424 (NGHGI 2026 release)
# Three long-format files -> standard wide format. Index gains a 5th level
# "source" (the inventory vintage), which is constant per country.
#   variable = Emissions|<Gas>|LULUCF            (Category == "LULUCF")
#            = Emissions|<Gas>|LULUCF|<Category>  (sub-sectors)
# Covers 1990-2025, incl. WRD (World) and EU27 aggregates.
# –-------------------------------------------------------

LULUCF_IX_ORDER = ["region", "region_name", "variable", "unit", "source"]


def load_lulucf_hub(path):
    """Convert a LULUCF Data Hub timeseries CSV to standard wide format."""
    df = pd.read_csv(path)
    df = df.rename(columns={"ISO3": "region", "Source": "source"})
    # The hub labels World as "WRD"; use "WLD" to match the rest of the pipeline.
    df["region"] = df["region"].replace({"WRD": "WLD"})
    # Aggregates (WLD, EU27) have no inventory vintage; fill so pivot_table does
    # not drop them (NaN index levels are dropped on grouping).
    df["source"] = df["source"].fillna("aggregate")
    df["region_name"] = df["region"].map(iso_to_name)
    df["variable"] = [
        f"Emissions|{g}|LULUCF" if c == "LULUCF" else f"Emissions|{g}|LULUCF|{c}"
        for g, c in zip(df["Gas"], df["Category"])
    ]
    df = df.rename(columns={"Units": "unit"})
    wide = df.pivot_table(
        index=LULUCF_IX_ORDER,
        columns="Year",
        values="CFluxes_yr",
        aggfunc="first",
    )
    wide.columns = [int(c) for c in wide.columns]
    return wide.sort_index()


LULUCF_HUB_FILES = {
    "data/raw/timeseries_GCB_original_4.1.0.csv":
        "data/processed/lulucf_hub_gcb_original_4.1.0.csv",
    "data/raw/timeseries_GCB_translated_4.1.0.csv":
        "data/processed/lulucf_hub_gcb_translated_4.1.0.csv",
    "data/raw/timeseries_NGHGI_gap-filled_4.1.0.csv":
        "data/processed/lulucf_hub_nghgi_gapfilled_4.1.0.csv",
}

for in_path, out_path in LULUCF_HUB_FILES.items():
    load_lulucf_hub(in_path).to_csv(out_path)
