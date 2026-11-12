"""data_clean.py
Loads raw data from data/raw and processes it to produce data for analysis in data/processed
"""

import pandas as pd
import pandas_indexing as pix
from pandas_indexing import isin
from helper_functions import iso_to_name, name_to_iso

MtC_TO_MtCO2 = 3.664
IX_ORDER = ["region", "region_name", "variable", "unit"]

EU27 = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
        "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
        "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]

# G20 membership uses EU27 as the European bloc (so France/Germany/Italy are not
# double-counted). EU27 must be aggregated before G20 so the sum can use it.
G20 = ["AUS", "ARG", "BRA", "CAN", "CHN", "EU27", "IND", "IDN", "JPN", "MEX",
       "RUS", "SAU", "KOR", "TUR", "GBR", "USA", "ZAF"]

# G20 and G20_excl_CHN are omitted for now: primary_energy.csv is missing Turkey,
# which would make their energy-based intensities undercount. Add them back here
# (e.g. "G20": G20, "G20_excl_CHN": [m for m in G20 if m != "CHN"]) once fixed.
AGGREGATES = {
    "EU27": EU27,
}


def add_aggregate(df, name, members):
    """Append a `name` aggregate row (sum of `members`) to a single-variable,
    region-indexed frame, unless a row for `name` is already present (e.g. a
    native aggregate from the source)."""
    regions = df.index.get_level_values("region")
    if name in set(regions):
        return df
    rows = df[regions.isin(members)]
    if rows.empty:
        return df
    variable = df.index.get_level_values("variable")[0]
    unit = df.index.get_level_values("unit")[0]
    agg = rows.sum(axis=0, min_count=1).to_frame().T
    agg.index = pd.MultiIndex.from_tuples(
        [(name, name, variable, unit)], names=IX_ORDER)
    return pd.concat([df, agg])


def add_aggregates(df):
    """Add every regional aggregate in AGGREGATES (EU27, G20, G20_excl_CHN)."""
    for name, members in AGGREGATES.items():
        df = add_aggregate(df, name, members)
    return df

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

# World Bank WDI, GDP (constant 2015 US$), NY.GDP.MKTP.KD, 2026 release (through 2025)
gdp = pd.read_csv("data/raw/API_NY.GDP.MKTP.KD_DS2_en_csv_v2_64.csv", skiprows=4, index_col=[0, 1, 2, 3])
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
gdp = add_aggregates(gdp)  # WDI has no EU27/G20 aggregates; build from members
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
pe = pe.loc[isin(region=[r for r in pe.pix.project('region').index
                         if len(r) == 3 or r in AGGREGATES])]
pe = add_aggregates(pe)  # keep native EU27, add G20 / G20_excl_CHN
pe.to_csv("data/processed/primary_energy.csv")


# Intensities are produced further down, under "Intensities (per CO2 source)",
# once the GCB energy-CO2 series (coal+oil+gas) has been built, so they can be
# emitted for both the GCB and EDGAR energy-CO2 sources.

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

# GCB energy-CO2 series = coal + oil + gas. Excludes cement, flaring and other
# carbonate/process emissions, to match the scope of EDGAR energy CO2. Uses the
# same variable name as EDGAR ("Emissions|CO2|Energy") so the two are
# interchangeable downstream.
ENERGY_PARTS = [
    "Emissions|CO2|Fossil|Coal",
    "Emissions|CO2|Fossil|Oil",
    "Emissions|CO2|Fossil|Gas",
]
gcb_energy_co2 = (
    gcb26_wide.loc[gcb26_wide.index.get_level_values("variable").isin(ENERGY_PARTS)]
    .groupby(level=["region", "region_name"]).sum(min_count=1)
)
gcb_energy_co2 = gcb_energy_co2.pix.assign(
    variable="Emissions|CO2|Energy", unit="Mt CO2"
).reorder_levels(IX_ORDER)
gcb_energy_co2 = add_aggregates(gcb_energy_co2)  # flat file has no aggregates
gcb_energy_co2.to_csv("data/processed/gcb_energy_co2.csv")

# –-------------------------------------------------------
# Intensities (per CO2 source)
# –-------------------------------------------------------
# One set of carbon intensities per energy-CO2 source (GCB coal+oil+gas and
# EDGAR energy), suffixed by source. Energy intensity (energy / GDP) uses no
# emissions, so it is source-independent and written once. Ratios are aligned on
# region (ISO), so source-specific region_name labels do not need to match.

edgar_energy_co2 = pd.read_csv("data/processed/edgar_energy_co2.csv", index_col=IX_ORDER)
edgar_energy_co2.columns = edgar_energy_co2.columns.astype(int)


def _by_region(df):
    """Collapse a (region, region_name, variable, unit) frame to region-indexed
    with integer year columns."""
    out = df.droplevel([lvl for lvl in df.index.names if lvl != "region"])
    out.columns = out.columns.astype(int)
    return out


def _ratio(numerator, denominator, variable, unit):
    """numerator / denominator aligned on region (ISO) and year; region_name is
    taken from the numerator."""
    name_map = dict(zip(numerator.index.get_level_values("region"),
                        numerator.index.get_level_values("region_name")))
    out = _by_region(numerator).div(_by_region(denominator))
    out = out.dropna(how="all").dropna(how="all", axis=1)
    out.index = pd.MultiIndex.from_arrays(
        [out.index, out.index.map(name_map),
         [variable] * len(out), [unit] * len(out)],
        names=IX_ORDER,
    )
    return out


# Energy intensity of GDP (source-independent)
_ratio(pe, gdp, "Energy Intensity", "MJ / $").to_csv(
    "data/processed/energy_intensity_gdp.csv")

# Carbon intensities, one set per energy-CO2 source
energy_co2_sources = {"gcb": gcb_energy_co2, "edgar": edgar_energy_co2}
for label, co2 in energy_co2_sources.items():
    _ratio(co2, gdp, "Carbon Intensity", "g CO2 / $").to_csv(
        f"data/processed/carbon_intensity_gdp_{label}.csv")
    _ratio(co2, pe, "Carbon Intensity", "g CO2 / MJ").to_csv(
        f"data/processed/carbon_intensity_energy_{label}.csv")

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
