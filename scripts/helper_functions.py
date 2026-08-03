"""Module for supporting helper functions in the analysis"""

import pycountry
import os
import sys
import pandas as pd

class HiddenPrints:
    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stdout = self._original_stdout


# Converting ISO Alpha-3 to ISO international names
def iso_to_name(iso):
    """Converts from 3-character ISO to country"""
    if iso == "WLD":
        return "World"
    country = pycountry.countries.get(alpha_3=iso)
    if country is None:
        return iso
    else:
        return country.name


NAME_REPLACE = {
    "Aruba, Kingdom of the Netherlands": "ABW",
    "Advanced Economies": "ADV",
    "Andorra, Principality of": "AND",
    "Armenia, Republic of": "ARM",
    "ASEAN-5": "ASEAN5",
    "Afghanistan, Islamic Republic of": "AFG",
    "Azerbaijan, Republic of": "AZE",
    "Bahrain, Kingdom of": "BHR",
    "Bahamas, The": "BHS",
    "Belarus, Republic of": "BLR",
    "Bolivia": "BOL",
    "China, People's Republic of": "CHN",
    "Comoros, Union of the": "COM",
    "Congo, Republic of": "COD",
    "Democratic Republic of the Congo": "COD",
    "Congo, Democratic Republic of the": "COD",
    "Czech Republic": "CZE",
    "Germany": "DEU",
    "Emerging and Developing Asia": "EDA",
    "Emerging and Developing Europe": "EDE",
    "Emerging Market and Developing Economies (EMDE)": "EMDEs",
    "Egypt, Arab Republic of": "EGY",
    "Equatorial Guinea, Republic of": "GNQ",
    "Euro Area (EA)": "EA19",
    "Eswatini, Kingdom of": "SWZ",
    "Estonia, Republic of": "EST",
    "Ethiopia, The Federal Democratic Republic of": "ETH",
    "European Union (EU)": "EU27",
    "Fiji, Republic of": "FJI",
    "Gambia, The": "GMB",
    "Hong Kong Special Administrative Region, People's Republic of China": "HKG",
    "Croatia, Republic of": "HRV",
    "Iran": "IRN",
    "Iran (Islamic Republic of)": "IRN",
    "Kazakhstan, Republic of": "KAZ",
    "Kyrgyz Republic": "KGZ",
    "Kosovo, Republic of": "XKX",
    "Laos": "PDR",
    "Latin America and the Caribbean (LAC)": "LAC",
    "Latvia, Republic of": "LVA",
    "Lesotho, Kingdom of": "LSO",
    "Lithuania, Republic of": "LTU",
    "Madagascar, Republic of": "MDG",
    "Macao Special Administrative Region, People's Republic of China": "MAC",
    "Mauritania, Islamic Republic of": "MRT",
    "Marshall Islands, Republic of the": "MHL",
    "Middle East and Central Asia": "MECA",
    "Moldova": "MDA",
    "Republic of Moldova": "MDA",
    "Mozambique, Republic of": "MOZ",
    "North Macedonia, Republic of": "MKD",
    "Netherlands, The": "NLD",
    "Netherlands (Kingdom of the)": "NLD",
    "Nauru, Republic of": "NRU",
    "Other Advanced Economies (Advanced Economies excluding G7 and Euro Area countries)": "OAE",
    "Palau, Republic of": "PLW",
    "Poland, Republic of": "POL",
    "North Korea": "PRK",
    "Russia": "RUS",
    "San Marino, Republic of": "SMR",
    "St. Kitts and Nevis": "KNA",
    "Slovenia, Republic of": "SVN",
    "Slovak Republic": "SVK",
    "South Sudan, Republic of": "SSD",
    "São Tomé and Príncipe, Democratic Republic of": "STP",
    "Sub-Saharan Africa (SSA)": "SSA",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Taiwan Province of China": "TWN",
    "Tajikistan, Republic of": "TJK",
    "Tanzania": "TZA",
    "Trinidad & Tobago": "TTO",
    "United Republic of Tanzania": "TZA",
    "Timor-Leste, Democratic Republic of": "TLS",
    "Türkiye, Republic of": "TUR",
    "US":"USA",
    "Uzbekistan, Republic of": "UZB",
    "Vietnam": "VNM",
    "Venezuela": "VEN",
    "Venezuela, República Bolivariana de": "VEN",
    "St. Vincent and the Grenadines": "VCT",
    "St. Lucia": "LCA",
    "West Bank and Gaza": "PSE",
    "World": "WLD",
    "Yemen, Republic of": "YEM",
    "South Korea": "KOR",
    "Republic of Korea": "KOR",
}


def name_to_iso(name):
    """Converts from country name to ISO"""
    if name in NAME_REPLACE:
        return NAME_REPLACE[name]
    elif pycountry.countries.get(name=name) is None:
        return name
    else:
        return pycountry.countries.get(name=name).alpha_3


def segment_by_peaks(data, value_col="emissions", year_col="year"):
    """
    Return the index values (e.g. years) where a new all-time peak is reached,
    marking the start of each fresh segment.

    The first index value is always included. Each subsequent value that
    exceeds every previous value is also included.

    Args:
        data: pd.Series (values, with any index) or pd.DataFrame with
              year as index or as a column (year_col) and emissions as value_col.
        value_col: column name for the values when data is a DataFrame.
        year_col: column name for years when data is a DataFrame with year as a column.

    Returns:
        List of index values (e.g. years) where a new segment starts.
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if year_col in data.columns:
            data = data.sort_values(year_col).set_index(year_col)[value_col]
        else:
            data = data.sort_index()[value_col]

    segment_starts = [data.index[0]]
    running_max = data.iloc[0]

    for i in range(1, len(data)):
        val = data.iloc[i]
        if val > running_max:
            running_max = val
            segment_starts.append(data.index[i])

    return segment_starts

def find_peak_in(start_yr, end_yr, hist_data):
    """Return the year of maximum emissions between start_yr and end_yr."""
    if isinstance(hist_data, pd.DataFrame) and 'year' in hist_data.columns:
        subset = hist_data[(hist_data['year'] >= start_yr) & (hist_data['year'] <= end_yr)]
        if subset.empty:
            return None
        return subset.loc[subset['emissions'].idxmax(), 'year']
    else:
        subset = hist_data.loc[start_yr:end_yr, 'emissions']
        if subset.empty:
            return None
        return subset.idxmax()

def combine_countries(countries, data, countries_name=None):

    """
    Combines (sums) data for multiple countries into a single entry in the data CSV.
    Args:
        countries: list of ISO country codes to combine
        data: filename of the data CSV (relative to ../data/processed/) or df
        countries_name: optional name for the combined entry; if None, will be a '+'-joined string of the country codes

        Returns:
            DataFrame with the combined data for the specified countries, indexed by variable and unit.
    """

    countries_name = countries_name or '+'.join(countries)

    if isinstance(data, str):
        data_path = f"../data/processed/{data}"
        try:
            data = pd.read_csv(data_path, index_col=[0, 1, 2, 3])
        except Exception as e:
            raise ValueError(f"Could not load data from {data_path}: {e}")

    elif isinstance(data, pd.DataFrame):
        assert data.index.names == [
            "region",
            "region_name",
            "variable",
            "unit",
        ]
        data = data.copy()
    data.columns = data.columns.astype(int)
    data.index.names = ["region", "region_name", "variable", "unit"]

    # Check if regiong / group already exists in the data
    if countries_name in data.index.get_level_values("region").unique():
        raise ValueError(f"Region/group '{countries_name}' already exists in the data. Please choose a different name or remove the existing entry.")
    
    else:
        # Fix: use get_level_values for MultiIndex filtering
        data_filt = data[data.index.get_level_values("region").isin(countries)]

        if data_filt.empty:
            raise ValueError(f"No data found for countries: {countries}")

        # Group by variable and unit before summing, so multiple variables
        # are handled correctly rather than summed across
        aggregated = data_filt.groupby(level=["variable", "unit"]).sum()

        # Reconstruct the full MultiIndex, inferring variable/unit from the data
        # but allowing region and region_name to be passed in or derived
        aggregated.index = pd.MultiIndex.from_tuples(
            [
                (countries_name, countries_name, variable, unit)
                for variable, unit in aggregated.index
            ],
            names=["region", "region_name", "variable", "unit"],
        )
        data_updated = pd.concat([data, aggregated])
        data_updated.to_csv(data_path)

    return aggregated

def delete_group(countries_name, data_csv):
    data_path = f"../data/processed/{data_csv}"
    data = pd.read_csv(data_path, index_col=[0, 1, 2, 3])
    data = data[data.index.get_level_values("region") != countries_name]
    data.to_csv(data_path)

#  Calculate carbon intensity of energy and GDP, and energy intensity of GDP, for groups
def calc_intensity(region, numerator_csv, denominator_csv, 
                            output_csv,
                            emissions_var=str,
                            energy_var=str,
                            intensity_label=str):
    """
    
    Calculates intensity (e.g. carbon intensity of energy) for a specified region/group and saves it to the output CSV.
    
    Args:
        region: ISO code of the region/group to calculate intensity for (e.g. "USA", "EU27", "World", or a custom group name created by combine_countries)  
        numerator_csv: Name of the CSV file containing the numerator data
        denominator_csv: Name of the CSV file containing the denominator data
        output_csv: Name of the CSV file to save the intensity data
        emissions_var: Variable name for the emissions data
        energy_var: Variable name for the energy data
        intensity_label: Label for the intensity variable (e.g. "Carbon Intensity of Energy")
        Returns:
            DataFrame with the calculated intensity for the specified region/group, indexed by variable and unit.   
    
    """

    def load_and_filter(csv_name, variable):
        data_path = f"../data/processed/{csv_name}"
        data = pd.read_csv(data_path, index_col=[0, 1, 2, 3])
        data.columns = data.columns.astype(int)
        data.index.names = ["region", "region_name", "variable", "unit"]
        # Filter for countries and variable
        region_mask = data.index.get_level_values("region") == region
        var_mask = data.index.get_level_values("variable") == variable
        filtered = data[region_mask & var_mask]
        if filtered.empty:
            raise ValueError(f"No data found for variable '{variable}' and region: {region}")
        return filtered

    numerator = load_and_filter(numerator_csv, emissions_var).groupby(level=["variable", "unit"]).sum()
    denominator = load_and_filter(denominator_csv, energy_var).groupby(level=["variable", "unit"]).sum()

    # Align on shared years before dividing
    shared_years = numerator.columns.intersection(denominator.columns)
    if shared_years.empty:
        raise ValueError("No overlapping years between emissions and energy data")

    intensity = numerator[shared_years].values / denominator[shared_years].values

    # Infer units from the data
    emissions_unit = numerator.index.get_level_values("unit")[0]
    energy_unit = denominator.index.get_level_values("unit")[0]

    intensity_df = pd.DataFrame(
        intensity,
        columns=shared_years,
        index=pd.MultiIndex.from_tuples(
            [(region, region, intensity_label, f'{emissions_unit}/{energy_unit}')],
            names=["region", "region_name", "variable", "unit"]
        )
    )
    # Load output csv, drop any existing row for this group, append and save
    output_path = f"../data/processed/{output_csv}"
    existing = pd.read_csv(output_path, index_col=[0, 1, 2, 3])
    existing.columns = existing.columns.astype(int)
    existing.index.names = ["region", "region_name", "variable", "unit"]
    existing = existing[existing.index.get_level_values("region") != region]
    updated = pd.concat([existing, intensity_df])
    updated.to_csv(output_path)

    return intensity_df
