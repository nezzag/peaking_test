"""Module for supporting helper functions in the analysis"""

import pycountry
import os
import sys

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
