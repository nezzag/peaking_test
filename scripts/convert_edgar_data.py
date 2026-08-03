import pandas as pd
import openpyxl
from collections import defaultdict

XLSX_PATH = '/mnt/user-data/uploads/EDGAR_2026_GHG_booklet_2026_with_LULUCF.xlsx'
OUT_PATH = '/mnt/user-data/outputs/edgar_emissions_v2.csv'

GAS_LABELS = {
    'CO2': 'CO2',
    'GWP_100_AR5_CH4': 'CH4',
    'GWP_100_AR5_N2O': 'N2O',
    'GWP_100_AR5_F-gases': 'F-gases',
}

UNIT = 'Mt CO2eq/yr'

EU27 = {'AUT','BEL','BGR','HRV','CYP','CZE','DNK','EST','FIN','FRA','DEU',
        'GRC','HUN','IRL','ITA','LVA','LTU','LUX','MLT','NLD','POL','PRT',
        'ROU','SVK','SVN','ESP','SWE'}

print("Loading EDGAR xlsx...")
wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)

# ── 1. Main emissions (unchanged from v1) ────────────────────────────────────

ws_totals = wb['GHG_totals_by_country']
rows_totals = list(ws_totals.iter_rows(values_only=True))
header_totals = rows_totals[0]
years_totals = [c for c in header_totals if isinstance(c, int)]

records = []

for row in rows_totals[1:]:
    code, name = row[0], row[1]
    if not code or not name:
        continue
    vals = {yr: row[i+2] for i, yr in enumerate(years_totals)}
    records.append({
        'region': code,
        'region_name': name,
        'variable': 'Emissions|GHG|Total',
        'unit': UNIT,
        **vals
    })

ws_sector = wb['GHG_by_sector_and_country']
rows_sector = list(ws_sector.iter_rows(values_only=True))
header_sector = rows_sector[0]
years_sector = [c for c in header_sector if isinstance(c, int)]

sector_data = {}
for row in rows_sector[1:]:
    gas, sector, code, name = row[0], row[1], row[2], row[3]
    if not all([gas, sector, code, name]):
        continue
    sector_data[(gas, sector, code, name)] = {yr: row[i+4] for i, yr in enumerate(years_sector)}

for (gas, sector, code, name), vals in sector_data.items():
    gas_label = GAS_LABELS.get(gas, gas)
    records.append({
        'region': code,
        'region_name': name,
        'variable': f'Emissions|{gas_label}|{sector}',
        'unit': UNIT,
        **vals
    })

gas_totals = defaultdict(lambda: defaultdict(float))
for (gas, sector, code, name), vals in sector_data.items():
    key = (gas, code, name)
    for yr, v in vals.items():
        if v is not None:
            gas_totals[key][yr] += v

for (gas, code, name), yr_vals in gas_totals.items():
    gas_label = GAS_LABELS.get(gas, gas)
    records.append({
        'region': code,
        'region_name': name,
        'variable': f'Emissions|{gas_label}|Total',
        'unit': UNIT,
        **yr_vals
    })

print(f"  Main emissions: {len(records)} rows so far")

# ── 2. LULUCF net GHG ────────────────────────────────────────────────────────

ws_lulucf = wb['LULUCF_countries']
rows_lulucf = list(ws_lulucf.iter_rows(values_only=True))
header_lulucf = rows_lulucf[0]
years_lulucf = [c for c in header_lulucf if isinstance(c, int)]
yr_pos = {yr: 5+i for i, yr in enumerate(years_lulucf)}

# Collect Net GHG for individual countries (proper ISO codes only)
country_lulucf = {}  # code -> {yr: val}
country_names = {}   # code -> name

for row in rows_lulucf[1:]:
    substance, sector, code, name = row[0], row[1], row[2], row[3]
    if substance == 'GWP_100_AR5_GHG' and sector == 'Net' and code and name:
        country_lulucf[code] = {yr: row[yr_pos[yr]] for yr in years_lulucf}
        country_names[code] = name

# Individual country rows
for code, vals in country_lulucf.items():
    records.append({
        'region': code,
        'region_name': country_names[code],
        'variable': 'Emissions|GHG|LULUCF',
        'unit': UNIT,
        **vals
    })

# EU27: sum member states
eu27_lulucf = defaultdict(float)
for code in EU27:
    if code in country_lulucf:
        for yr, val in country_lulucf[code].items():
            if val is not None:
                eu27_lulucf[yr] += val

records.append({
    'region': 'EU27',
    'region_name': 'EU27',
    'variable': 'Emissions|GHG|LULUCF',
    'unit': UNIT,
    **eu27_lulucf
})

# Global: sum all individual countries
global_lulucf = defaultdict(float)
for code, vals in country_lulucf.items():
    for yr, val in vals.items():
        if val is not None:
            global_lulucf[yr] += val

records.append({
    'region': 'WLD',
    'region_name': 'WLD',
    'variable': 'Emissions|GHG|LULUCF',
    'unit': UNIT,
    **global_lulucf
})

print(f"  After LULUCF rows: {len(records)} rows")

# ── 3. Rename GLOBAL TOTAL → WLD ─────────────────────────────────────────────
for r in records:
    if r['region'] == 'GLOBAL TOTAL':
        r['region'] = 'WLD'
        r['region_name'] = 'WLD'

# ── 4. Total GHG incl LULUCF ─────────────────────────────────────────────────
# Build lookup from existing records for GHG Total and LULUCF
ghg_total = {}   # region -> {yr: val}
lulucf_total = {}  # region -> {yr: val}

for r in records:
    region = r['region']
    if r['variable'] == 'Emissions|GHG|Total':
        ghg_total[region] = {yr: r.get(yr) for yr in years_totals}
    elif r['variable'] == 'Emissions|GHG|LULUCF':
        lulucf_total[region] = {yr: r.get(yr) for yr in years_lulucf}

# Regions that have both
common_years = sorted(set(years_totals) & set(years_lulucf))
regions_with_both = set(ghg_total.keys()) & set(lulucf_total.keys())

# Get region names from existing records
region_names = {r['region']: r['region_name'] for r in records if 'region_name' in r}

for region in regions_with_both:
    combined = {}
    for yr in common_years:
        ghg_val = ghg_total[region].get(yr)
        lulucf_val = lulucf_total[region].get(yr)
        if ghg_val is not None and lulucf_val is not None:
            combined[yr] = ghg_val + lulucf_val
    if combined:
        records.append({
            'region': region,
            'region_name': region_names.get(region, region),
            'variable': 'Emissions|GHG|Total_incl_LULUCF',
            'unit': UNIT,
            **combined
        })

print(f"  After total incl LULUCF rows: {len(records)} rows")

# ── 5. Write CSV ──────────────────────────────────────────────────────────────
all_years = sorted(set(years_totals) | set(years_lulucf))
df = pd.DataFrame(records)
df = df.set_index(['region', 'region_name', 'variable', 'unit'])
year_cols = sorted([c for c in df.columns if isinstance(c, int)])
df = df[year_cols]
df.to_csv(OUT_PATH)

print(f"\nDone! Written to {OUT_PATH}")
print(f"Shape: {df.shape}")
print(f"\nVariables:")
for v in sorted(df.index.get_level_values('variable').unique()):
    print(f"  {v}")

# Sanity check
print("\nSanity check - WLD GHG Total incl LULUCF (2023):")
wld_ghg = df.loc[('WLD', 'WLD', 'Emissions|GHG|Total', UNIT), 2023]
wld_lulucf = df.loc[('WLD', 'WLD', 'Emissions|GHG|LULUCF', UNIT), 2023]
wld_total = df.loc[('WLD', 'WLD', 'Emissions|GHG|Total_incl_LULUCF', UNIT), 2023]
print(f"  GHG excl LULUCF: {wld_ghg:.2f}")
print(f"  LULUCF:          {wld_lulucf:.2f}")
print(f"  Total incl LULUCF: {wld_total:.2f} (should = {wld_ghg + wld_lulucf:.2f})")
