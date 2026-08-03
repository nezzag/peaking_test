import pandas as pd

IN_PATH  = '/mnt/user-data/uploads/ember_electricity.csv'
OUT_PATH = '/mnt/user-data/outputs/ember_electricity_formatted.csv'

print("Loading Ember data...")
df = pd.read_csv(IN_PATH, on_bad_lines='skip')

# ── 1. Define rows to extract ─────────────────────────────────────────────────
# (Category, Subcategory, Unit) → variable name template
# For Fuel rows the variable name includes the fuel type (Variable column)
# For Total rows the variable name is fixed

def get_variable_name(row):
    cat    = row['Category']
    subcat = row['Subcategory']
    var    = row['Variable']
    unit   = row['Unit']

    if cat == 'Electricity generation' and subcat == 'Fuel' and unit == 'TWh':
        return f'Electricity_generation|{var}'
    elif cat == 'Power sector emissions' and subcat == 'Fuel' and unit == 'mtCO2':
        return f'Emissions|CO2|Power Industry|{var}'
    elif cat == 'Electricity generation' and subcat == 'Total' and unit == 'TWh':
        return 'Electricity_generation|Total'
    elif cat == 'Power sector emissions' and subcat == 'Total' and unit == 'mtCO2':
        return 'Emissions|CO2|Power Industry|Total'
    else:
        return None

# ── 2. Filter to rows we want ─────────────────────────────────────────────────
mask = (
    ((df['Category'] == 'Electricity generation') & (df['Subcategory'] == 'Fuel')  & (df['Unit'] == 'TWh'))   |
    ((df['Category'] == 'Power sector emissions') & (df['Subcategory'] == 'Fuel')  & (df['Unit'] == 'mtCO2')) |
    ((df['Category'] == 'Electricity generation') & (df['Subcategory'] == 'Total') & (df['Unit'] == 'TWh'))   |
    ((df['Category'] == 'Power sector emissions') & (df['Subcategory'] == 'Total') & (df['Unit'] == 'mtCO2'))
)
sub = df[mask].copy()
sub['variable'] = sub.apply(get_variable_name, axis=1)

# Drop rows where variable couldn't be assigned
sub = sub[sub['variable'].notna()]

# ── 3. Clean up unit labels ───────────────────────────────────────────────────
unit_map = {'TWh': 'TWh', 'mtCO2': 'MtCO2'}
sub['unit'] = sub['Unit'].map(unit_map)

# ── 4. Rename region columns ──────────────────────────────────────────────────
sub = sub.rename(columns={
    'ISO 3 code': 'region',
    'Area':       'region_name',
})

# Drop rows with no ISO code
sub = sub[sub['region'].notna() & (sub['region'] != '')]

# ── 5. Pivot years to columns ─────────────────────────────────────────────────
sub['Year'] = sub['Year'].astype(int)

pivot = (
    sub[['region', 'region_name', 'variable', 'unit', 'Year', 'Value']]
    .pivot_table(
        index=['region', 'region_name', 'variable', 'unit'],
        columns='Year',
        values='Value',
        aggfunc='first'   # should be unique per region/variable/year
    )
    .reset_index()
)

# Flatten column names (pivot_table adds 'Year' as column name axis label)
pivot.columns.name = None

# ── 6. Set index and write ────────────────────────────────────────────────────
pivot = pivot.set_index(['region', 'region_name', 'variable', 'unit'])
pivot.to_csv(OUT_PATH)

print(f"Done! Written to {OUT_PATH}")
print(f"Shape: {pivot.shape}")
print(f"\nVariables:")
for v in sorted(pivot.index.get_level_values('variable').unique()):
    print(f"  {v}")
print(f"\nYear range: {pivot.columns.min()} – {pivot.columns.max()}")
print(f"\nSample regions:")
regions = pivot.index.get_level_values('region').unique()
print(f"  Total: {len(regions)}")
for r in list(regions[:5]):
    print(f"  {r}")
