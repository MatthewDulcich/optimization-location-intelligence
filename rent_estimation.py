import pandas as pd
import numpy as np
from readData import getPopulation  # Import the getPopulation function

# ---- Step 1: Load Zillow ZHVI CSV
zhvi_path = 'data/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
zhvi_df = pd.read_csv(zhvi_path)

# ---- Step 2: Load Population Data using getPopulation()
population_df = getPopulation()  # Use the function to get the population data

# Ensure the population data has the correct structure
if 'Population' not in population_df.columns:
    # Get the latest year column for population
    latest_year = population_df.columns[-3]  # Assuming the last numeric column is the latest year
    population_df = population_df[['County', 'State', latest_year]]
    population_df.rename(columns={latest_year: 'Population'}, inplace=True)

# ---- Step 3: Prep ZHVI Data
month_cols = zhvi_df.columns[zhvi_df.columns.str.match(r'^\d{4}-\d{2}-\d{2}$')]
if month_cols.empty:
    raise ValueError("No columns matching the YYYY-MM-DD format were found. Please check the column names in the CSV file.")
latest_month = month_cols[-1]
zhvi_df = zhvi_df[['RegionID', 'RegionName', 'StateName', latest_month]]
zhvi_df.rename(columns={
    'RegionName': 'County',
    'StateName': 'State',
    latest_month: 'ZHVI'
}, inplace=True)
zhvi_df.dropna(subset=['ZHVI'], inplace=True)

# ---- Step 4: Merge ZHVI + Population
merged_df = zhvi_df.merge(population_df, on=['County', 'State'], how='left')

# ---- Step 5: Classify Urban vs Rural
urban_threshold = 150000
merged_df['Urban'] = np.where(merged_df['Population'] >= urban_threshold, 1, 0)

# ---- Step 6: Set Multiplier Based on Urban/Rural
merged_df['Multiplier'] = np.where(merged_df['Urban'] == 1, 2.25, 1.5)

# ---- Step 7: Calculate Commercial Rent
merged_df['Commercial_rent_per_sqft_year'] = merged_df['ZHVI'] * merged_df['Multiplier']

# Assume store size
store_size_sqft = 2500
merged_df['Estimated_annual_rent'] = (merged_df['Commercial_rent_per_sqft_year'] * store_size_sqft).astype(int)

print(merged_df[['County', 'State', 'Population', 'Urban', 'Multiplier', 'Commercial_rent_per_sqft_year', 'Estimated_annual_rent']].head())