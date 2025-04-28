import pandas as pd
import numpy as np
from readData import getPopulation

# ---- Configurable Values ----
ZHVI_CSV_PATH = 'data/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
COMMERCIAL_RENT_CSV_PATH = 'data/limited_commercial_rent_data.csv'
URBAN_THRESHOLD = 50000
SUBURBAN_THRESHOLD = 5000
STORE_SIZE_SQFT = 2500  # Size of the store

# Default multipliers if no matching market found
DEFAULT_MULTIPLIERS = {
    'Urban': 2.25,
    'Suburban': 1.75,
    'Rural': 1.5
}

def calculate_rent_estimation():
    # Load data
    zhvi_df = pd.read_csv(ZHVI_CSV_PATH)
    commercial_rent_df = pd.read_csv(COMMERCIAL_RENT_CSV_PATH)
    population_df = getPopulation()

    # Fix Population Data
    if 'Population' not in population_df.columns:
        latest_year = population_df.columns[-3]
        population_df = population_df[['County', 'State', latest_year]]
        population_df.rename(columns={latest_year: 'Population'}, inplace=True)

    # Extract latest month for ZHVI
    month_cols = zhvi_df.columns[zhvi_df.columns.str.match(r'^\d{4}-\d{2}-\d{2}$')]
    latest_month = month_cols[-1]
    zhvi_df = zhvi_df[['RegionID', 'RegionName', 'StateName', latest_month]]
    zhvi_df.rename(columns={
        'RegionName': 'County',
        'StateName': 'State',
        latest_month: 'ZHVI'
    }, inplace=True)

    zhvi_df.dropna(subset=['ZHVI'], inplace=True)

    # Merge ZHVI and Population
    merged_df = zhvi_df.merge(population_df, on=['County', 'State'], how='left')

    # Classify area
    def classify_area(population):
        if pd.isna(population):
            return 'Rural'  # Default fallback
        if population >= URBAN_THRESHOLD:
            return 'Urban'
        elif population >= SUBURBAN_THRESHOLD:
            return 'Suburban'
        else:
            return 'Rural'

    merged_df['Area_Type'] = merged_df['Population'].apply(classify_area)

    # Clean and match county names
    merged_df['County_cleaned'] = merged_df['County'].str.replace(' County', '', regex=False).str.strip()
    commercial_rent_df['Market'] = commercial_rent_df['Market'].str.strip()
    commercial_rent_df['Price Per Sq. Ft.'] = commercial_rent_df['Price Per Sq. Ft.'].replace('[\$,]', '', regex=True).astype(float)

    # Merge Commercial Rent Data
    merged_df = merged_df.merge(commercial_rent_df[['Market', 'Price Per Sq. Ft.']], left_on='County_cleaned', right_on='Market', how='left')

    # Estimate residential rent per square foot
    merged_df['Estimated_Residential_Rent_Per_Sqft'] = ((merged_df['ZHVI'] * 0.007 * 12) / 1500)

    # Calculate multiplier
    merged_df['Calculated_Multiplier'] = merged_df['Price Per Sq. Ft.'] / merged_df['Estimated_Residential_Rent_Per_Sqft']

    # Fill missing multipliers with default by Area_Type
    merged_df['Final_Multiplier'] = merged_df['Calculated_Multiplier']
    for area_type, default_multiplier in DEFAULT_MULTIPLIERS.items():
        merged_df.loc[(merged_df['Area_Type'] == area_type) & (merged_df['Final_Multiplier'].isna()), 'Final_Multiplier'] = default_multiplier

    # Calculate commercial rent per sqft per year
    merged_df['Commercial_rent_per_sqft_year'] = merged_df['Estimated_Residential_Rent_Per_Sqft'] * merged_df['Final_Multiplier']

    # Calculate estimated annual rent for store size
    merged_df['Estimated_annual_rent'] = (merged_df['Commercial_rent_per_sqft_year'] * STORE_SIZE_SQFT).round(2)

    return merged_df

if __name__ == "__main__":
    result_df = calculate_rent_estimation()
    print(result_df[['County', 'State', 'Population', 'Area_Type', 'Final_Multiplier', 'Commercial_rent_per_sqft_year', 'Estimated_annual_rent']])