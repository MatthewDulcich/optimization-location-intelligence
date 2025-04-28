import pandas as pd
import numpy as np
from readData import getPopulation  # Import the getPopulation function

# ---- Configurable Values ----
ZHVI_CSV_PATH = 'data/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'  # Path to ZHVI data
URBAN_THRESHOLD = 150000  # Population threshold for urban classification
SUBURBAN_THRESHOLD = 50000  # Population threshold for suburban classification
STORE_SIZE_SQFT = 2500  # Assumed store size in square feet

# Multipliers for area types
MULTIPLIERS = {
    'Urban': 2.25,
    'Suburban': 1.75,
    'Rural': 1.5
}

def calculate_rent_estimation():
    # ---- Step 1: Load Zillow ZHVI CSV
    zhvi_df = pd.read_csv(ZHVI_CSV_PATH)

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

    # ---- Step 5: Classify Urban, Suburban, and Rural
    def classify_area(population):
        if population >= URBAN_THRESHOLD:
            return 'Urban'
        elif population >= SUBURBAN_THRESHOLD:
            return 'Suburban'
        else:
            return 'Rural'

    merged_df['Area_Type'] = merged_df['Population'].apply(classify_area)

    # ---- Step 6: Set Multiplier Based on Area Type
    merged_df['Multiplier'] = merged_df['Area_Type'].map(MULTIPLIERS)

    # ---- Step 7: Calculate Commercial Rent
    merged_df['Commercial_rent_per_sqft_year'] = merged_df['ZHVI'] * merged_df['Multiplier']

    # Assume store size
    merged_df['Estimated_annual_rent'] = (merged_df['Commercial_rent_per_sqft_year'] * STORE_SIZE_SQFT).astype(int)

    return merged_df

if __name__ == "__main__":
    result_df = calculate_rent_estimation()

    print(result_df[['County', 'State', 'Population', 'Area_Type', 'Multiplier', 'Commercial_rent_per_sqft_year', 'Estimated_annual_rent']].head())