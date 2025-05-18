import argparse
from readData import *
from optimizationFunctionBothVars import optimize
from rent_estimation import calculate_rent_estimation
import geopandas as gpd
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime
import numpy as np
import cartopy.feature as cfeature
import pandas as pd

# Use argparse for different constraints
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--budget', type=float, help='Total Budget', default=100000)
parser.add_argument('-N', '--NumberLocations', type=int, help='Max number of Restaurants', default=10)
parser.add_argument('-r', '--risk', type=float, help='Acceptable maximum risk level. Between 0, 1', default=0.8)

args = parser.parse_args()

budget = args.budget
N = args.NumberLocations
risk = args.risk

print(f"Budget: {budget}, Max Locations: {N}, Risk: {risk}")

# Load data
income = getIncome()
income.drop(['MedianIncome', 'MedianIncomeRatio'], axis=1, inplace=True)
populations = getPopulation()
populations = populations[['State', 'County', '2024']]
populations.rename({'2024': 'Population'}, axis=1, inplace=True)
rent = calculate_rent_estimation()
rent = rent[['State', 'County', 'Estimated_annual_rent']]
minwage = getMinWage()
restaurant = pd.read_csv("data/restaurant_data.csv")
print(restaurant[['fips', 'totalRestaurants']])

# Merge data
data = populations.merge(income)
data = data.merge(rent)
data = data.merge(minwage)

print(data['State'].unique())

data['id'] = data['CountyID'].apply(lambda x: x[-5:])
# New FIPS codes https://developer.ap.org/ap-elections-api/docs/CT_FIPS_Codes_forPlanningRegions.htm
data.loc[data['id'] == '09110', 'id'] = '09013'  # Capitol Planning Region = Tolland
data.loc[data['id'] == '09120', 'id'] = '09001'  # Greater Bridgeport = Fairfield
data.loc[data['id'] == '09130', 'id'] = '09007'  # Lower CT River Valley = Middlesex
data.loc[data['id'] == '09140', 'id'] = '09003'  # Naugatuck Valley Planning Region = Hartford
data.loc[data['id'] == '09150', 'id'] = '09015'  # Northeastern CT Planning Region = Windham
data.loc[data['id'] == '09160', 'id'] = '09005'  # Northwest Hills Planning Region = Litchfield
data.loc[data['id'] == '09170', 'id'] = '09009'  # South Central Planning Region = New Haven
data.loc[data['id'] == '09180', 'id'] = '09011'  # Southeastern Planning Region = New London
data.loc[data['id'] == '09190', 'id'] = '09001'  # Western Connecticut Planning Region = Fairfield
data['id'] = data['id'].astype(int)

data = data.merge(restaurant[['fips', 'totalRestaurants']], left_on='id', right_on='fips', how='left')
data.rename(columns={'totalRestaurants': 'HistoricRestaurants'}, inplace=True)
data.loc[data['id'] == 46102, 'id'] = 46113  # Shannon County is Oglala Lakota

data.loc[data['HistoricRestaurants'] == 0, 'HistoricRestaurants'] = data['HistoricRestaurants'].mean()
data.loc[np.isnan(data['HistoricRestaurants']), 'HistoricRestaurants'] = data['HistoricRestaurants'].mean()

# Limit data to the first 100 rows for testing
# data = data.head(20)

# Run optimization on the full dataset
results = optimize(
    budget=budget,
    N=N,
    risk=risk,
    totalPop=data['Population'],
    IR=data['MeanIncomeRatio'],
    minwage=data['MinWage'],
    rent=data['Estimated_annual_rent'],
    NRestaurants=data['HistoricRestaurants']
)
data['UnitsSold'] = results['optimal_x']
data['Served'] = [1 if xi > 0 else 0 for xi in results['optimal_x']]
data['Profit'] = results['profit']
result_df = data.copy()

# Handle NaN values before plotting
result_df['UnitsSold'] = pd.to_numeric(result_df['UnitsSold'], errors='coerce').fillna(0)
result_df['Profit'] = pd.to_numeric(result_df['Profit'], errors='coerce').fillna(0)

# Cap the maximum value of historic restaurant counts for plotting
result_df['HistoricRestaurants'] = result_df['HistoricRestaurants'].clip(upper=100)

# Graphing results
geoData = gpd.read_file(
    'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson'
)

# Ensure the "id" column is an integer
geoData['id'] = geoData['id'].astype(int)
result_df['id'] = result_df['id'].astype(int)

# Remove Alaska, Hawaii, and Puerto Rico using their state names
statesToRemove = ['Alaska', 'Hawaii', 'Puerto Rico']
geoData = geoData[~geoData['STATE'].isin(statesToRemove)]
result_df = result_df[~result_df['State'].isin(statesToRemove)]

# Merge geoData with result_df
result_df = geoData.merge(result_df, on='id')

# Plot Units Sold by County
ax = plt.axes(projection=gcrs.PlateCarree())
norm = Normalize(vmin=result_df['UnitsSold'].min(), vmax=result_df['UnitsSold'].max())
im = gplt.choropleth(
    result_df,
    hue='UnitsSold',
    projection=gcrs.PlateCarree(),
    cmap='viridis',
    ax=ax,
    norm=norm
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
plt.colorbar(cbar, ax=ax, pad=-0.25, shrink=0.4)
plt.title(f'Units Sold by County', y=0.8, fontsize='small')
plt.savefig(f"Images/Units_Sold_By_County.png", bbox_inches='tight', dpi=300)
plt.close()

# Plot Profit by County
ax = plt.axes(projection=gcrs.PlateCarree())
norm = Normalize(vmin=result_df['Profit'].min(), vmax=result_df['Profit'].max())
im = gplt.choropleth(
    result_df,
    hue='Profit',
    projection=gcrs.PlateCarree(),
    cmap='bwr',
    ax=ax,
    norm=norm
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap='bwr')
plt.colorbar(cbar, ax=ax, pad=-0.25, shrink=0.4)
plt.title(f'Profit by County', y=0.8, fontsize='small')
plt.savefig(f"Images/Profit_By_County.png", bbox_inches='tight', dpi=300)
plt.close()

# Plot Total Restaurants by County
ax = plt.axes(projection=gcrs.PlateCarree())
norm = Normalize(vmin=0, vmax=150)
im = gplt.choropleth(
    result_df,
    hue='HistoricRestaurants',
    projection=gcrs.PlateCarree(),
    cmap='plasma',
    ax=ax,
    norm=norm
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
plt.colorbar(cbar, ax=ax, pad=-0.25, shrink=0.4)
plt.title(f'Total Restaurants by County', y=0.8, fontsize='small')
plt.savefig(f"Images/Total_Restaurants_By_County.png", bbox_inches='tight', dpi=300)
plt.close()

print("Optimization complete. Results saved and graphs generated.")
