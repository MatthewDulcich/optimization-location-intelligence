import argparse
from readData import *
from optimizationFunction import optimize_integer
from rent_estimation import calculate_rent_estimation
import geopandas as gpd
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
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

print(budget, N, risk)

# Load data
income = getIncome()
income.drop(['MedianIncome', 'MedianIncomeRatio'], axis=1, inplace=True)
income['id'] = income['CountyID'].apply(lambda x: x.split('US')[1])

# Correct specific FIPS codes
income.loc[income['id'] == '46102', 'id'] = '46113'  # Shannon County is Oglala Lakota
income.loc[income['id'] == '09110', 'id'] = '09013'  # Capitol Planning Region = Tolland
income.loc[income['id'] == '09120', 'id'] = '09001'  # Greater Bridgeport = Fairfield
income.loc[income['id'] == '09130', 'id'] = '09007'  # Lower CT River Valley = Middlesex
income.loc[income['id'] == '09140', 'id'] = '09003'  # Naugatuck Valley Planning Region = Hartford
income.loc[income['id'] == '09150', 'id'] = '09015'  # Northeastern CT Planning Region = Windham
income.loc[income['id'] == '09160', 'id'] = '09005'  # Northwest Hills Planning Region = Litchfield
income.loc[income['id'] == '09170', 'id'] = '09009'  # South Central Planning Region = New Haven
income.loc[income['id'] == '09180', 'id'] = '09011'  # Southeastern Planning Region = New London
income.loc[income['id'] == '09190', 'id'] = '09001'  # Western Connecticut Planning Region = Fairfield

populations = getPopulation()
populations = populations[['State', 'County', '2024']]
populations.rename({'2024': 'Population'}, axis=1, inplace=True)
rent = calculate_rent_estimation()
rent = rent[['State', 'County', 'Estimated_annual_rent']]
minwage = getMinWage()

# Load the geospatial data for US counties
geoData = gpd.read_file(
    'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson'
)

# Remove Alaska, Hawaii, and Puerto Rico
statesToRemove = ['02', '15', '72']
geoData = geoData[~geoData.STATE.isin(statesToRemove)]

# Merge the GeoJSON data with the income data
data = geoData.merge(income, on='id')

# Merge with other datasets without specifying 'County'
data = data.merge(populations)  # Automatically merges on matching columns
data = data.merge(rent)         # Automatically merges on matching columns
data = data.merge(minwage)      # Automatically merges on matching columns

# Rename and process columns
data['id'] = data['id'].astype(int)
data.rename({'Estimated_annual_rent': 'Rent'}, axis=1, inplace=True)
data['Population'] = np.log(data['Population'])
data['Rent'] = np.log(data['Rent'])

# Run the optimization function
result = optimize_integer(
    budget=budget,
    N=N,
    risk=risk,
    totalPop=data['Population'].tolist(),
    IR=data['MeanIncomeRatio'].tolist(),
    minwage=data['MinWage'].tolist(),
    rent=data['Rent'].tolist()
)

print("Optimization Result:", result)

# Add the optimization results to the DataFrame
# Assuming the optimization function returns a dictionary with 'id' and 'Stores'
optimization_results = pd.DataFrame({
    'id': data['id'],  # Ensure the 'id' column matches the GeoJSON data
    'Stores': result['x']  # Replace 'x' with the actual key for the number of stores
})

# Merge the optimization results into the data DataFrame
data = data.merge(optimization_results, on='id')

# Plot variables
varnames = ['Minimum Wage', 'Log Population', 'Mean Income', 'Log Rent']
for i, var in enumerate(['MinWage', 'Population', 'MeanIncome', 'Rent']):
    ax = plt.axes(projection=gcrs.PlateCarree())
    im = gplt.choropleth(
        data,
        hue=var,
        projection=gcrs.PlateCarree(),
        extent=[-150, 15, -40, 60],
        cmap='viridis',
        ax=ax
    )

    norm = Normalize(vmin=data[var].min(), vmax=data[var].max())
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    plt.colorbar(cbar, ax=ax, pad=-0.25, shrink=0.4)
    plt.title(f'{varnames[i]} By County', y=0.8, fontsize='small')
    plt.savefig(f"Images/{var}.png", bbox_inches='tight', dpi=300)
    plt.close()

# Plot the number of stores in the counties
ax = plt.axes(projection=gcrs.PlateCarree())
im = gplt.choropleth(
    data,
    hue='Stores',  # Use the column for the number of stores
    projection=gcrs.PlateCarree(),
    extent=[-150, 15, -40, 60],
    cmap='OrRd',  # Use a red color map for better visualization
    ax=ax
)

# Normalize the color scale
norm = Normalize(vmin=data['Stores'].min(), vmax=data['Stores'].max())
cbar = plt.cm.ScalarMappable(norm=norm, cmap='OrRd')
plt.colorbar(cbar, ax=ax, pad=-0.25, shrink=0.4)

# Add a title
plt.title('Number of Stores by County', y=0.8, fontsize='small')

# Save the image
plt.savefig("Images/Number_of_Stores.png", bbox_inches='tight', dpi=300)
plt.close()
