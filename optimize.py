import argparse
from readData import *
from optimizationFunction import optimize, optimize_tf, optimize_integer  # Import optimize_integer
from rent_estimation import calculate_rent_estimation
import tensorflow as tf

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
populations = getPopulation()
populations = populations[['State', 'County', '2024']]
populations.rename({'2024': 'Population'}, axis=1, inplace=True)
rent = calculate_rent_estimation()
rent = rent[['State', 'County', 'Estimated_annual_rent']]
minwage = getMinWage()

data = populations.merge(income)
data = data.merge(rent)
data = data.merge(minwage)

# Convert data to lists for pulp
totalPop = data['Population'].tolist()
IR = data['MeanIncomeRatio'].tolist()
minwage = data['MinWage'].tolist()
rent = data['Estimated_annual_rent'].tolist()

# Run the optimization
result = optimize_integer(
    budget=budget,
    N=N,
    risk=risk,
    totalPop=totalPop,
    IR=IR,
    minwage=minwage,
    rent=rent
)

# Print results
print("Optimized Number of Stores (x):", result["x"])
print("Optimized Prices (P):", result["P"])
print("Optimized Profit:", result["profit"])

# Add the optimized number of stores to the DataFrame
data['Optimized_Stores'] = result["x"]

# Display the data with state, county, and number of stores
print("\nNumber of Stores per Location:")
print(data[['State', 'County', 'Optimized_Stores']])
