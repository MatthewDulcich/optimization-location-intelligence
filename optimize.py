import argparse
from readData import *
from optimizationFunction import optimize
from rent_estimation import calculate_rent_estimation

# Use argparse for different constraints
parser = argparse.ArgumentParser()
parser.add_argument('-b','--budget',type=float,help='Total Budget', default = 100000)
parser.add_argument('-N','--NumberLocations',type=int,help='Max number of Restaurants', default = 10)
parser.add_argument('-r','--risk',type=float,help='Acceptable maximum risk level. Between 0, 1', default = 0.8)

args = parser.parse_args()

budget = args.budget
N = args.NumberLocations
risk = args.risk

print(budget, N, risk)

income = getIncome()
income.drop(['MedianIncome','MedianIncomeRatio'],axis=1,inplace=True)
populations = getPopulation()
populations = populations[['State','County','2024']]
populations.rename({'2024':'Population'},axis=1,inplace=True)
rent = calculate_rent_estimation()
rent = rent[['State','County','Estimated_annual_rent']]
minwage = getMinWage()
restaurant = pd.read_csv("data/restaurant_data.csv")
print(restaurant[['fips','totalRestaurants']])


data = populations.merge(income)
data = data.merge(rent)
data = data.merge(minwage)
data['id'] = data['CountyID'].apply(lambda x: int(x[-5:]))

data = data.merge(restaurant[['fips','totalRestaurants']],left_on='id',right_on='fips')
print(data)
print("Number counties with no restuarant data",data.loc[data['totalRestaurants']==0].shape[0])

data.loc[data['totalRestaurants']==0,'totalRestaurants'] = data['totalRestaurants'].mean()

data = data.iloc[200:300,:]
#print(data)
#exit()

from datetime import datetime
start = datetime.now()
results = optimize(budget = budget,
                   N = N,
                   risk = risk,
                   totalPop = data['Population'],
                   IR = data['MeanIncomeRatio'],
                   minwage = data['MinWage'],
                   rent = data['Estimated_annual_rent'],
                   NRestaurants = data['totalRestaurants'])

print(results)
print(datetime.now() - start)
x = results.x
profit = results.fun * -100000 # Function value times scale value
data['Prices'] = x[100:]
data['Nstores'] = x[:100]
print(data[['Prices','Nstores']].describe())
print(data['Nstores'].sum())
print(data.loc[data['Nstores']>=1])
print(f"Profit {profit}")
