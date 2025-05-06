import argparse
from readData import *
from optimizationFunction import *
from rent_estimation import calculate_rent_estimation
import matplotlib.pyplot as plt

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

data = populations.merge(income)
data = data.merge(rent)
data = data.merge(minwage)

#print(data)
#exit()

from datetime import datetime
'''
objectiveFunction(x, totalPop, IR, minwage, rent)
start = datetime.now()
results = optimize(budget = budget,
                   N = N,
                   risk = risk,
                   totalPop = data['Population'],
                   IR = data['MeanIncomeRatio'],
                   minwage = data['MinWage'],
                   rent = data['Estimated_annual_rent'])
'''
print(data)
data=data.loc[data['County'].isin(['Los Angeles',
                                   'San Fransisco',
                                   'Baltimore',
                                   'Denver',
                                   'Maricopa',
                                   'San Diego',
                                   'King'])]
for i in range(data.shape[0]):
    state = data['State'].iloc[i]
    county = data['County'].iloc[i]
    pop = data['Population'].iloc[i]
    IR = data['MeanIncomeRatio'].iloc[i]
    minwage = data['MinWage'].iloc[i]
    rent = data['Estimated_annual_rent'].iloc[i]
    obj=[]
    '''
    for x in np.arange(0,10,0.1):
        for P in np.arange(0,25,0.1):
            obj.append([x, P, x * profit(x, P, pop, IR, minwage, rent)])
    obj = np.array(obj)
    print(obj)

    fig = plt.figure()
    ax = plt.axes()
    #ax = fig.add_subplot(projection='3d')
    ax.scatter(obj[:,0],obj[:,1],c=obj[:,2],marker='s',s=5)
    ax.set_xlabel('Number of Stores')
    ax.set_ylabel('Price of Good')
    #ax.set_zlabel('Profit')
    plt.show()
    '''
    obj = []
    demand_val = []
    rev = []
    cost = []
    P=18
    for j, x in enumerate(np.arange(0,10,0.1)):
        obj.append([x * profit(x, P, pop, IR, minwage, rent)])
        demand_val.append(demand(x, P, pop))
        rev.append(x * revenue(P, IR, demand_val[j]))
        cost.append(x * costs(P, IR, minwage, rent, demand_val[j]))

    fig,axs=plt.subplots(ncols=3,figsize=(12,4))
    axs[0].plot(np.arange(0,10,0.1),obj)
    axs[0].set_xlabel("Number of Stores (Price=18 constant)")
    axs[0].set_ylabel("Profit")
    axs[0].set_title(f"{state},{county}")

    axs[1].plot(np.arange(0,10,0.1),demand_val)
    axs[1].set_xlabel("Number of Stores (Price=18 constant)")
    axs[1].set_ylabel("Demand")
    axs[1].set_title(f"{state},{county}")

    axs[2].plot(np.arange(0,10,0.1),rev,label='Revenue')
    axs[2].plot(np.arange(0,10,0.1),cost,label='Cost')
    axs[2].set_xlabel("Number of Stores (Price=18 constant)")
    axs[2].set_ylabel("Revenue")
    axs[2].set_title(f"{state},{county}")
    axs[2].legend()
    plt.tight_layout()
    plt.show()


    obj = []
    demand_val = []
    rev = []
    cost = []
    x=1
    for j, P in enumerate(np.arange(0,50,0.1)):
        obj.append([x * profit(x, P, pop, IR, minwage, rent)])
        demand_val.append(demand(x, P, pop))
        rev.append(x * revenue(P, IR, demand_val[j]))
        cost.append(x * costs(P, IR, minwage, rent, demand_val[j]))

    fig,axs=plt.subplots(ncols=3,figsize=(12,4))
    axs[0].plot(np.arange(0,50,0.1),obj)
    axs[0].set_xlabel("Price (x=1 constant)")
    axs[0].set_ylabel("Profit")
    axs[0].set_title(f"{state},{county}")

    axs[1].plot(np.arange(0,50,0.1),demand_val)
    axs[1].set_xlabel("Price (x=1 constant)")
    axs[1].set_ylabel("Demand")
    axs[1].set_title(f"{state},{county}")

    axs[2].plot(np.arange(0,50,0.1),rev,label='Revenue')
    axs[2].plot(np.arange(0,50,0.1),cost,label='Cost')
    axs[2].set_xlabel("Price (x=1 constant)")
    axs[2].set_ylabel("Revenue")
    axs[2].set_title(f"{state},{county}")
    axs[2].legend()
    plt.tight_layout()
    plt.show()
exit()
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
