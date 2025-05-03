import matplotlib.pyplot as plt
import numpy as np
from readData import *
from rent_estimation import calculate_rent_estimation

def demand(L,P):
    '''
    Calculates demand as a function of price.

    args:
        L: Maximum demand at unit price 1.0.

        a: change in elasticity as a function of price.

        P: Price.

    returns:
        demand: Demand corresponding to input variables.
    '''

    a = 0.003*P

    demand = L*np.exp(-a*P)

    return demand

def revenue(demand,P,IR):
    '''
    Calculates Revenue of Product

    args:
        demand: The demand of a given product

        P: Price of a given product

        IR: Income ratio of county to maximum county income. Between 0 and 1.

    returns:
        revenue: Demand times Price
    '''
    return demand*P**IR

prices = np.arange(0,50,1)

income = getIncome()
populations = getPopulation()
populations = populations[['State','County','2024']]
populations.rename({'2024':'Population'},axis=1,inplace=True)
rent = calculate_rent_estimation()
rent = rent[['State','County','Estimated_annual_rent']]

data = populations.merge(income)
data = data.merge(rent)
print(data['MeanIncomeRatio'].describe())
data['MeanIncomeRatio'] = data['MeanIncomeRatio']**0.5
print(data['MeanIncomeRatio'].describe())

print(data)
for i in range(10):
    pop = data['Population'].iloc[i] # Population of Autauga County, Alabama
    state = data['State'].iloc[i]
    county = data['County'].iloc[i]
    IR = data['MeanIncomeRatio'].iloc[i]

    # Assume demand for unit price is just the county population
    demands = [demand(pop,price) for price in prices]
    logdemands = [np.log(demand(pop,price)) for price in prices]

    revenues = [revenue(i,j,IR) for i,j in zip(demands,prices)]

    fig, axs = plt.subplots(ncols=4,figsize=(12,3))

    plt.suptitle(f"{state}, {county}")
    axs[0].plot(prices,demands)
    axs[0].set_xlabel('Price')
    axs[0].set_ylabel('Demand')
    axs[0].set_title('Price vs Demand')

    axs[1].plot(prices,logdemands)
    axs[1].set_xlabel('Price')
    axs[1].set_ylabel('Log Demand')
    axs[1].set_title('Price vs Log Demand')

    axs[2].plot(prices,revenues)
    axs[2].set_xlabel('Price')
    axs[2].set_ylabel('Revenue')
    axs[2].set_title('Price vs Revenue')

    axs[3].plot(demands,revenues)
    axs[3].set_xlabel('Demand')
    axs[3].set_ylabel('Revenue')
    axs[3].set_title('Demand vs Revenue')
    plt.tight_layout()
    plt.show()
