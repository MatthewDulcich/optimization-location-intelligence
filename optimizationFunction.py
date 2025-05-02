import numpy as np
import pandas as pd
from scipy.optimize import minimize


NUMBER_OF_COUNTIES = 3141 # Number of counties


# ######################################################
# Main functions
# ######################################################


def profit(x, P, totalPop, IR, minwage, rent):
    '''
    Calculates Profit = Revenue - Costs

    args:
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county
        IR: Income Ratio relative to maximum county income
        minwage: Minimum wage in county
        rent: Rent of restaurant

    returns:
        profit: Profit of restaurant
    '''
    demand_val = demand(x, P, totalPop)
    rev = revenue(demand_val, P, IR)
    cost = costs(minwage, rent, demand_val, P, IR)
    return rev - cost


def revenue(demand, P, IR):
    '''
    Calculates Revenue of Product

    args:
        demand: The demand of a given product
        P: Price of a given product
        IR: Income ratio of county to maximum county income. Between 0 and 1.

    returns:
        revenue: Demand times Price
    '''
    return P**IR * demand


def log_revenue(log_demand, P, IR):
    '''
    Calculates log of Revenue of Product. It is a concave function.

    args:
        demand: The demand of a given product
        P: Price of a given product
        IR: Income ratio of county to maximum county income. Between 0 and 1.

    returns:
        log_revenue: Log of Revenue
    '''
    return log_demand + IR * np.log(P)


def demand(x, P, totalPop):
    '''
    Calculates demand as a function of price.

    args:
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county

    returns:
        demand: Demand corresponding to input variables.
    '''
    a = 0.003 * P
    # L: Maximum demand at unit price 1.0.
    L = totalPop / x
    return L * np.exp(-a * P)


def log_demand(x, P, totalPop):
    '''
    Calculates log of demand as a function of price. It is a concave function.

    args:
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county
    
    returns:
        log_demand: Log of demand corresponding to input variables.
    '''
    a = 0.003 * P
    # L: Maximum demand at unit price 1.0.
    L = totalPop / x
    return np.log(L) - a * P


def costs(minwage, rent, demand, P, IR, employees=15):
    '''
    Calculates costs of restaurant.

    args:
        minwage: Minimum wage in county
        rent: Rent of restaurant
        demand: Demand of restaurant
        P: Price of restaurant
        IR: Income ratio of county to maximum county income. Between 0 and 1.
        # initialCosts: Initial costs of opening restaurant
        employees: Number of employees in restaurant (default is 15)
    
    returns:
        netCosts: Total costs of restaurant
    '''
    loss_incurred = loss(demand, P, IR)
    operating_costs = minwage * employees + rent + loss_incurred
    return operating_costs


def loss(demand, P, IR):
    '''
    Calculates losses: 8.2% of revenue incurred by restaurant.

    args:
        demand: Demand of restaurant
        IR: Income ratio of county to maximum county income. Between 0 and 1.
        P: Price of restaurant

    returns:
        loss: Losses incurred by restaurant
    '''
    return 0.082 * revenue(demand, P, IR)


# ######################################################
# Objective function
# ######################################################


def objectiveFunction(x, totalPop, IR, minwage, rent):
    '''
    Objective function for optimization problem.

    args:
        x: Total number of stores + Price per store
        totalPop: Total population in county
        IR: Income Ratio relative to maximum county income

    returns:
        Objective value to minimize
    '''
    x = x[:NUMBER_OF_COUNTIES]
    P = x[NUMBER_OF_COUNTIES:]
    return -x.T @ profit(x, P, totalPop, IR, minwage, rent)


def getConstraints(x, budget, N, risk, totalPop, IR, minwage, rent):
    '''
    Creates constraints for optimization function given certain variables.

    Args:
        budget (int): Total budget for owning restaurants.
        N (int): Maximum number of stores to open.
        risk (float): Total acceptable risk ratio per location (cost/revenue)
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county
        IR: Income Ratio relative to maximum county income
        minwage: Minimum wage in county
        rent: Rent of restaurant

    Returns:
        constraints: List of constraints to use in optimization function.
    '''
    x = x[:NUMBER_OF_COUNTIES]
    P = x[NUMBER_OF_COUNTIES:]
    demand_val = demand(x, P, totalPop)
    cost = x.T @ costs(minwage=minwage, rent=rent, demand=demand_val, P=P, IR=IR) 
    rev = revenue(demand_val, P, IR)
    return [
        {'type': 'ineq', 'fun': lambda _: budget - cost},  # Budget >= x@costs
        {'type': 'ineq', 'fun': lambda _: N - sum(x)},  # N >= sum(x)
        {'type': 'ineq', 'fun': lambda _: risk - cost / rev},  # risk > cost/revenue
        {'type': 'ineq', 'fun': lambda _: x},  # All x >= 0
        {'type': 'ineq', 'fun': lambda _: P},  # All P >= 0
    ]


def optimize(totalPop, IR, budget, risk, N, minwage, rent):
    '''
    Optimizes the objective function given certain variables.

    Args:
        x0: Initial guess for optimization
        P: Price
        totalPop: Total population in county
        IR: Income Ratio relative to maximum county income
        budget (int): Total budget for owning restaurants.
        N (int): Maximum number of stores to open.
        risk (float): Total acceptable risk ratio per location (cost/revenue)
        minwage: Minimum wage in county
        rent: Rent of restaurant
        
    Returns:
        result: Result of optimization function.
    '''
    x0 = np.ones((NUMBER_OF_COUNTIES, 1))  # Initial guess for optimization
    P = 18 * np.ones((NUMBER_OF_COUNTIES, 1))  # Initial guess for price
    x0 = np.concatenate((x0, P), axis=0)  # Concatenate x and P  
    constraints = getConstraints(x0, budget, N, risk, totalPop, IR, minwage, rent)
    result = minimize(
        fun = objectiveFunction, 
        x0=x0, 
        args=(totalPop, IR, minwage, rent), 
        constraints=constraints
    )
    return result