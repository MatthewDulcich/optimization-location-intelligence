import numpy as np


def profit(x, P, totalPop, IR, minwage, rent, initialCosts):
    '''
    Calculates Profit = Revenue - Costs

    args:
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county
        IR: Income Ratio relative to maximum county income
        minwage: Minimum wage in county
        rent: Rent of restaurant
        initialCosts: Initial costs of opening restaurant

    returns:
        profit: Profit of restaurant
    '''
    demand_val = demand(x, P, totalPop)
    rev = revenue(demand_val, P, IR)
    cost = costs(minwage, rent, demand_val, P, initialCosts)
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


def costs(minwage, rent, demand, P, initialCosts, employees=15):
    '''
    Calculates costs of restaurant.

    args:
        minwage: Minimum wage in county
        rent: Rent of restaurant
        demand: Demand of restaurant
        P: Price of restaurant
IR: Income ratio of county to maximum county income. Between 0 and 1.
        initialCosts: Initial costs of opening restaurant
        employees: Number of employees in restaurant (default is 15)
    
    returns:
        netCosts: Total costs of restaurant
    '''
    loss_incurred = loss(demand, P)
    operating_costs = minwage * employees + rent + loss_incurred
    return operating_costs + initialCosts


def loss(demand, P, IR):
    '''
    Calculates losses: 8.2% of revenue incurred by restaurant.

    args:
        demand: Demand of restaurant
        IR: Income ratio of county to maximum county income. Between 0 and 1.
        P: Price of restaurant

    returns:
        revenue(oss: Los, P, IR)
    '''
    return 0.082 * revenue(demand, P, IR)


# ######################################################
# Inequality constraints >= 0
# ######################################################


def max_risk(cost, revenue, risk=0.8):
    '''
    Calculates maximum risk of restaurant.

    args:
        cost: Total costs of restaurant
        revenue: Total revenue of restaurant
        risk: Acceptable risk threshold (default is 0.8)

    returns:
        maxRisk: Maximum risk of restaurant
    '''
    return risk - cost / revenue


def store_possibility(x):
    '''
    Calculates if store is possible.

    args:
        x: Total number of stores in given county

    returns:
        storePossibility: True if store is possible, False otherwise
    '''
    return x


def max_budget(budget, cost):
    '''
    Calculates maximum budget of restaurant.

    args:
        budget: Total budget of restaurant
        cost: Total costs of restaurant

    returns:
        maxBudget: Maximum budget of restaurant
    '''
    return budget - sum(cost)


def max_stores(N, x):
    '''
    Calculates maximum number of stores in given county.

    args:
        N: Maximum number of stores to open
        x: Total number of stores in given county

    returns:
        maxStores: Remaining number of stores that can be opened
    '''
    return N - sum(x)


# ######################################################
# Objective function
# ######################################################


def objectiveFunction(x, P, totalPop, IR):
    '''
    Objective function for optimization problem.

    args:
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county
        IR: Income Ratio relative to maximum county income

    returns:
        Objective value to minimize
    '''
    return -x.T @ profit(x, P, totalPop, IR)


def getConstraints(budget, N, risk, x, P, totalPop, IR):
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

    Returns:
        constraints: List of constraints to use in optimization function.
    '''
    demand_val = demand(x, P, totalPop)
    cost = costs(minwage=0, rent=0, demand=demand_val, P=P, initialCosts=0)  # Replace with actual values
    rev = revenue(demand_val, P, IR)
    return [
        {'type': 'ineq', 'fun': lambda _: budget - sum(cost)},  # Budget >= x@costs
        {'type': 'ineq', 'fun': lambda _: N - sum(x)},  # N >= sum(x)
        {'type': 'ineq', 'fun': lambda _: risk - cost / rev},  # risk > cost/revenue
        {'type': 'ineq', 'fun': lambda _: x}  # All x >= 0
    ]