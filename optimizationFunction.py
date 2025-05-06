import numpy as np
import pandas as pd
from gradient_descent import gd_minimize

'''
    Since there are many variables in the optimization function,
    we will use an order.
    The order is as follows:
    1. Variables: x, P
    2. Integers: budget, N
    3. Floats: risk
    4. Dataframes: totalPop, IR, minwage, rent
    5. Rest: demand, revenue, costs, loss
'''

#NUMBER_OF_COUNTIES = 3143  # Number of counties
NUMBER_OF_COUNTIES = 100  # Number of counties

# ######################################################
# Main functions
# ######################################################


def profit(x, P, totalPop, IR, minwage, rent):
    '''
    Calculates Profit = Revenue - Costs.

    Args:
        x: Total number of stores in given county.
        P: Price.
        totalPop: Total population in county.
        IR: Income Ratio relative to maximum county income.
        minwage: Minimum wage in county.
        rent: Rent of restaurant.

    Returns:
        profit: Profit of restaurant.
    '''
    # Use numpy arrays for better memory efficiency
    x = np.asarray(x)
    P = np.asarray(P)
    totalPop = np.asarray(totalPop)
    IR = np.asarray(IR)
    minwage = np.asarray(minwage)
    rent = np.asarray(rent)
    
    demand_val = demand(x, P, totalPop)
    rev = revenue(P, IR, demand_val)
    cost = costs(P, IR, minwage, rent, demand_val)
    
    # Clean up intermediate arrays
    del demand_val
    return rev - cost


def revenue(P, IR, demand):
    '''
    Calculates Revenue of Product.

    Args:
        P: Price of a given product.
        IR: Income ratio of county to maximum county income (0 to 1).
        demand: The demand of a given product.

    Returns:
        revenue: Demand times Price.
    '''
    return P**(IR**0.5) * demand


def log_revenue(P, IR, log_demand):
    '''
    Calculates log of Revenue of Product (concave function).

    Args:
        P: Price of a given product.
        IR: Income ratio of county to maximum county income (0 to 1).
        log_demand: Log of demand of a given product.

    Returns:
        log_revenue: Log of Revenue.
    '''
    return log_demand + IR * np.log(P)


def demand(x, P, totalPop):
    '''
    Calculates demand as a function of price.

    Args:
        x: Total number of stores in given county.
        P: Price.
        totalPop: Total population in county.

    Returns:
        demand: Demand corresponding to input variables.
    '''
    print(f"x: {x:<20} P: {P:<20} totalPop: {totalPop:<10}")  # Debugging
    a = 0.003 * P
    # L: Maximum demand at unit price 1.0.
    if x >= 1:
        L = totalPop / x
    else:
        L = 0 # No demand with no stores
    if P > 50:
        L = 0

    return L * np.exp(-a * P)


def log_demand(x, P, totalPop):
    '''
    Calculates log of demand as a function of price (concave function).

    Args:
        x: Total number of stores in given county.
        P: Price.
        totalPop: Total population in county.

    Returns:
        log_demand: Log of demand corresponding to input variables.
    '''
    a = 0.003 * P
    L = totalPop / x  # Maximum demand at unit price 1.0.
    return np.log(L) - a * P


def costs(P, IR, minwage, rent, demand, employees=15):
    '''
    Calculates costs of restaurant.

    Args:
        minwage: Minimum wage in county.
        rent: Rent of restaurant.
        demand: Demand of restaurant.
        P: Price of restaurant.
        IR: Income ratio of county to maximum county income (0 to 1).
        employees: Number of employees in restaurant (default is 15).

    Returns:
        netCosts: Total costs of restaurant.
    '''
    loss_incurred = loss(P, IR, demand)
    operating_costs = minwage * employees + rent + loss_incurred
    return operating_costs


def loss(P, IR, demand):
    '''
    Calculates losses: 8.2% of revenue incurred by restaurant.

    Args:
        P: Price of restaurant.
        IR: Income ratio of county to maximum county income (0 to 1).
        demand: Demand of restaurant.

    Returns:
        loss: Losses incurred by restaurant.
    '''
    return 0.082 * revenue(P, IR, demand)


# ######################################################
# Objective function
# ######################################################


def objectiveFunction(x, totalPop, IR, minwage, rent):
    '''
    Objective function for optimization problem.

    Args:
        x: Total number of stores + Price per store.
        totalPop: Total population in county.
        IR: Income Ratio relative to maximum county income.
        minwage: Minimum wage in county.
        rent: Rent of restaurant.

    Returns:
        Objective value to minimize.
    '''
    # Split x into stores and prices
    P = x[NUMBER_OF_COUNTIES:]
    x = x[:NUMBER_OF_COUNTIES]
    
    # Convert pandas Series to numpy arrays to ensure consistent indexing
    totalPop = np.asarray(totalPop)
    IR = np.asarray(IR)
    minwage = np.asarray(minwage)
    rent = np.asarray(rent)
    
    # Use numpy arrays for better memory efficiency
    total = -sum(x[i] * profit(x[i], P[i], totalPop[i], IR[i], minwage[i], rent[i]) 
                for i in range(len(x)))
    
    return total / 100000


def getConstraints(x, budget, N, risk, totalPop, IR, minwage, rent):
    '''
    Creates constraints for optimization function given certain variables.

    Args:
        x: Total number of stores + Price per county.
        budget (int): Total budget for owning restaurants.
        N (int): Maximum number of stores to open.
        risk (float): Total acceptable risk ratio per location (cost/revenue).
        totalPop: Total population in county.
        IR: Income Ratio relative to maximum county income.
        minwage: Minimum wage in county.
        rent: Rent of restaurant.

    Returns:
        constraints: List of constraints to use in optimization function.
    '''
    # Convert pandas Series to numpy arrays
    totalPop = np.asarray(totalPop)
    IR = np.asarray(IR)
    minwage = np.asarray(minwage)
    rent = np.asarray(rent)
    
    def budget_constraint(x):
        P = x[NUMBER_OF_COUNTIES:]
        x = x[:NUMBER_OF_COUNTIES]
        
        # Use generator expression for memory efficiency
        demand_val = (demand(i,j,k) for i,j,k in zip(x, P, totalPop))
        cost = (costs(p, ir, mw, r, d) 
                for p,ir,mw,r,d in zip(P,IR,minwage,rent,demand_val))
        total_cost = sum(x[i] * c for i,c in enumerate(cost))
        
        return budget - total_cost

    def total_stores_constraint(x):
        return N - sum(x[:NUMBER_OF_COUNTIES])

    def risk_constraint(x):
        P = x[NUMBER_OF_COUNTIES:]
        x = x[:NUMBER_OF_COUNTIES]
        
        # Use generator expressions
        demand_val = (demand(i,j,k) for i,j,k in zip(x, P, totalPop))
        cost = (costs(p, ir, mw, r, d) 
                for p,ir,mw,r,d in zip(P,IR,minwage,rent,demand_val))
        rev = (revenue(p, ir, d) 
               for p, ir, d in zip(P, IR, demand_val))
        
        return [(risk - c/r) if r != 0 else -1 
                for c, r in zip(cost, rev)]

    return [
        {'type': 'ineq', 'fun': lambda x: budget_constraint(x)},
        {'type': 'eq', 'fun': lambda x: total_stores_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: risk_constraint(x)},
    ]


def optimize(budget, N, risk, totalPop, IR, minwage, rent):
    '''
    Optimizes the objective function given certain variables.

    Args:
        totalPop: Total population in county.
        IR: Income Ratio relative to maximum county income.
        budget (int): Total budget for owning restaurants.
        risk (float): Total acceptable risk ratio per location (cost/revenue).
        N (int): Maximum number of stores to open.
        minwage: Minimum wage in county.
        rent: Rent of restaurant.

    Returns:
        result: Result of optimization function.
    '''

    x0 = np.zeros(NUMBER_OF_COUNTIES)
    x0[:N] = 1
    P = 18 * np.ones(NUMBER_OF_COUNTIES)
    x0 = np.concatenate((x0,P))

    constraints = getConstraints(x0, budget, N, risk, totalPop, IR, minwage, rent)

    # Bounds for number of stores and price in each county. Must be positive. Price < 50
    bounds = [(0,100) for i in range(NUMBER_OF_COUNTIES)] + \
             [(0,50) for i in range(NUMBER_OF_COUNTIES)]

    result = gd_minimize(
        fun=objectiveFunction,
        x0=x0,
        args=(totalPop, IR, minwage, rent),
        constraints=constraints,
        bounds=bounds,
        options = {'maxiter':1000, 'disp': True} # Display optimization process
    )
    return result
