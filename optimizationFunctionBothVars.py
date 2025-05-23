import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def profit(x, P, totalPop, IR, minwage, rent, NRestaurants):
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

    demand_val = demand(x, P, totalPop)#, NRestaurants)
    rev = revenue(P, IR, demand_val)
    cost = costs(P, IR, minwage, rent, demand_val)
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


def demand(x, P, totalPop):#, Nrestaurants):
    '''
    Calculates demand as a function of price.

    Args:
        x: Total number of stores in given county.
        P: Price.
        totalPop: Total population in county.

    Returns:
        demand: Demand corresponding to input variables.
    '''
    a = 0.001 * P

    #L = totalPop*(0.075*12)
    #L = L - L * (1/Nrestaurants+x)*1000 * np.log(((1/totalPop)*10*x**2+1))
    # L: Maximum demand at unit price 1.0.
    #if x >= 1:
    if x >= 1:
        #L = totalPop*.25*12 / x
        L = totalPop*.25*365 / x
    else:
        L = 0
    #else:
    #    L = 0 # No demand with no stores
    #if P > 50:
    #    L = 0

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
    operating_costs = minwage * (40*52) * employees + rent + loss_incurred
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


def objectiveFunction(x, totalPop, IR, minwage, rent, NRestaurants, nvars):
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
    P = x[int(nvars/2):]
    x = x[:int(nvars/2)]
    #return -x.T @ profit(x, P, totalPop, IR, minwage, rent)
    total = -x.T @ [profit(x, P, Pop, IR, mw, r, NR) for x,P,Pop,IR,mw,r,NR in zip(x,P,totalPop,IR,minwage,rent,NRestaurants)]
    return total / 100000


def getConstraints(budget, N, risk, totalPop, IR, minwage, rent, nvars):
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
    #P = x[NUMBER_OF_COUNTIES:].squeeze() # second half of input variables
    #x = x[:NUMBER_OF_COUNTIES].squeeze() # first half of input variables

    #demand_val = demand(x, P, totalPop)
    #cost = x.T @ costs(P=P, IR=IR, minwage=minwage, rent=rent, demand=demand_val)
    #rev = revenue(P, IR, demand_val)

    #demand_val = [demand(i,j,k) for i,j,k in zip(x, P, totalPop)]
    #cost = [costs(P, IR, mw, r, d) for P,IR,mw,r,d in zip(P,IR,minwage,rent,demand_val)]
    #total_cost = x.T @ cost # Get total costs for budget constraint
    #rev = [revenue(P, IR, d) for P, IR, d in zip(P, IR, demand_val)] # Get revenue for risk constraint
    #risk_constraint = [(risk - c/r) for c, r in zip(cost,rev)] # Risk - cost / revenue > 0

    def budget_constraint(x,nvars):
        P = x[int(nvars/2):]
        x = x[:int(nvars/2)]

        demand_val = [demand(i,j,k) for i,j,k in zip(x, P, totalPop)]
        cost = [costs(P, IR, mw, r, d) for P,IR,mw,r,d in zip(P,IR,minwage,rent,demand_val)]
        total_cost = x.T @ cost # Get total costs for budget constraint

        return budget - total_cost / 1000000

    def total_stores_constraint(x,nvars):
        x = x[:int(nvars/2)]
        return N - sum(x)

    def risk_constraint(x):
        P = x[NUMBER_OF_COUNTIES:]
        x = x[:NUMBER_OF_COUNTIES]

        demand_val = [demand(i,j,k) for i,j,k in zip(x, P, totalPop)]
        cost = [costs(P, IR, mw, r, d) for P,IR,mw,r,d in zip(P,IR,minwage,rent,demand_val)]
        rev = [revenue(P, IR, d) for P, IR, d in zip(P, IR, demand_val)] # Get revenue for risk constraint

        return [(risk - c/r) if r != 0 else -1 for c, r in zip(cost,rev)]

    def int_constraint(x,nvars):
        x = x[:int(nvars/2)]
        return max(int(x) - x)

    return [
        {'type': 'ineq', 'fun': lambda x: budget_constraint(x,nvars)}, # Budget >= total cost (make 1d scaler)
        {'type': 'eq', 'fun': lambda x: total_stores_constraint(x,nvars)}, # N >= sum(x)
        #{'type': 'ineq', 'fun': lambda x: risk_constraint(x)}, # risk > cost/revenue
        #{'type': 'ineq', 'fun': lambda x: x}, # All x, P > 0
    ]
    #return [
    #    {'type': 'ineq', 'fun': lambda _: budget - total_cost},  # Budget >= total cost (make 1d scaler)
    #    {'type': 'ineq', 'fun': lambda _: N - sum(x)},  # N >= sum(x)
    #    {'type': 'ineq', 'fun': lambda _: risk_constraint},  # risk > cost/revenue
    #    {'type': 'ineq', 'fun': lambda _: x},  # All x >= 0
    #    {'type': 'ineq', 'fun': lambda _: P},  # All P >= 0
    #]


def optimize(budget, N, risk, totalPop, IR, minwage, rent, NRestaurants, nvars):
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

    x0 = np.ones(int(nvars/2))
    x0[:N] = 1 # Start with N stores with cheapest rent for budget constraint
    P = 18 * np.ones(int(nvars/2))
    x0 = np.concatenate((x0,P))

    constraints = getConstraints(budget, N, risk, totalPop, IR, minwage, rent, nvars)

    # Bounds for number of stores and price in each county. Must be positive. Price < 50
    bounds = [(0,10) for i in range(int(nvars/2))] + \
             [(0,30) for i in range(int(nvars/2))]

    result = minimize(
        fun=objectiveFunction,
        x0=x0,
        args=(totalPop, IR, minwage, rent, NRestaurants, nvars),
        constraints=constraints,
        bounds=bounds,
        options = {'maxiter':500},
        method = 'SLSQP'
    )
    return result
