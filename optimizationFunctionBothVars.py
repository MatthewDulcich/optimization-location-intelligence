import numpy as np
import pandas as pd
import cvxpy as cp

'''
    Since there are many variables in the optimization function,
    we will use an order.
    The order is as follows:
    1. Variables: x, z
    2. Integers: budget, N
    3. Floats: risk
    4. Dataframes: totalPop, IR, minwage, rent
    5. Rest: demand, revenue, costs, loss
'''

NUMBER_OF_COUNTIES = 3143  # Number of counties
# NUMBER_OF_COUNTIES = 100  # Number of counties

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
    demand_val = demand(x, P, totalPop, NRestaurants)
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


def demand(x, P, totalPop, NRestaurants):
    '''
    Calculates demand as a function of price.

    Args:
        x: Total number of stores in given county.
        P: Price.
        totalPop: Total population in county.

    Returns:
        demand: Demand corresponding to input variables.
    '''
    a = 0.003 * P
    if x >= 1:
        L = totalPop * 0.25 * 12 / x
    else:
        L = 0
    return L * np.exp(-a * P)


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
    operating_costs = minwage * (40 * 52) * employees + rent + loss_incurred
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
# CVXPY Optimization
# ######################################################

def optimize(budget, N, risk, totalPop, IR, minwage, rent, NRestaurants):
    '''
    Optimizes the objective function using CVXPY.

    Args:
        totalPop: Total population in county (NumPy array or list).
        IR: Income Ratio relative to maximum county income (NumPy array or list).
        budget (int): Total budget for owning restaurants.
        risk (float): Total acceptable risk ratio per location (cost/revenue).
        N (int): Maximum number of stores to open.
        minwage: Minimum wage in county (NumPy array or list).
        rent: Rent of restaurant (NumPy array or list).
        NRestaurants: Number of existing restaurants in each county (NumPy array or list).

    Returns:
        result: Result of optimization function.
    '''

    # Convert all inputs to NumPy arrays to ensure compatibility
    totalPop = np.array(totalPop)
    IR = np.array(IR)
    minwage = np.array(minwage)
    rent = np.array(rent)
    NRestaurants = np.array(NRestaurants)

    # Number of counties
    n_counties = len(totalPop)

    # Define variables
    x = cp.Variable(n_counties, integer=True)  # Units sold in each county (integer)
    z = cp.Variable(n_counties, boolean=True)  # Binary indicator for serving each county

    # Constraints
    constraints = []

    # Nonnegativity constraint for x
    constraints.append(x >= 0)

    # Demand limit: if county i is served, x[i] <= demand[i]; if not served (z[i] = 0), x[i] = 0
    for i in range(n_counties):
        max_demand = totalPop[i] * 0.25 * 12  # Example demand cap
        constraints.append(x[i] <= max_demand * z[i])  # Big-M constraint linking x and z

    # Budget constraint: total cost (variable + fixed) cannot exceed budget
    variable_cost = cp.multiply(minwage, x) + rent
    fixed_cost = cp.multiply(NRestaurants, z)
    total_cost = cp.sum(variable_cost) + cp.sum(fixed_cost)
    constraints.append(total_cost <= budget)

    # Limit the number of stores
    constraints.append(cp.sum(z) <= N)

    # Objective: maximize total profit
    profit_terms = []
    for i in range(n_counties):
        a_i = totalPop[i] * 0.25 * 12  # Example demand intercept
        b_i = 0.003  # Example price sensitivity
        # Concave revenue term: (a_i / b_i) * x[i] - (1 / b_i) * cp.square(x[i])
        revenue_i = (a_i / b_i) * x[i] - (1 / b_i) * cp.square(x[i])
        # Subtract costs: variable_cost[i] + fixed_cost[i]
        profit_i = revenue_i - variable_cost[i] - fixed_cost[i]
        profit_terms.append(profit_i)

    # Maximize total profit
    objective = cp.Maximize(cp.sum(profit_terms))

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem using ECOS_BB
    problem.solve(solver=cp.ECOS_BB)

    # Get the results
    optimal_x = x.value  # Optimal number of units sold
    optimal_z = z.value  # Optimal binary decisions

    return {
        'optimal_x': optimal_x,
        'optimal_z': optimal_z,
        'profit': problem.value
    }
