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
    max_demand = totalPop * 0.1  # Allow 10% of the population as demand
    constraints.append(x <= cp.multiply(max_demand, z))

    # Limit the number of restaurants per county
    max_restaurants_per_county = 20  # Increase the limit
    constraints.append(x <= cp.multiply(max_restaurants_per_county, z))  # Use cp.multiply

    # Budget constraint: total cost (variable + fixed) cannot exceed budget
    variable_cost = cp.multiply(minwage, x) + rent
    fixed_cost = cp.multiply(NRestaurants, z)
    total_cost = cp.sum(variable_cost) + cp.sum(fixed_cost)
    constraints.append(total_cost <= budget)

    # Objective: maximize total profit
    a = totalPop * 0.25 * 365  # Vectorized demand intercept
    b = 0.003  # Price sensitivity
    revenue = cp.multiply(a / b, x) - cp.multiply(1 / b, cp.square(x))  # Vectorized revenue
    profit = revenue - (variable_cost + fixed_cost + cp.multiply(1000, x))  # Vectorized profit
    objective = cp.Maximize(cp.sum(profit))

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem using ECOS_BB
    problem.solve(
        solver=cp.ECOS_BB,
        abstol=1e-6,
        reltol=1e-6,
        feastol=1e-6,
        max_iters=10000
    )

    # Fallback: Round to nearest integer to enforce integrality
    if x.value is not None:
        optimal_x = np.round(x.value).astype(int)
    else:
        optimal_x = None

    optimal_z = z.value

    # Debug prints to verify solver output
    print("Solver status:", problem.status)
    if x.value is not None:
        print("Max optimized x[i]:", np.max(x.value))
        print("First 10 optimized x:", x.value[:10])
    else:
        print("No solution returned for x; solver status:", problem.status)

    # Debugging constraints
    print("Debugging constraints:")
    for i, constraint in enumerate(constraints):
        print(f"Constraint {i}: {constraint.value if constraint.value is not None else 'Not computed'}")

    # Debugging total cost
    print("Total cost (variable + fixed):", total_cost.value if total_cost.value is not None else "Not computed")
    print("Budget:", budget)

    # Return the integer-rounded solution
    return {
        'optimal_x': optimal_x,
        'optimal_z': optimal_z,
        'profit': problem.value
    }
