import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum
from pyscipopt import Expr

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
# SCIP Optimization
# ######################################################

def optimize(budget, N, risk, totalPop, IR, minwage, rent, NRestaurants):
    print("Starting optimization process...")

    # Convert all inputs to NumPy arrays to ensure compatibility
    totalPop = np.array(totalPop)
    IR = np.array(IR)
    minwage = np.array(minwage)
    rent = np.array(rent)
    NRestaurants = np.array(NRestaurants)

    # Number of counties
    n_counties = len(totalPop)
    print(f"Number of counties: {n_counties}")

    # Create a SCIP model
    model = Model("Restaurant Optimization")

    # Enable verbose output
    model.setIntParam("display/verblevel", 5)
    model.redirectOutput()  # Redirect SCIP output to the console

    # Define variables
    x = {}  # Integer variables for units sold in each county
    P = {}  # Continuous variables for price in each county
    z = {}  # Binary variables for whether a county is served

    for i in range(n_counties):
        x[i] = model.addVar(vtype="I", name=f"x_{i}", lb=0)  # Integer variable
        P[i] = model.addVar(vtype="C", name=f"P_{i}", lb=0, ub=100)  # Continuous variable with upper bound
        z[i] = model.addVar(vtype="B", name=f"z_{i}")        # Binary variable

    print("Variables defined.")

    # Add constraints
    print("Adding constraints...")
    total_cost_expr = 0  # Initialize total cost expression
    for i in range(n_counties):
        # Define linear revenue approximation for constraints
        L = totalPop[i] * 0.25 * 12
        revenue_expr = 0.9 * L * P[i]
        loss_expr = 0.082 * revenue_expr

        # Define cost expression for constraints using linear approximation
        operating_costs_expr = minwage[i] * (40 * 52) * 15 + rent[i] + loss_expr

        # Accumulate total cost
        total_cost_expr += operating_costs_expr

        # Population-based cap
        model.addCons(x[i] <= totalPop[i] * 0.1)

        # Absolute max per region
        model.addCons(x[i] <= 20)

        # Link x and z: if z[i] == 0, x[i] must be 0
        model.addCons(x[i] <= z[i] * 20)

        # Link P and z: if z[i] == 0, P[i] must be 0
        model.addCons(P[i] <= z[i] * 100)

    # Add total cost constraint outside the loop
    model.addCons(total_cost_expr <= budget)
    print("Constraints added.")

    # Objective: maximize profit
    print("Defining objective function...")

    profit_terms = []
    for i in range(n_counties):
        L = totalPop[i] * 0.25 * 12
        # Linear revenue approximation
        revenue_expr = 0.9 * L * P[i]
        loss_expr = 0.082 * revenue_expr
        operating_costs_expr = minwage[i] * (40 * 52) * 15 + rent[i] + loss_expr
        profit_expr = revenue_expr - operating_costs_expr
        profit_terms.append(profit_expr)

    profit_expr_total = quicksum(profit_terms)
    model.setObjective(profit_expr_total, "maximize")
    print("Objective function defined.")

    # Solve the problem
    print("Starting SCIP optimization...")
    model.optimize()

    # Extract results
    if model.getStatus() == "optimal":
        print("Optimal solution found.")
        optimal_x = [model.getVal(x[i]) for i in range(n_counties)]
        optimal_P = [model.getVal(P[i]) for i in range(n_counties)]
        optimal_z = [model.getVal(z[i]) for i in range(n_counties)]
        total_profit = model.getObjVal()
    else:
        print("No optimal solution found.")
        optimal_x = None
        optimal_P = None
        optimal_z = None
        total_profit = None

    # Debugging output
    print(f"Solver status: {model.getStatus()}")
    if optimal_x is not None:
        print(f"Optimal x (first 10): {optimal_x[:10]}")
        print(f"Optimal P (first 10): {optimal_P[:10]}")
        print(f"Optimal z (first 10): {optimal_z[:10]}")
        print(f"Total profit: {total_profit}")

    # Return results
    return {
        'optimal_x': optimal_x,
        'optimal_P': optimal_P,
        'optimal_z': optimal_z,
        'profit': total_profit
    }
