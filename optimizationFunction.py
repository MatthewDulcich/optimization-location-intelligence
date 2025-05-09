import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import tensorflow as tf
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpAffineExpression

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

# NUMBER_OF_COUNTIES = 3143  # Number of counties
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
    demand_val = demand(x, P, totalPop)
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
    P = x[NUMBER_OF_COUNTIES:]
    x = x[:NUMBER_OF_COUNTIES]
    #return -x.T @ profit(x, P, totalPop, IR, minwage, rent)
    total = -x.T @ [profit(x, P, Pop, IR, mw, r) for x,P,Pop,IR,mw,r in zip(x,P,totalPop,IR,minwage,rent)]
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

    def budget_constraint(x):
        P = x[NUMBER_OF_COUNTIES:]
        x = x[:NUMBER_OF_COUNTIES]

        demand_val = [demand(i,j,k) for i,j,k in zip(x, P, totalPop)]
        cost = [costs(P, IR, mw, r, d) for P,IR,mw,r,d in zip(P,IR,minwage,rent,demand_val)]
        total_cost = x.T @ cost # Get total costs for budget constraint

        return budget - total_cost

    def total_stores_constraint(x):
        x = x[:NUMBER_OF_COUNTIES]
        return N - sum(x)

    def risk_constraint(x):
        P = x[NUMBER_OF_COUNTIES:]
        x = x[:NUMBER_OF_COUNTIES]

        demand_val = [demand(i,j,k) for i,j,k in zip(x, P, totalPop)]
        cost = [costs(P, IR, mw, r, d) for P,IR,mw,r,d in zip(P,IR,minwage,rent,demand_val)]
        rev = [revenue(P, IR, d) for P, IR, d in zip(P, IR, demand_val)] # Get revenue for risk constraint

        return [(risk - c/r) if r != 0 else -1 for c, r in zip(cost,rev)]

    return [
        {'type': 'ineq', 'fun': lambda x: budget_constraint(x)}, # Budget >= total cost (make 1d scaler)
        {'type': 'eq', 'fun': lambda x: total_stores_constraint(x)}, # N >= sum(x)
        {'type': 'ineq', 'fun': lambda x: risk_constraint(x)}, # risk > cost/revenue
        #{'type': 'ineq', 'fun': lambda x: x}, # All x, P > 0
    ]
    #return [
    #    {'type': 'ineq', 'fun': lambda _: budget - total_cost},  # Budget >= total cost (make 1d scaler)
    #    {'type': 'ineq', 'fun': lambda _: N - sum(x)},  # N >= sum(x)
    #    {'type': 'ineq', 'fun': lambda _: risk_constraint},  # risk > cost/revenue
    #    {'type': 'ineq', 'fun': lambda _: x},  # All x >= 0
    #    {'type': 'ineq', 'fun': lambda _: P},  # All P >= 0
    #]


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
    x0 = np.concatenate((x0, P))

    constraints = getConstraints(x0, budget, N, risk, totalPop, IR, minwage, rent)

    # Bounds for number of stores and price in each county. Must be positive. Price < 50
    bounds = [(0, 100) for i in range(NUMBER_OF_COUNTIES)] + \
             [(0, 50) for i in range(NUMBER_OF_COUNTIES)]

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=500, desc="Optimization Progress", unit="iteration")

    # Define a callback function to update tqdm
    def callback(xk):
        progress_bar.update(1)

    # Run the optimization
    result = minimize(
        fun=objectiveFunction,
        x0=x0,
        args=(totalPop, IR, minwage, rent),
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 500},
        callback=callback  # Pass the callback function
    )

    # Close the progress bar
    progress_bar.close()

    return result


def objective_function_tf(x, totalPop, IR, minwage, rent):
    '''
    Objective function for optimization problem using TensorFlow.

    Args:
        x: Tensor containing decision variables (number of stores and prices).
        totalPop: Tensor of total population in each county.
        IR: Tensor of income ratios for each county.
        minwage: Tensor of minimum wages in each county.
        rent: Tensor of rent costs in each county.

    Returns:
        Objective value to minimize (negative profit).
    '''
    NUMBER_OF_COUNTIES = totalPop.shape[0]
    P = x[NUMBER_OF_COUNTIES:]  # Prices
    x = x[:NUMBER_OF_COUNTIES]  # Number of stores

    # Calculate demand
    demand_val = tf.exp(-0.003 * P) * (totalPop / (x + 1e-6))

    # Debug: Check for invalid values in demand
    tf.debugging.assert_all_finite(demand_val, "Demand contains NaN or Inf")

    # Calculate revenue and costs
    revenue_val = P * demand_val * tf.sqrt(IR)
    cost_val = minwage * 15 + rent + 0.082 * revenue_val
    profit = revenue_val - cost_val

    # Debug: Check for invalid values in profit
    tf.debugging.assert_all_finite(profit, "Profit contains NaN or Inf")

    # Return negative profit (since we want to maximize profit)
    return -tf.reduce_sum(profit)


def optimize_tf(budget, N, risk, totalPop, IR, minwage, rent):
    NUMBER_OF_COUNTIES = totalPop.shape[0]

    # Initialize decision variables (number of stores and prices)
    x0 = tf.Variable(tf.concat([
        tf.ones(NUMBER_OF_COUNTIES),  # Start with 1 store per county
        tf.ones(NUMBER_OF_COUNTIES) * 20  # Start with a price of $20
    ], axis=0), dtype=tf.float32)

    # Define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.001)  # Smaller learning rate

    # Optimization loop
    for step in range(500):  # Number of iterations
        with tf.GradientTape() as tape:
            loss = objective_function_tf(x0, totalPop, IR, minwage, rent)

        # Check for NaN in loss
        if tf.math.is_nan(loss):
            print(f"Loss became NaN at step {step}. Stopping optimization.")
            break

        # Compute gradients and apply them
        grads = tape.gradient(loss, [x0])
        optimizer.apply_gradients(zip(grads, [x0]))

        # Enforce non-negativity and a small lower bound
        x0.assign(tf.maximum(x0, 1e-3))

        # Print progress every 50 steps
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

    # Extract optimized values
    optimized_x = x0.numpy()
    optimized_profit = -objective_function_tf(x0, totalPop, IR, minwage, rent).numpy()

    return optimized_x, optimized_profit


def optimize_integer(budget, N, risk, totalPop, IR, minwage, rent):
    '''
    Optimizes the objective function using integer programming.

    Args:
        budget: Total budget for owning restaurants.
        N: Maximum number of stores to open.
        risk: Total acceptable risk ratio per location (cost/revenue).
        totalPop: Total population in each county.
        IR: Income Ratio relative to maximum county income.
        minwage: Minimum wage in each county.
        rent: Rent of restaurant in each county.

    Returns:
        result: Optimized decision variables and profit.
    '''
    NUMBER_OF_COUNTIES = len(totalPop)

    # Create the optimization problem
    prob = LpProblem("Restaurant_Optimization", LpMaximize)

    # Define decision variables
    x = [LpVariable(f"x_{i}", lowBound=1, upBound=100, cat="Integer") for i in range(NUMBER_OF_COUNTIES)]
    P = [LpVariable(f"P_{i}", lowBound=0, upBound=50) for i in range(NUMBER_OF_COUNTIES)]
    demand = [LpVariable(f"demand_{i}", lowBound=0) for i in range(NUMBER_OF_COUNTIES)]  # Demand variable
    revenue = [LpVariable(f"revenue_{i}", lowBound=0) for i in range(NUMBER_OF_COUNTIES)]  # Revenue variable

    # Objective function: Maximize profit
    profit_terms = []
    for i in range(NUMBER_OF_COUNTIES):
        cost = minwage[i] * 15 + rent[i] + 0.082 * revenue[i]
        profit_terms.append(revenue[i] - cost)

    prob += lpSum(profit_terms), "Total_Profit"

    # Constraints for demand
    for i in range(NUMBER_OF_COUNTIES):
        prob += demand[i] <= totalPop[i] / 1.0, f"MaxDemandConstraint_{i}"  # Ensure demand does not exceed population

    # Constraints for revenue (linearize P[i] * demand[i])
    for i in range(NUMBER_OF_COUNTIES):
        prob += revenue[i] <= P[i] * totalPop[i], f"RevenueUpperBound_{i}"
        prob += revenue[i] <= demand[i] * 50, f"RevenueDemandBound_{i}"

    # Budget constraint
    cost_terms = [
        minwage[i] * 15 + rent[i] + 0.082 * revenue[i]
        for i in range(NUMBER_OF_COUNTIES)
    ]
    prob += lpSum(cost_terms) <= budget, "BudgetConstraint"

    # Total stores constraint
    prob += lpSum(x) <= N, "TotalStoresConstraint"

    # Risk constraint
    for i in range(NUMBER_OF_COUNTIES):
        prob += revenue[i] >= minwage[i] * 15 + rent[i] + 0.082 * revenue[i], f"RiskConstraint_{i}"

    # Solve the problem
    prob.solve()

    # Extract results
    x_values = [var.value() for var in x]
    P_values = [var.value() for var in P]
    total_profit = prob.objective.value()

    return {
        "x": x_values,
        "P": P_values,
        "profit": total_profit
    }
