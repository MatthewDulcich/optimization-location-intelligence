import numpy as np
from gekko import GEKKO

def optimize(budget, N, risk, totalPop, IR, minwage, rent, NRestaurants):
    print("Starting GEKKO optimization...")

    # Convert inputs to NumPy arrays
    totalPop = np.array(totalPop)
    IR = np.array(IR)
    minwage = np.array(minwage)
    rent = np.array(rent)
    NRestaurants = np.array(NRestaurants)

    n_counties = len(totalPop)
    print(f"Number of counties: {n_counties}")

    # Initialize GEKKO model
    m = GEKKO(remote=False)
    print("GEKKO model initialized.")

    # Set maximum iterations
    m.options.MAX_ITER = 10000

    # Variables
    print("Defining variables...")
    x = [m.Var(lb=0, ub=20) for _ in range(n_counties)]  # NumStores
    P = [m.Var(lb=5, ub=100) for _ in range(n_counties)]               # Set a more realistic lower bound
    print(f"Defined {len(x)} variables for NumStores.")
    print(f"Defined {len(P)} continuous variables for Price.")

    # Parameters
    employees = 15
    alpha = 0.1
    print(f"Parameters set: employees={employees}, alpha={alpha}")

    profit_chunks = []
    cost_chunks = []

    print("Adding constraints and calculating profit and cost terms...")
    for i in range(n_counties):
        # Skip zero population counties
        if totalPop[i] <= 0:
            print(f"County {i}: Skipping due to zero population.")
            m.Equation(x[i] == 0)
            m.Equation(P[i] == 0)
            continue

        # Calculate demand, revenue, and cost
        demand = totalPop[i] * 0.25 * 12 / (x[i] + 1e-6) * m.exp(-0.001 * P[i])
        revenue = P[i]**(IR[i]**0.5) * demand
        print(f"County {i}: Demand = {demand}, Revenue = {revenue}")
        loss = 0.082 * revenue
        cost = (minwage[i] * 40 * 52 * employees * 0.25) + (rent[i] * 0.25) + loss

        # Print revenue, cost, and profit for the first 5 counties
        if i < 5:  # Print for the first 5 counties
            print(f"County {i}: Demand = {demand}, Revenue = {revenue}, Cost = {cost}, Profit = {revenue - cost}")

        # Append intermediate terms
        cost_chunks.append(m.Intermediate(x[i] * cost))
        profit_chunks.append(m.Intermediate(x[i] * (revenue - cost)))

        # Add constraints
        # m.Equation(revenue >= 1000)  # Example: Minimum revenue of 1000 per county
        # m.Equation(x[i] <= alpha * totalPop[i])
        # print(f"County {i}: Added constraint x[{i}] <= {alpha * totalPop[i]:.2f}")
        # m.Equation(x[i] <= 20)
        # print(f"County {i}: Added constraint x[{i}] <= 20")

    print("Constraints and intermediate terms added.")

    # Chunked summation to avoid GEKKO's 15,000-character limit
    chunk_size = 200
    print(f"Chunk size for summation: {chunk_size}")

    profit_terms = []
    for i in range(0, len(profit_chunks), chunk_size):
        chunk = m.Intermediate(sum(profit_chunks[i:i + chunk_size]))
        profit_terms.append(chunk)
        print(f"Processed profit chunk {i // chunk_size + 1}.")

    total_profit_expr = m.Intermediate(sum(profit_terms))
    print("Total profit expression calculated.")

    cost_terms = []
    for i in range(0, len(cost_chunks), chunk_size):
        chunk = m.Intermediate(sum(cost_chunks[i:i + chunk_size]))
        cost_terms.append(chunk)
        print(f"Processed cost chunk {i // chunk_size + 1}.")

    total_cost_expr = m.Intermediate(sum(cost_terms))
    print("Total cost expression calculated.")

    # Budget constraint
    m.Equation(total_cost_expr <= budget)
    print(f"Added budget constraint: total_cost_expr <= {budget}")

    # # Max total restaurant constraint
    # m.Equation(m.sum([x[i] for i in range(n_counties)]) <= N)
    # print(f"Added max total restaurant constraint: sum(x) <= {N}")

    # Objective
    m.Obj(-total_profit_expr)  # Maximize profit
    print("Objective function set to maximize total profit.")

    # Solve
    print("Starting solver...")
    m.solve(disp=True)
    print("Solver finished.")

    # Extract results
    print("Extracting results...")
    optimal_x = [x[i].value[0] for i in range(n_counties)]
    optimal_P = [P[i].value[0] for i in range(n_counties)]

    # Approximate total profit
    print("Optimization complete.")
    for i in range(min(10, n_counties)):
        print(f"County {i}: NumStores = {optimal_x[i]:.0f}, Price = {optimal_P[i]:.2f}")

    return {
        'optimal_x': optimal_x,
        'optimal_P': optimal_P,
        'profit': -m.options.objfcnval
    }
