# Optimization for Restaurant Location and Pricing

This project optimizes the number of restaurant locations and product pricing across U.S. counties to maximize profit while satisfying constraints on budget, store limits, and operational cost-risk. The optimization uses real-world data on demographics, rent, and wages.

---

## Objective Function

The optimization minimizes the **negative total profit**, defined as:

```
Objective Function = -(1 / 100,000) * sum(x[i] * Profit_i)
```

Where:
- `x[i]`: Number of stores in county `i`
- `Profit_i = Revenue_i - Cost_i`
- `Revenue_i = P[i]^(sqrt(IR[i])) * Demand[i]`
- `Demand[i] = (TotalPop[i] / x[i]) * 0.25 * 365 * exp(-0.001 * P[i]^2)`
- `Cost_i = minwage[i] * 15 * 40 * 52 + rent[i] + 0.082 * Revenue_i`

---

## Constraints

### 1. Budget Constraint
```
sum(x[i] * Cost_i) <= Budget
```

### 2. Store Count Constraint
```
sum(x[i]) = N
```

### 3. Variable Bounds
- `0 <= x[i] <= N`
- `0 <= P[i] <= 30`

---

## Decision Variables

- `x[i]`: Number of stores in county `i` (continuous, rounded after optimization)
- `P[i]`: Product price in county `i`

---

## Solver and Optimization Approach

We use the **SLSQP** (Sequential Least Squares Quadratic Programming) solver from `scipy.optimize` to solve the nonlinear, non-convex optimization problem. The solver:
- Uses quadratic approximation of the Lagrangian
- Supports general nonlinear equality and inequality constraints
- Works with differentiable but non-convex functions

### Other Solvers Considered

| Solver         | Notes                                                                 |
|----------------|-----------------------------------------------------------------------|
| CVXPY          | Failed due to non-convex revenue and demand structure                |
| Pyomo (IPOPT)  | Infeasible; sensitive to exponential terms and poor Jacobian scaling |
| SCIP           | Did not converge on relaxed problem or integer form                  |
| Gekko          | Performed better with nonlinear constraints, but tuning was difficult |
| Genetic Algo   | Development incomplete due to time constraints                        |

---

## Implementation Challenges

- **Scaling**: 3000 counties × 2 variables = 6000+ variables → solved state-by-state
- **Non-convexity**: Multiple local minima; initialization mattered
- **Numerical Instability**: Exponential terms caused flat/steep gradients
- **Poor Jacobians**: Solver sensitivity due to ill-conditioned derivatives

---

## Repository Setup

### Clone the Repository
```bash
git clone https://github.com/MatthewDulcich/optimization-location-intelligence.git
cd optimization-location-intelligence
```

### Set Up Python Environment with `pyenv`
```bash
pyenv install 3.11.9
pyenv virtualenv 3.11.9 opt-env
pyenv local opt-env
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## Optimization Workflow

```python
def optimize(budget, N, totalPop, IR, minwage, rent):
    x0 = np.zeros(NUM_COUNTIES)
    x0[:N] = 1
    P = 18 * np.ones(NUM_COUNTIES)
    x0 = np.concatenate((x0, P))

    constraints = getConstraints(x0, budget, N, totalPop, IR, minwage, rent)
    bounds = [(0, N)] * NUM_COUNTIES + [(0, 30)] * NUM_COUNTIES

    result = minimize(
        fun=objectiveFunction,
        x0=x0,
        args=(totalPop, IR, minwage, rent),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'disp': True}
    )
    return result
```

---

## Project Structure

```
optimization-location-intelligence/
├── data/                  # Datasets and preprocessing
├── scripts/               # Optimization and analysis scripts
├── results/               # Output from runs
├── plots/                 # Graphs and visualizations
├── main.py                # Main driver
├── requirements.txt
└── README.md
```

---

## Authors

- Matthew Dulcich
- James Frech
- Peeyush Dyavarashetty
- Krishna Taduri

---

## License

MIT License
