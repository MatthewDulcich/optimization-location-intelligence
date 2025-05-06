# Optimization for Restaurant Location and Pricing

This project optimizes the number of stores and pricing strategy for restaurants across counties to maximize profit while adhering to budget, risk, and operational constraints.

---

## Objective Function

The **objective function** aims to **maximize profit** by minimizing the negative profit across all counties. The profit is calculated as:

**Profit = Revenue - Costs**

The **objective function** is defined as:

**Objective Function** = `-(1 / 100,000) * sum(x[i] * Profit(x[i], P[i], totalPop[i], IR[i], minwage[i], rent[i]))`

Where:
- `x[i]`: Number of stores in county `i`.
- `P[i]`: Price per product in county `i`.
- `totalPop[i]`: Total population in county `i`.
- `IR[i]`: Income ratio of county `i` relative to the maximum county income.
- `minwage[i]`: Minimum wage in county `i`.
- `rent[i]`: Rent of the restaurant in county `i`.

---

## Constraints

The optimization problem is subject to the following constraints:

### 1. Budget Constraint
The total cost of opening and operating stores must not exceed the budget:

**Budget Constraint**:  
`budget - sum(x[i] * Costs(P[i], IR[i], minwage[i], rent[i], demand[i])) >= 0`

Where:
- `Costs(P, IR, minwage, rent, demand) = minwage * employees + rent + loss`
- `loss = 0.082 * Revenue(P, IR, demand)`

---

### 2. Total Stores Constraint
The total number of stores opened across all counties must not exceed the maximum allowed:

**Total Stores Constraint**:  
`N - sum(x[i]) = 0`

---

### 3. Risk Constraint
The risk ratio (cost/revenue) for each county must not exceed the acceptable risk threshold:

**Risk Constraint**:  
`risk - (Costs(P[i], IR[i], minwage[i], rent[i], demand[i]) / Revenue(P[i], IR[i], demand[i])) >= 0`

If revenue is zero (`Revenue = 0`), the constraint defaults to `-1` to indicate infeasibility.

---

## Decision Variables

1. **Number of Stores (`x`)**:
   - `x[i]`: Number of stores in county `i`.
   - Bounds: `0 <= x[i] <= 100`.

2. **Price (`P`)**:
   - `P[i]`: Price per product in county `i`.
   - Bounds: `0 <= P[i] <= 50`.

---

## Functions

### 1. Demand Function
The demand for a product in a county is calculated as:

**Demand(x, P, totalPop)** =  
`(totalPop / x) * exp(-0.003 * P^2)` if `x >= 1` and `P <= 50`, otherwise `0`.

---

### 2. Revenue Function
The revenue for a product in a county is calculated as:

**Revenue(P, IR, demand)** =  
`P^(sqrt(IR)) * demand`

---

### 3. Costs Function
The costs for operating a restaurant in a county are calculated as:

**Costs(P, IR, minwage, rent, demand)** =  
`minwage * employees + rent + loss`

Where:  
`loss = 0.082 * Revenue(P, IR, demand)`

---

## Optimization Workflow

### 1. Initialization
Start with an initial guess for `x` and `P`:

- `x_0 = [1, 1, ..., 1]` (for `N` counties)
- `P_0 = [18, 18, ..., 18]` (initial price per county)

---

### 2. Constraints
Apply the budget, total stores, and risk constraints.

---

### 3. Bounds
Ensure `x` and `P` remain within their respective bounds.

---

### 4. Optimization
Use the `gd_minimize` function to minimize the objective function subject to the constraints and bounds.

---

## Implementation

The optimization is implemented in the `optimize` function:
```python
def optimize(budget, N, risk, totalPop, IR, minwage, rent):
    x0 = np.zeros(NUMBER_OF_COUNTIES)
    x0[:N] = 1
    P = 18 * np.ones(NUMBER_OF_COUNTIES)
    x0 = np.concatenate((x0, P))

    constraints = getConstraints(x0, budget, N, risk, totalPop, IR, minwage, rent)

    bounds = [(0, 100) for _ in range(NUMBER_OF_COUNTIES)] + \
             [(0, 50) for _ in range(NUMBER_OF_COUNTIES)]

    result = gd_minimize(
        fun=objectiveFunction,
        x0=x0,
        args=(totalPop, IR, minwage, rent),
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )
    return result