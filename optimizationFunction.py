
def getConstraints(budget, N, risk):
    '''
    Creates constraints for optimization function given certain variables.

    Args:
        budget (int): Total budget for owning restaurants.
        N (int): Maximum number of stores to open.
        risk (float): Total acceptable risk ratio per location (cost/revenue)

    Returns:
        contraints: List of contraints to use in optimization function.
    '''
    contraints = [
                  {'type': 'ineq', 'fun': lambda x: budget - sum(costNet)}, # Budget >= x@costs
                  {'type': 'ineq', 'fun': lambda x: N - sum(x)}, # N >= sum(x)
                  {'type': 'ineq', 'fun': lambda x: risk - cost/revenue}, # risk > cost/revenue
                  {'type': 'ineq', 'fun': lambda x: x} # All x >= 0
                 ]

def demand(x,P,totalPop):
    '''
    Calculates demand as a function of price.

    args:
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county

    returns:
        demand: Demand corresponding to input variables.
    '''

    a = 0.003*P

    # L: Maximum demand at unit price 1.0.
    L = totalPop / x

    demand = L*np.exp(-a*P)

    return demand

def revenue(demand,P,IR):
    '''
    Calculates Revenue of Product

    args:
        demand: The demand of a given product

        P: Price of a given product

        IR: Income ratio of county to maximum county income. Between 0 and 1.

    returns:
        revenue: Demand times Price
    '''
    return demand*P**IR

def costs(x):
    '''
    SOMEONE FILL IN HERE!!!!!!!!!!!!!!!!!!!!!!!
    '''
    return(x)

def profit(x, P, totalPop, IR):
    '''
    Calculates Profit

    args:
        x: Total number of stores in given county
        P: Price
        totalPop: Total population in county
        IR: Income Ratio relative to maximum county income

    '''
    demand = demand(x,P,totalPop)

    rev = revenue(demand,P,IR)
    cost = costs(x)

    return rev - cost

def objectiveFunction(x, P, totalPop, IR):
    return - x.T @ profit(x, P, totalPop, IR)
