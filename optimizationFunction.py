
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
# TODO: Integrate P into costs
def costs(minwage, rent, loss, initialCosts, employees=15):
    '''
    Calculates costs of restaurant.
    args:
        minwage: Minimum wage in county
        rent: Rent of restaurant
        loss: Losses incurred by restaurant
        initialCosts: Initial costs of opening restaurant
        employees: Number of employees in restaurant (default is 15)
    returns:
        netCosts: Total costs of restaurant
    '''

    operatingCosts = minwage * employees + rent + loss
    netCosts = operatingCosts + initialCosts


    return(netCosts)

def profit(x, P, totalPop, IR, minwage, rent, loss, initialCosts):
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
    cost = costs(minwage, rent, loss, initialCosts, employees=15)

    return rev - cost

def objectiveFunction(x, P, totalPop, IR):
    return - x.T @ profit(x, P, totalPop, IR)

def getConstraints(budget, N, risk, x, P, demand, IR):
    '''
    Creates constraints for optimization function given certain variables.

    Args:
        budget (int): Total budget for owning restaurants.
        N (int): Maximum number of stores to open.
        risk (float): Total acceptable risk ratio per location (cost/revenue)

    Returns:
        contraints: List of contraints to use in optimization function.
    '''
    cost = costs(x, P)
    revenue = revenue(demand, P, IR)
    contraints = [
                  {'type': 'ineq', 'fun': budget - sum(cost)}, # Budget >= x@costs
                  {'type': 'ineq', 'fun': N - sum(x)}, # N >= sum(x)
                  {'type': 'ineq', 'fun': risk - cost/revenue}, # risk > cost/revenue
                  {'type': 'ineq', 'fun': x} # All x >= 0
                 ]
