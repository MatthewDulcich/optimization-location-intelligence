

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
