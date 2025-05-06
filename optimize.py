import argparse
from readData import *
from optimizationFunction import optimize
from rent_estimation import calculate_rent_estimation

# Use generators for data processing
def process_data():
    income = getIncome()
    income.drop(['MedianIncome','MedianIncomeRatio'], axis=1, inplace=True)
    
    populations = getPopulation()
    populations = populations[['State','County','2024']]
    populations.rename({'2024':'Population'}, axis=1, inplace=True)
    
    rent = calculate_rent_estimation()
    rent = rent[['State','County','Estimated_annual_rent']]
    
    minwage = getMinWage()
    
    # Merge data in chunks to reduce memory usage
    data = populations.merge(income)
    del populations, income  # Explicitly free memory
    
    data = data.merge(rent)
    del rent
    
    data = data.merge(minwage)
    del minwage
    
    # Only keep the slice we need
    data = data.iloc[200:300,:]
    
    return data

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--budget', type=float, help='Total Budget', default=100000)
    parser.add_argument('-N','--NumberLocations', type=int, help='Max number of Restaurants', default=10)
    parser.add_argument('-r','--risk', type=float, help='Acceptable maximum risk level. Between 0, 1', default=0.8)
    
    args = parser.parse_args()
    
    # Process data
    data = process_data()
    
    # Run optimization
    from datetime import datetime
    start = datetime.now()
    
    results = optimize(
        budget=args.budget,
        N=args.NumberLocations,
        risk=args.risk,
        totalPop=data['Population'],
        IR=data['MeanIncomeRatio'],
        minwage=data['MinWage'],
        rent=data['Estimated_annual_rent']
    )
    
    # Process results
    x = results.x
    profit = results.fun * -100000
    
    data['Prices'] = x[100:]
    data['Nstores'] = x[:100]
    
    # Print results
    print(f"Optimization completed in {datetime.now() - start}")
    print(data[['Prices','Nstores']].describe())
    print(f"Total stores: {data['Nstores'].sum()}")
    print(data.loc[data['Nstores']>=1])
    print(f"Profit: {profit}")
    
    # Clean up
    del data, results, x

if __name__ == "__main__":
    main()
