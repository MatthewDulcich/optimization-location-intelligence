import argparse
from readData import *
from optimizationFunction import *

# Use argparse for different constraints
parser = argparse.ArgumentParser()
parser.add_argument('-b','--budget',type=int,help='Total Budget')
parser.add_argument('-N','--NumberLocations',type=int,help='Max number of Restaurants')
parser.add_argument('-r','--risk',type=float,help='Acceptable maximum risk level. Between 0, 1.')

args = parser.parse_args()

budget = args.budget
N = args.NumberLocations
risk = args.risk

print(budget, N, risk)

income = getIncome()
population = getPopulation()
minwage = getMinWage()

constraints = getConstraints(budget = budget, N = N, risk = risk)
