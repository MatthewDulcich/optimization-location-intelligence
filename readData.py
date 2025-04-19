import pandas as pd
import numpy as np

def getMinWage(filepath='data/StateMinimumWage.txt'):
    data = pd.read_csv(filepath)

    return data

def getPopulation(filepath='data/co-est2024-pop.xlsx'):
    data = pd.read_excel(filepath,
                         skiprows=[0,1,2,3,3149,3150,3151,3152,3153,3154,3155],
                         names=['Region','Base','2020','2021','2022','2023','2024'])
    data['County'] = data['Region'].apply(lambda x: x.split(' County, ')[0].split('.')[1])
    data['State'] = data['Region'].apply(lambda x: x.split(', ')[1])
    data.drop('Region',axis=1,inplace=True)

    return data

def getIncome(filepath='data/ACSST5Y2023.S1901-Data.csv'):
    data = pd.read_csv(filepath,
                       skiprows=[0])
    # Parse county and state into separate columns
    data['County'] = data['Geographic Area Name'].apply(lambda x: x.split(' County, ')[0])
    data['State'] = data['Geographic Area Name'].apply(lambda x: x.split(', ')[1])

    # Potentially use if want more than household incomes (family, married couples, etc.)
    #median_cols = [i for i in data.columns if 'Median' in i and 'Margin of Error' not in i]
    #mean_cols = [i for i in data.columns if 'Mean' in i and 'Margin of Error' not in i]

    # List necessary columns
    columns = ['Geography',
               'County',
               'State',
               'Estimate!!Households!!Median income (dollars)',
               'Estimate!!Households!!Mean income (dollars)']

    # Remove unnecessary columns and provide better column names
    data.drop([i for i in data.columns if i not in columns],axis = 1,inplace = True)
    data.rename({'Estimate!!Households!!Median income (dollars)':'MedianIncome',
                 'Estimate!!Households!!Mean income (dollars)':'MeanIncome',
                 'Geography':'CountyID'},
                 axis = 1, inplace = True)
    # Remove Puerto Rico from the data
    data = data.loc[data['State']!='Puerto Rico']

    # Fill missing values for median income
    data.loc[data['MedianIncome'].str.contains('-'),'MedianIncome'] = 0
    data['MedianIncome'] = data['MedianIncome'].astype(int)

    # Get ratios of income to max income
    max_mean_income = data['MeanIncome'].max()
    max_median_income = data['MedianIncome'].max()

    data['MeanIncomeRatio'] = data['MeanIncome']/max_mean_income
    data['MedianIncomeRatio'] = data['MedianIncome']/max_median_income

    return data

if __name__ == '__main__':
    print(getPopulation())
    print(getMinWage())
    print(getIncome())
