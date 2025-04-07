import pandas as pd

def getMinWage(filepath='data/StateMinimumWage.txt'):
    data = pd.read_csv(filepath)

    return data

def getPopulation(filepath='data/co-est2024-pop.xlsx'):
    data = pd.read_excel(filepath,
                         skiprows=[0,1,2,3,3149,3150,3151,3152,3153,3154,3155],
                         names=['Region','Base','2020','2021','2022','2023','2024'])
    data['County'] = data['Region'].apply(lambda x: x.split(' County, ')[0].split('.')[1])
    data['State'] = data['Region'].apply(lambda x: x.split(',')[1])
    data.drop('Region',axis=1,inplace=True)

    return data

if __name__ == '__main__':
    print(getPopulation())
    print(getMinWage())
