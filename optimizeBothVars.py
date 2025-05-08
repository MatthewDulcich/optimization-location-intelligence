import argparse
from readData import *
#from optimizationFunction import optimize, profit
from optimizationFunctionBothVars import *
from rent_estimation import calculate_rent_estimation
import geopandas as gpd
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime
import numpy as np
import cartopy.feature as cfeature

# Use argparse for different constraints
parser = argparse.ArgumentParser()
parser.add_argument('-b','--budget',type=float,help='Total Budget', default = 100000)
parser.add_argument('-N','--NumberLocations',type=int,help='Max number of Restaurants', default = 10)
parser.add_argument('-r','--risk',type=float,help='Acceptable maximum risk level. Between 0, 1', default = 0.8)

args = parser.parse_args()

budget = args.budget
N = args.NumberLocations
risk = args.risk

print(budget, N, risk)

income = getIncome()
income.drop(['MedianIncome','MedianIncomeRatio'],axis=1,inplace=True)
populations = getPopulation()
populations = populations[['State','County','2024']]
populations.rename({'2024':'Population'},axis=1,inplace=True)
rent = calculate_rent_estimation()
rent = rent[['State','County','Estimated_annual_rent']]
minwage = getMinWage()
restaurant = pd.read_csv("data/restaurant_data.csv")
print(restaurant[['fips','totalRestaurants']])

data = populations.merge(income)

data = data.merge(rent)
data = data.merge(minwage)

print(data['State'].unique())
#exit()
data['id'] = data['CountyID'].apply(lambda x: x[-5:])
# New FIPS codes https://developer.ap.org/ap-elections-api/docs/CT_FIPS_Codes_forPlanningRegions.htm
data.loc[data['id'] == '09110', 'id'] = '09013' # Capitol Planning Region = Tolland
data.loc[data['id'] == '09120', 'id'] = '09001' # Greater Bridgeport = Fairfield
data.loc[data['id'] == '09130', 'id'] = '09007' # Lower CT River Valley = Middlesex
data.loc[data['id'] == '09140', 'id'] = '09003' # Naugatuck Valley Planning Region = Hartford
data.loc[data['id'] == '09150', 'id'] = '09015' # Northeastern CT Planning Region = Windham
data.loc[data['id'] == '09160', 'id'] = '09005' # Northwest Hills Planning Region = Litchfield
data.loc[data['id'] == '09170', 'id'] = '09009' # South Central Planning Region = New Haven
data.loc[data['id'] == '09180', 'id'] = '09011' # Southeastern Planning Region = New London
data.loc[data['id'] == '09190', 'id'] = '09001' # Western Connecticut Planning Region = Fairfield
data['id'] = data['id'].astype(int)

data = data.merge(restaurant[['fips','totalRestaurants']],left_on='id',right_on='fips',how='left')
data.loc[data['id'] == 46102, 'id'] = 46113 # Shannon County is Oglala Lakota

#data['totalRestaurants'] = [0]*data.shape[0]
#print(data['State'].unique())
#exit()
#print(data)
#print(restaurant)
print(data['State'].unique())

#exit()
print("Number counties with no restuarant data",data.loc[data['totalRestaurants']==0].shape[0])

data.loc[data['totalRestaurants']==0,'totalRestaurants'] = data['totalRestaurants'].mean()
data.loc[np.isnan(data['totalRestaurants']),'totalRestaurants'] = data['totalRestaurants'].mean()

result_df = pd.DataFrame()
for i in range(int(data.shape[0]/100)+1):
    start = datetime.now()
    data2 = data.iloc[i*100:i*100+100,:]
    nvars = data2.shape[0]*2
    print(data2)
    #exit()

    from datetime import datetime
    start = datetime.now()
    results = optimize(budget = budget,
                       N = N,
                       risk = risk,
                       totalPop = data2['Population'],
                       IR = data2['MeanIncomeRatio'],
                       minwage = data2['MinWage'],
                       rent = data2['Estimated_annual_rent'],
                       NRestaurants = data2['totalRestaurants'],
                       nvars=nvars)

    print(results)
    print(datetime.now() - start)
    x = results.x
    data2['Prices'] = x[int(nvars/2):]
    data2['NStores'] = x[:int(nvars/2)]
    result_df = pd.concat([result_df,data2])
    print(datetime.now()-start)

profit_total = results.fun * -100000 # Function value times scale value
#data['Prices'] = x
#print(data['Prices'].describe())
result_df['Profit'] = result_df.apply(lambda x: profit(x['NStores'], x['Prices'], x['Population'], x['MeanIncomeRatio'],
                                             x['MinWage'], x['Estimated_annual_rent'], x['totalRestaurants']),axis=1)
result_df['Demand'] = result_df.apply(lambda x: demand(x['NStores'], x['Prices'], x['Population'], x['totalRestaurants']),axis=1)
result_df['Revenue'] = result_df.apply(lambda x: revenue(x['Prices'], x['MeanIncomeRatio'], x['Demand']),axis=1)
result_df['Profit_Margin'] = result_df['Profit'] / result_df['Revenue'] * 100
print(result_df)
#[profit(x, P, Pop, IR, mw, r) for x,P,Pop,IR,mw,r in zip(x,P,totalPop,IR,minwage,rent)]
print(profit_total)
geoData = gpd.read_file(
    'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson'
)
statesToRemove = ['02', '15', '72']
geoData = geoData[~geoData.STATE.isin(statesToRemove)]
geoData['id'] = geoData['id'].astype(int)
#data['id'] = data['CountyID'].apply(lambda x: x.split('US')[1])
print(result_df['id'])
print(geoData['id'])

print("IDs")
print(result_df['id'])
print(result_df.loc[result_df['id']=='09001'])
print(geoData.loc[geoData['id']=='09001'])
print(result_df.loc[result_df['State']=='Connecticut'])


result_df = geoData.merge(result_df,on='id')

result_df['Prices'] = result_df['Prices']*result_df['MeanIncomeRatio']**0.5
ax = plt.axes(projection = gcrs.PlateCarree())

norm = Normalize(vmin=result_df['Prices'].min(),
                 vmax=result_df['Prices'].max())
# Basic plot with just county outlines
im = gplt.choropleth(
    result_df,
    hue = 'Prices',
    #hue = 'Estimated_annual_rent',
    #hue = 'id',
    projection=gcrs.PlateCarree(),
    extent=[-150,15,-40,60],
    cmap = 'viridis',
    ax = ax,
    norm=norm
    #legend_kwds={"orientation": "horizontal", "pad": 0.01}
)


cbar = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
plt.colorbar(cbar, ax=ax, pad = -0.25, shrink = 0.4)
plt.title(f'Price By County',y = .8, fontsize='small')
plt.savefig(f"Images/Optimal_Price_X.png",bbox_inches='tight',dpi=300)
plt.close()


ax = plt.axes(projection = gcrs.PlateCarree())

norm = Normalize(vmin=-1e6,
                 vmax=1e6)
# Basic plot with just county outlines
im = gplt.choropleth(
    result_df,
    hue = 'Profit',
    #hue = 'Estimated_annual_rent',
    #hue = 'id',
    projection=gcrs.PlateCarree(),
    extent=[-150,15,-40,60],
    cmap = 'bwr',
    ax = ax,
    norm=norm
    #legend_kwds={"orientation": "horizontal", "pad": 0.01}
)
print(result_df['Profit'].describe())


cbar = plt.cm.ScalarMappable(norm=norm, cmap='bwr')
plt.colorbar(cbar, ax=ax, pad = -0.25, shrink = 0.4)
plt.title(f'Profit By County', y = .8, fontsize='small')
plt.savefig(f"Images/Optimal_Profit_X_Price.png",bbox_inches='tight',dpi=300)
plt.close()

result_df['NStores'] = result_df['NStores'].round()

ax = plt.axes(projection = gcrs.PlateCarree())
#ax.add_feature(cfeature.STATES,edgecolor='green')
norm = Normalize(vmin=0,
                 vmax=10)
# Basic plot with just county outlines
im = gplt.choropleth(
    result_df,
    hue = 'NStores',
    #hue = 'Estimated_annual_rent',
    #hue = 'id',
    projection=gcrs.PlateCarree(),
    extent=[-150,15,-40,60],
    cmap = 'viridis',
    ax = ax,
    norm=norm
    #legend_kwds={"orientation": "horizontal", "pad": 0.01}
)
print(result_df['NStores'].describe())
print("Total stores",result_df['NStores'].sum())

cbar = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
plt.colorbar(cbar, ax=ax, boundaries=np.arange(0,11,1), pad = -0.25, shrink = 0.4)
plt.title(f'Number of Stores By County', y = .8, fontsize='small')
plt.savefig(f"Images/Optimal_NStores_2vars.png",bbox_inches='tight',dpi=300)
plt.close()





result_df.loc[result_df['Profit'] < 0,'Profit'] = 0
result_df.loc[result_df['Profit_Margin'] < 0,'Profit_Margin'] = 0
print("Total Profit",sum(result_df['Profit']))
n_counties_with_stores = result_df.loc[result_df['Profit']>0].shape[0]
print("Profit per store",sum(result_df['Profit'])/n_counties_with_stores)
print(result_df.loc[result_df['Profit']==0].shape[0])

result_df = result_df.loc[result_df['Profit']>0]
print(result_df[['Demand','Revenue','Profit_Margin','Profit','NStores','Population']].describe())

pd.set_option('display.max_rows', 500)
print(result_df[['id','NAME','Population','MeanIncome','Prices','NStores','Profit']])

print("Total stores after rounding",result_df['NStores'].sum())

exit()
#data['Nstores'] = x[:100]
print(data[['Prices','Nstores']].describe())
print(data['Nstores'].sum())
print(data.loc[data['Nstores']>=1])
print(f"Profit {profit}")
