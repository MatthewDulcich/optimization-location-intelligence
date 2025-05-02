# Import the geopandas and geoplot libraries
import geopandas as gpd
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib.pyplot as plt
from readData import *
from rent_estimation import calculate_rent_estimation

# Load the json file with county coordinates
geoData = gpd.read_file(
    'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson'
)

income = getIncome()
populations = getPopulation()
populations = populations[['State','County','2024']]
populations.rename({'2024':'Population'},axis=1,inplace=True)
rent = calculate_rent_estimation()
rent = rent[['State','County','Estimated_annual_rent']]

data = populations.merge(income)
data = data.merge(rent)

geoData = geoData[~geoData.STATE.isin(['72'])]

print(geoData)
print(income)
income['id'] = income['CountyID'].apply(lambda x: x.split('US')[1])

print(income.loc[~income['id'].isin(geoData['id']),['id','State','County']])
print(geoData.loc[~geoData['id'].isin(income['id']),['id','STATE','COUNTY','NAME']])

# New FIPS codes https://developer.ap.org/ap-elections-api/docs/CT_FIPS_Codes_forPlanningRegions.htm
income.loc[income['id'] == '46102', 'id'] = '46113' # Shannon County is Oglala Lakota
income.loc[income['id'] == '09110', 'id'] = '09013' # Capitol Planning Region = Tolland
income.loc[income['id'] == '09120', 'id'] = '09001' # Greater Bridgeport = Fairfield
income.loc[income['id'] == '09130', 'id'] = '09007' # Lower CT River Valley = Middlesex
income.loc[income['id'] == '09140', 'id'] = '09003' # Naugatuck Valley Planning Region = Hartford
income.loc[income['id'] == '09150', 'id'] = '09015' # Northeastern CT Planning Region = Windham
income.loc[income['id'] == '09160', 'id'] = '09005' # Northwest Hills Planning Region = Litchfield
income.loc[income['id'] == '09170', 'id'] = '09009' # South Central Planning Region = New Haven
income.loc[income['id'] == '09180', 'id'] = '09011' # Southeastern Planning Region = New London
income.loc[income['id'] == '09190', 'id'] = '09001' # Western Connecticut Planning Region = Fairfield

print(income.loc[~income['id'].isin(geoData['id']),['id','State','County']])
print(geoData.loc[~geoData['id'].isin(income['id']),['id','STATE','COUNTY','NAME']])

geoData = geoData.merge(income,on='id')#left_on='GEO_ID',right_on='CountyID')

#print(geoData)
#print((geoData['GEO_ID'] == geoData['CountyID']).unique())
#print(geoData.loc[geoData['NAME'] != geoData['County'],['NAME','County']])
#exit()

# Make sure the "id" column is an integer
geoData.id = geoData['id'].astype(int)

# Remove Alaska, Hawaii and Puerto Rico.
statesToRemove = ['02', '15', '72']
geoData = geoData[~geoData.STATE.isin(statesToRemove)]

# Basic plot with just county outlines
gplt.choropleth(
    geoData,
    hue = 'MeanIncome',
    #hue = 'Estimated_annual_rent',
    #hue = 'id',
    projection=gcrs.PlateCarree(),
    extent=[-150,15,-40,60]
)
plt.show()
