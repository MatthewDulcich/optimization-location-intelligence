# Import the geopandas and geoplot libraries
import geopandas as gpd
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib.pyplot as plt
from readData import getIncome

# Load the json file with county coordinates
geoData = gpd.read_file(
    'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/US-counties.geojson'
)

data = getIncome()
geoData = geoData.merge(data,left_on='GEO_ID',right_on='CountyID')
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
    #hue = 'id',
    projection=gcrs.PlateCarree(),
    extent=[-150,15,-40,60]
)
plt.show()
