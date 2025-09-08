import ee
import geemap

# Authenticate and initialize Earth Engine
# The user needs to authenticate and provide the project name when running this.
try:
    ee.Initialize(project='ee708-rainfall-downscaling')
except Exception as e:
    print('Could not initialize Earth Engine with the project, trying to authenticate...')
    ee.Authenticate()
    ee.Initialize(project='ee708-rainfall-downscaling')

# Define the bounding box for the Himalayas region using specific coordinates.
himalayas_coords = [
    [76.47717699953913, 37.82664513320144],
    # [77.39147302732823, 36.47134074092161],
    # [79.67486846517245, 36.42314935342572],
    [80.90026755062496, 35.571803592889346],
    # [79.94743070144554, 32.78366923793975],
    # [81.87950044641099, 30.949760527513547],
    # [88.91295141369213, 28.71385417780965],
    # [94.50093992837051, 30.22256514409202],
    [97.21511065845843, 29.790425258748012],
    [97.9846584261799, 28.219530525561403],
    [97.17357167816135, 27.0017376993619],
    [93.32013533599073, 21.895638006891925],
    [92.12434489366078, 21.640209347541635],
    [91.44246786493959, 22.391938555703494],
    [87.81972158780012, 21.51648763575694],
    [87.17733874211916, 20.269295275574773],
    [80.95032103492291, 15.681782913430276],
    [80.28755447922079, 10.745717535241797],
    [80.03787822576705, 10.131854192880327],
    [77.5784805324031, 7.706238742849392],
    [76.24263442936031, 8.706332281101915],
    [74.43180260094009, 12.350805112400755],
    [72.39906488921469, 18.301660670307868],
    [67.94905459916471, 23.54293033675648],
    # [69.43726122140745, 27.764648638633588],
    [68.21883265960672, 23.247691821952614],
    [ 67.97708437455793, 23.839881647923207],
    [68.4698586761021, 24.46504974206444],
    [69.200023021483, 24.63005695508358],
    [70.58524083020568, 24.717456092256207],
    [69.16683481739948, 26.453152924415633],
    [69.00152321474258, 27.24672911668973],
    [69.91371433766922, 28.321561130479736],
    [71.14725193419356, 28.44732698104629],
    [73.27561368771326, 32.25007221718046],
    [71.84380570344365, 35.84156760093365],
    [73.14597692999519, 37.490491017813504],
    # [86.84464490510447, 35.64673507322577],  # Closing the polygon
]

region = ee.Geometry.Polygon(himalayas_coords)

# Define a new center point for map visualization based on the polygon's extent.
center_lon = 88.0
center_lat = 29.0

# Load the CHIRPS daily precipitation data
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')

# Filter for the most recent image
most_recent = chirps.sort('system:time_start', False).first()

# Get the original projection of the CHIRPS data
chirps_projection = most_recent.projection()

# Downscale the image to 0.25 degrees
downscaled = most_recent.reduceResolution(
    reducer=ee.Reducer.max(),
    maxPixels=1024
).reproject(
    crs=chirps_projection.crs(),
    scale=25000
)

# Create a mask for areas with precipitation greater than 0
precipitation_mask = downscaled.gt(0)

# Apply the mask to the image
masked_precipitation = downscaled.updateMask(precipitation_mask)

# Define visualization parameters for precipitation
palette = [
    '#000', '#0aab1e', '#e7eb05', '#ff4a2d', '#e90000'
]
vis_params = {
    'min': 1.0,
    'max': 17.0,
    'palette': palette
}

# Create a map centered on the region of interest
map_himalayas = geemap.Map(center=[center_lat, center_lon], zoom=5)

# Add the precipitation layer to the map, clipped to the region
map_himalayas.addLayer(masked_precipitation.clip(region), vis_params, 'CHIRPS Precipitation')

# Save the map to an HTML file
output_file = 'himalayas_map.html'
map_himalayas.to_html(output_file)

print(f"Map saved to {output_file}")

# Generate a URL for a 512x512 thumbnail image
thumbnail_params = {
    'dimensions': '512x512',
    'region': region,
    'palette': palette,
    'min': vis_params['min'],
    'max': vis_params['max']
}
thumbnail_url = masked_precipitation.getThumbUrl(thumbnail_params)
print(f"512x512 thumbnail URL: {thumbnail_url}")
