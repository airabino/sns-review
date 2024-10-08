import os
import requests
import json
import zipfile
import io

# States to Get (fips)
states = ['06']

# Loading in the keys

with open('keys.json', 'r') as file:

    keys = json.load(file)

with open('state_fips.json', 'r') as file:

    encoding = json.load(file)

# Creating a folder for generated outputs

directory = 'Data/Outputs/'

try:

    os.makedirs(directory)

except OSError as e:

    pass

# Pulling State shapes from Census Bureau

directory = 'Data/State/'
url = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
file = 'tl_2023_us_state.shp'

try:

    os.makedirs(directory)

except OSError as e:

    pass

path = directory + file

if not os.path.isfile(path):

    response = requests.get(url)
    zipped_data = zipfile.ZipFile(io.BytesIO(response._content))

    zipped_data.extractall(directory)

print('States Pulled')

# Pulling Tract shapes from Census Bureau

directory = 'Data/Tract/'
url = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_06_tract.zip"
file = 'tl_2023_06_tract.shp'

try:

    os.makedirs(directory)

except OSError as e:

    pass

path = directory + file

if not os.path.isfile(path):

    response = requests.get(url)
    zipped_data = zipfile.ZipFile(io.BytesIO(response._content))

    zipped_data.extractall(directory)

print('Tracts Pulled')

# Pulling Urban Areas shapes from Census Bureau

directory = 'Data/UAC/'
url = "https://www2.census.gov/geo/tiger/TIGER2023/UAC/tl_2023_us_uac20.zip"
file = 'tl_2023_us_uac20.shp'

try:

    os.makedirs(directory)

except OSError as e:

    pass

path = directory + file

if not os.path.isfile(path):

    response = requests.get(url)
    zipped_data = zipfile.ZipFile(io.BytesIO(response._content))

    zipped_data.extractall(directory)

print('UAC Pulled')


# Pulling Place shapes from Census Bureau

directory = 'Data/Place/'
shapefile_url = 'https://www2.census.gov/geo/tiger/TIGER2023/PLACE/'
population_url = (
    'https://www2.census.gov/programs-surveys/popest/tables/2020-2023/cities/totals/'
)

try:

    os.makedirs(directory)

except OSError as e:

    pass

for state in states:

    path = directory + state + '/'

    if not os.path.exists(path):

        os.mkdir(path)

        # Pulling the shapefile
        url = shapefile_url + f'tl_2023_{state}_place.zip'
    
        response = requests.get(url)
        zip = zipfile.ZipFile(io.BytesIO(response.content))
        zip.extractall(path)
        
        # Pulling the population data
        out_file = path + 'population.xlsx'
        
        url = population_url + f'SUB-IP-EST2023-POP-{state}.xlsx'
    
        response = requests.get(url)

        with open(out_file, 'wb') as file:

            file.write(response.content)
            file.close()

print('Places Pulled')

# url = (
#     f"https://developer.nrel.gov/api/alt-fuel-stations/v1.csv?access=public" +
#     f"&api_key=w30dJhEvcdCHxEHImCSL0GIAXETjLIJ41lgoN0Jr&cards_accepted=all" +
#     f"&cng_fill_type=all&cng_has_rng=all&cng_psi=all&country=US&download=true" +
#     f"&e85_has_blender_pump=false&ev_charging_level=dc_fast&ev_connector_type=all" +
#     f"&ev_network=all&fuel_type=ELEC&funding_sources=all&hy_is_retail=true" +
#     f"&limit=all&lng_has_rng=all&lpg_include_secondary=false&maximum_vehicle_class=all" +
#     f"&offset=0&owner_type=all&state=US-CA&status=E&utf8_bom=true"
#     )
# Pulling supply stations from AFDC

directory = 'Data/Stations/'
file = 'stations.json'
key = keys['afdc']

try:

    os.makedirs(directory)

except OSError as e:

    pass

for state in states:

    path = directory + state + '/'

    if not os.path.exists(path):

        os.mkdir(path)

        name = encoding['all'][state]

        url = (
            f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json?" + 
            f"fuel_type=ELEC&limit=all&state={name}" +
            f"&api_key={key}"
        )

        response = requests.get(url)

        with open(path + file, 'w') as file:

            json.dump(response.json(), file, indent = 4)

print('Stations Pulled')

# Pulling 2022 NHTS data from ORNL

directory = 'Data/NHTS/'
url = "https://nhts.ornl.gov/assets/2022/download/csv.zip"
file = 'tripv2pub.csv'

try:

    os.makedirs(directory)

except OSError as e:

    pass

path = directory + file

if not os.path.isfile(path):

    response = requests.get(url)
    zipped_data = zipfile.ZipFile(io.BytesIO(response._content))

    zipped_data.extractall(directory)

print('NHTS Pulled')

print('Done')