import numpy as np
import pandas as pd
import geopandas as gpd

from .graph import graph_from_nlg

def empty_graph(shapefile, population, label = ''):

    gdf_place = gpd.read_file(shapefile)

    df_place = pd.read_excel(
        population,
        skiprows = 3, skipfooter = 5,
    )

    df_place = df_place.rename(columns = {'Unnamed: 0': 'Name', 'Unnamed: 1': 'Base'})

    df_place_names = df_place['Name'].to_numpy()
    df_place_population = df_place[2022].to_numpy()
    gdf_place_names = gdf_place['NAME'].to_numpy()

    pop = {}

    for name in gdf_place_names:

        for idx, check_name in enumerate(df_place_names):

            check_name = check_name.split(',')[0]

            if name in check_name:

                pop[name] = df_place_population[idx]
                break

    populations = []

    for name in gdf_place_names:

        populations.append(pop.get(name, 0))

    gdf_place['population'] = populations

    lon, lat = np.array([x.coords.xy for x in gdf_place.geometry.centroid]).T[0]
    names = gdf_place['NAME'].to_numpy()
    pop = gdf_place['population'].to_numpy()

    nodes = []

    for idx in range(len(names)):

        name = names[idx]

        if label:

            name += ', ' + label

        if type(pop[idx]) is str:

            continue

        node = {
            'id': name,
            'x': lon[idx],
            'y': lat[idx],
            'population': pop[idx],
            'type': 'place',
        }

        nodes.append(node)

    links = []
        
    return graph_from_nlg({'nodes': nodes, 'links': links})