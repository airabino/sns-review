'''
Module for handling of graphs.

Graphs must be in Node-Link Graph (NLG) format:
Long example can be found at - https://gist.github.com/mbostock/4062045

Short example (right triangle):

{
	"nodes":[
	{"id": 0, "x": 0, "y": 0},
	{"id": 1, "x": 1, "y": 0},
	{"id": 2, "x": 0, "y": 1}
	],
	"links":[
	{"source": 0, "target": 1, "length": 1},
	{"source": 1, "target": 2, "length": "1.414"},
	{"source": 2, "target": 0, "length": 1},
	]
}

NLG dictionaries can be loaded from JSON or shapefile with or without links (adjacency)

NLG dictionaries can be created from DataFrames without links

NLG dictionaries are saved as .json files

!!!!! In this module graph refers to a networkx graph, nlg to a NLG graph !!!!!

NLG terminology maps to NetworkX terminology as follows:
Node -> node,
Link -> edge, adj

Nodes of a graph may also be referred to as vertices
'''
import json
import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from scipy.spatial import KDTree

def cypher(graph):

	encoder = {k: idx for idx, k in enumerate(graph.nodes)}
	decoder = {idx: k for idx, k in enumerate(graph.nodes)}

	return encoder, decoder

def graph_from_communities(graph, communities):

    _node = graph._node

    nodes = []

    for idx, community in enumerate(communities):

        x_coordinates = []
        y_coordinates = []
        populations = []

        for source in community:

            x_coordinates.append(_node[source]['x'])
            y_coordinates.append(_node[source]['y'])
            populations.append(_node[source]['population'])

        node = {
            'id': f'community_{idx}',
            'x': np.mean(x_coordinates),
            'y': np.mean(y_coordinates),
            'population': sum(populations),
            'places': [k for k in community],
        }

        nodes.append(node)

    links = []

    return graph_from_nlg({'nodes': nodes, 'links': links})

# Functions for NLG JSON handling 

class NpEncoder(json.JSONEncoder):
	'''
	Encoder to allow for numpy types to be converted to default types for
	JSON serialization. For use with json.dump(s)/load(s).
	'''
	def default(self, obj):

		if isinstance(obj, np.integer):

			return int(obj)

		if isinstance(obj, np.floating):

			return float(obj)

		if isinstance(obj, np.ndarray):

			return obj.tolist()

		return super(NpEncoder, self).default(obj)

def nlg_to_json(nlg, filename):
	'''
	Writes nlg to JSON, overwrites previous
	'''

	with open(filename, 'w') as file:

		json.dump(nlg, file, indent = 4, cls = NpEncoder)

def append_nlg(nlg, filename):
	'''
	Writes nlg to JSON, appends to existing - NEEDS UPDATING
	'''

	nlg_from_file = Load(filename)

	nlg = dict(**nlg_from_file, **nlg)

	with open(filename, 'a') as file:

		json.dump(nlg, file, indent = 4, cls = NpEncoder)

def nlg_from_json(filename):
	'''
	Loads graph from nlg JSON
	'''

	with open(filename, 'r') as file:

		nlg = json.load(file)

	return nlg

# Functions for NetworkX graph .json handling

def graph_to_json(graph, filename, **kwargs):
	'''
	Writes graph to JSON, overwrites previous
	'''

	with open(filename, 'w') as file:

		json.dump(nlg_from_graph(graph, **kwargs), file, indent = 4, cls = NpEncoder)

def graph_from_json(filename, **kwargs):
	'''
	Loads graph from nlg JSON
	'''

	with open(filename, 'r') as file:

		nlg = json.load(file)

	return graph_from_nlg(nlg, **kwargs)

# Functions for converting between NLG and NetworkX graphs

def graph_from_nlg(nlg, **kwargs):

	return nx.node_link_graph(nlg, multigraph = False, **kwargs)

def nlg_from_graph(nlg, **kwargs):

	nlg = nx.node_link_data(nlg, **kwargs)

	return nlg

# Functions for loading graphs from shapefiles

def graph_from_shapefile(filepath, node_attributes = {}, link_attributes = {}, **kwargs):
	'''
	Loads a graph from a shapefile containing nodes and links.
	Also allows for reformating of the graph into a standard numerically indexed graph.

	Reformatting:

	Momepy builds graphs where node ids are tuples of coordinates (x, y). This is somewhat
	inconvenient for indexing and plotting and results in larger .json files expecially
	when coordinates are longitude and latitude specified to 10+ decimal places.
	Reformatted graphs have numerical node ids and the coordinates are moved to the 'x'
	and 'y' node fields.

	See reformat_graph for description of node_attributes and link_attributes
	'''
	contains_links = kwargs.get('contains_links', True)
	conditions = kwargs.get('conditions', [])

	if contains_links:

		# Loading the road map shapefile into a GeoDataFrame
		gdf = gpd.read_file(filepath)

		for condition in conditions:

			gdf = gdf[eval(condition)]

		# Making sure that cartographic crs is used so
		# Haversine distances can be accurately computed
		gdf = gdf.to_crs(4326)

		# Creating a NetworkX Graph
		graph = graph_from_gdf(gdf)

		# Reformatting the Graph
		graph = reformat_graph(
			graph, node_attributes, link_attributes, **kwargs)

	else:

		# Loading the shapefile into a GeoDataFrame
		gdf = gpd.read_file(filepath)

		# Making sure that cartographic crs is used so
		# Haversine distances can be accurately computed
		gdf = gdf.to_crs(4326)

		nlg = nlg_from_dataframe(gdf, node_attributes)

		graph = graph_from_nlg(nlg)

	return graph

def graph_from_gdf(gdf, directed = False):
	'''
	Calls momepy gdf_to_nx function to make a Graph from a GeoDataFrame.
	In this case the primal graph (vertex-defined) is called for, multi-paths
	are disallowed (including self-loops), and directed Graphs are kept as directed.
	'''

	graph = momepy.gdf_to_nx(
		gdf,
		approach = 'primal',
		multigraph = False,
		directed = directed,
		)

	return graph

def reformat_graph(graph, node_attributes = {}, link_attributes = {}, **kwargs):
	'''
	Reformats a graph to contain numeric node IDs and specified edge information.
	This format makes later computation of routes easier.

	node_attributes -> {field: lambda function}
	link_attributes -> {field: lambda function}

	ex:

	link_attributes = {'speed': lambda e: e['speed'] * 1.609}
	where e is the graph edge graph._adj[origin][destination]
	'''

	# Extracting node data from the graph
	node_ids = list(graph.nodes)

	# Extracting the x and y coordinates from the Graph
	coordinates = np.array([key for key, value in graph._node.items()])
	x, y = coordinates.reshape((-1, 2)).T

	# Creating a spatial KD Tree for quick identification of matches
	# between Graph nodes and equivalent vertices
	kd_tree = KDTree(coordinates.reshape((-1, 2)))

	nodes = []
	links = []

	# Looping on nodes
	for source_idx, source in enumerate(node_ids):

		# Adding id, x, and y fields
		node = {
			'id': source_idx,
			'x': x[source_idx],
			'y': y[source_idx],
			}

		for field, fun in node_attributes.items():

			if type(fun) is str:
				
				fun = eval(fun)

			node[field] = fun(graph._node[source])

		nodes.append(node)

		# Pulling the coordinates of the adjacent nodes from the graph
		targets = graph._adj[source].keys()

		# Looping on adjacency
		for target in targets:

			# Finding the matching node index for the node coordinates
			target_idx = kd_tree.query(list(target))[1]

			link={
				'source': source_idx,
				'target': target_idx,
				}

			for field, fun in link_attributes.items():

				if type(fun) is str:
				
					fun = eval(fun)

				link[field] = fun(graph._adj[source][target])

			links.append(link)

	nlg = {'nodes': nodes, 'links': links}

	return graph_from_nlg(nlg, **kwargs)

# Functions for CSV handling

def graph_from_csv(filename, node_attributes = {}):
	'''
	Creates graph with empty adjacency from dataframe.
	See reformat_graph for description of node_attributes.
	'''

	dataframe = dataframe_from_csv(filename)
	nlg = nlg_from_dataframe(dataframe, node_attributes)

	return graph_from_nlg(nlg)

def dataframe_from_csv(filename, **kwargs):
	'''
	Loads data provided as CSV to DataFrame. Can also load multiple CSV with
	the same columns into a singe DataFRame
	'''

	if type(filename) is str:

		filename = [filename]

	dataframes = []

	for file in filename:

		dataframes.append(pd.read_csv(file, **kwargs))

	dataframe = pd.concat(dataframes, axis = 0)
	dataframe.reset_index(inplace = True, drop = True)

	return dataframe

def dataframe_from_xlsx(filename, **kwargs):
	'''
	Loads data provided as CSV to DataFrame. Can also load multiple CSV with
	the same columns into a singe DataFRame
	'''

	if type(filename) is str:

		filename = [filename]

	dataframes = []

	for file in filename:

		dataframes.append(pd.read_excel(file, **kwargs))

	dataframe = pd.concat(dataframes, axis = 0)
	dataframe.reset_index(inplace = True, drop = True)

	return dataframe

def nlg_from_dataframe(dataframe, node_attributes = {}):
	'''
	Creates NLG dictionary with empty links from dataframe.
	See reformat_graph for description of node_attributes.
	'''

	nodes = []

	for source_idx, source in dataframe.iterrows():

		# Adding id field and status field - status == 0 for adjacency not computed
		node = {
			'id': source_idx,
			}

		for field, fun in node_attributes.items():

			if type(fun) is str:

				fun = eval(fun)

			node[field] = fun(source)

		nodes.append(node)

	nlg = {'nodes': nodes, 'links': []}

	return nlg

def graph_from_dataframe(dataframe, node_attributes = {}):

	nlg = nlg_from_dataframe(dataframe, node_attributes = node_attributes)

	return graph_from_nlg(nlg)

def exclude_rows(dataframe, attributes):
	'''
	Removes DataFrame rows that meet criteria
	'''

	for attribute, values in attributes.items():

		dataframe = dataframe[~np.isin(dataframe[attribute].to_numpy(), values)].copy()
		dataframe.reset_index(inplace = True, drop = True)

	return dataframe

def keep_rows(dataframe, attributes):
	'''
	Removes DataFrame rows that do not meet criteria
	'''

	for attribute,values in attributes.items():

		dataframe = dataframe[np.isin(dataframe[attribute].to_numpy(), values)].copy()
		dataframe.reset_index(inplace = True, drop = True)

	return dataframe

# Functions for graph operations

def mark_nodes(graph, nodes, field, value, **kwargs):

	for node in nodes:

		graph._node[node][field] = value

	return graph

def remove_edges(graph, criteria = []):

	_adj = {}

	for source, adj in graph._adj.items():

		_adj[source] = {}

		for target, edge in adj.items():

			keep = True

			for fun in criteria:

				keep *= fun(edge)

			if keep:

				_adj[source][target] = edge

	graph._adj = _adj

	return graph

def subgraph(graph, nodes):

	_node = graph._node
	_adj = graph._adj

	node_list = [(n, _node[n]) for n in nodes]

	edge_list = []

	for source in nodes:
		for target in nodes:

			edge_list.append((source, target, _adj[source].get(target, None)))

	edge_list = [e for e in edge_list if e[2] is not None]

	subgraph = graph.__class__()

	subgraph.add_nodes_from(node_list)

	subgraph.add_edges_from(edge_list)

	subgraph.graph.update(graph.graph)

	return subgraph

def subgraph1(graph, nodes):

	subgraph = graph.__class__()

	subgraph.add_nodes_from((n, graph.nodes[n]) for n in nodes)

	subgraph.add_edges_from((n, nbr, d)
		for n, nbrs in graph.adj.items() if n in nodes
		for nbr, d in nbrs.items() if nbr in nodes
		)

	subgraph.graph.update(graph.graph)

	return subgraph


def supergraph(graphs):

	supergraph = graphs[0].__class__()

	nodes = []

	edges = []

	names = []

	show = True

	for graph in graphs:

		for source, adj in graph._adj.items():

			names.append(source)

			coords_s = (graph._node[source]['x'], graph._node[source]['y'])

			nodes.append((coords_s, graph._node[source]))

			for target, edge in adj.items():

				coords_t = (graph._node[target]['x'], graph._node[target]['y'])

				edges.append((coords_s, coords_t, edge))

	supergraph.add_nodes_from(nodes)

	supergraph.add_edges_from(edges)

	supergraph = nx.relabel_nodes(
		supergraph, {k: names[idx] for idx, k in enumerate(supergraph.nodes)}
		)

	return supergraph