import numpy as np

from heapq import heappush, heappop
from itertools import count

def visit(source, conditions, adj, visited, counter, order, stack, root, in_component):

	visited[source] = True

	depth = next(counter)
	order[source] = depth
	stack.append((depth, source))

	root[source] = source
	in_component[source] = False

	for target, edge in adj[source].items():

		feasible = True

		for condition in conditions:

			feasible *= condition(edge)

		if feasible:

			if not visited[target]:

				visit(
					target, conditions, adj, visited, counter,
					order, stack, root, in_component
					)

			if not in_component[root[target]]:

				# print(source, target, order[root[source]], root[target])

				if order[root[source]] <= order[root[target]]:

					root[source] = root[source]

				else:

					root[source] = root[target]

	if root[source] == source:
		while target != source:

			depth, target = stack.pop()
			in_component[target] = True

	# if root[source] == source:
	# 	while stack[-1][0] >= order[source]:

	# 		depth, target = stack.pop()
	# 		in_component[target] = True

	# elif root[source] 


def strongly_connected_components(graph, conditions = []):

	adj = graph._adj
	counter = count()
	stack = []
	visited = {source: False for source in graph.nodes}
	order = {}
	root = {}
	in_component = {}

	for source in graph.nodes:
		if not visited[source]:

			visit(
				source, conditions, adj, visited, counter, order, stack, root, in_component
				)

	components = {}

	for node, root in root.items():

		if root in components:

			components[root].append(node)

		else:

			components[root] = [node]

	components_list = [v for v in components.values()]
	indices = np.flip(np.argsort([len(v) for v in components.values()]))

	return [components_list[idx] for idx in indices]



