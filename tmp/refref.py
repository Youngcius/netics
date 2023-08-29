import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import networkx as nx
from unionfind import utils
from rich import console

"""
add visual nodes and more edges
"""
console = console.Console()
code_distance = 7

xdim, ydim = code_distance // 2, code_distance + 1
# create a temporary 1-round decoding graph to acquire projected edges
g = nx.Graph(utils.gene_surf_decoding_graph(code_distance, 1).edges)
n = g.number_of_nodes()
pseudo_node = n - 1
visual_node = n
for i in g.neighbors(pseudo_node):
    g.add_node(visual_node)
    g.add_edge(i, visual_node)
    if i % xdim == 0 and i // xdim < ydim - 2:
        g.add_edge(i + 2 * xdim, visual_node)
    if i % xdim == xdim - 1 and i // xdim > 1:
        g.add_edge(i - 2 * xdim, visual_node)
    visual_node += 1
g.remove_node(pseudo_node)
console.print(g.number_of_nodes(), g.number_of_edges())
console.print(g.edges)


# projected_edges = {edge: 0 for edge in g.edges}


def sort_edge(edge):
    edge[0], edge[1] = min(*edge), max(*edge)
    if edge[1] > pseudo_node:
        edge[1] -= 1

# sorted_edges = sorted(g.edges, key=sort_edge)  # sorted by data qubit index

# console.print(sorted_edges)
