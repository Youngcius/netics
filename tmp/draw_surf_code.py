from unionfind import utils
import networkx as nx
import matplotlib.pyplot as plt
import json

# with open('bench/edge_weights.json', 'r') as f:
#     weights = json.load(f)
#
# d = 11
# r = 20
# g = utils.gene_rep_decoding_graph(d, r, **weights)
# print(g.number_of_nodes(), g.number_of_edges())
#
# fig = utils.visualize_rep_decoding_graph(g, d, r)
# plt.show()


d = 7
r = 1
g = utils.gene_surf_decoding_graph(d, r, 0.01, 0.01)
fig = utils.visualize_surf_decoding_graph(g, d, r, True, True)
plt.show()

errors = utils.project_surf_errors(nx.get_edge_attributes(g, 'error'), d, r)
print(errors)
