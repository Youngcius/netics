"""
Classes defined in this file are distributed Union-Find decoders running on CPU
    DistDecoder: distributed Union-Find decoder base class
    DistRepDecoder: distributed Union-Find decoder for repetition code
    DistSurfDecoder: distributed Union-Find decoder for surface code

TODO: currently this distributed decoder is in form of emulation, not realistic parallel distributed processing
"""
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from alidecoding.decoders import Decoder, RepDecoder, SurfDecoder
from alidecoding.utils import proj_3d_to_2d
from alidecoding.syndromes import replace_surf_pseudo_with_visual, replace_rep_pseudo_with_visual


# from rich import console

# console = console.Console()


class DistDecoder(Decoder):
    """Distributed Union-Find decoder (emulating distributed processing)"""

    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int) -> None:
        super().__init__(decoding_graph, code_distance, num_rounds)
        nx.set_edge_attributes(self.decoding_graph, 0, 'growth')
        nx.set_node_attributes(self.decoding_graph, dict(zip(range(self.decoding_graph.number_of_nodes()),
                                                             range(self.decoding_graph.number_of_nodes()))), 'id')
        nx.set_node_attributes(self.decoding_graph, dict(zip(range(self.decoding_graph.number_of_nodes()),
                                                             range(self.decoding_graph.number_of_nodes()))), 'parent')
        nx.set_node_attributes(self.decoding_graph, dict(zip(range(self.decoding_graph.number_of_nodes()),
                                                             range(self.decoding_graph.number_of_nodes()))), 'cid')
        nx.set_node_attributes(self.decoding_graph, nx.get_node_attributes(self.decoding_graph, 'meas'), 'odd')
        nx.set_node_attributes(self.decoding_graph, nx.get_node_attributes(self.decoding_graph, 'meas'), 'st_odd')
        nx.set_node_attributes(self.decoding_graph, True, 'consistency')

    def decode(self):
        self.num_epochs = 0
        self.num_inner_epochs = []

        while True:
            self.num_epochs += 1
            inner_epoch = 0
            # console.rule('Epoch {}'.format(self.num_epochs))
            self.grow()
            while True:
                inner_epoch += 1
                self.merge()
                self.check()
                if all(nx.get_node_attributes(self.decoding_graph, 'consistency').values()):
                    break
            # console.print('Finished in {} inner epochs'.format(inner_epoch))
            self.num_inner_epochs.append(inner_epoch)
            if not any(nx.get_node_attributes(self.decoding_graph, 'odd').values()):
                break

        self.span_trees()

        self.peel_trees()

        self.cal_result()

    def grow(self):
        """
        Edge-wise growing stage
        ---
        Grow an edge if
            1) growth < weight
            2) incident nodes are in different clusters
            3) at least one of the incident nodes belongs to an odd cluster
        """
        edge_growths = nx.get_edge_attributes(self.decoding_graph, 'growth')
        for edge in edge_growths:
            if (edge_growths[edge] < self.decoding_graph.edges[edge]['weight'] and
                    self.decoding_graph.nodes[edge[0]]['cid'] != self.decoding_graph.nodes[edge[1]]['cid']):
                if self.decoding_graph.nodes[edge[0]]['odd']:
                    edge_growths[edge] += 1
                if self.decoding_graph.nodes[edge[1]]['odd']:
                    edge_growths[edge] += 1

        nx.set_edge_attributes(self.decoding_graph, edge_growths, 'growth')

    def merge(self):
        """
        Merge clusters according to fully grown edges
        ---
        There are 3 substages in the merging procedure:
            1. update cid and parent
            2. update st_odd
            3. update odd
        """

        def xor_iterable(*args):
            """XOR of an iterable"""
            return sum(args) % 2

        # update cid and parent
        node_cids = nx.get_node_attributes(self.decoding_graph, 'cid')
        node_parents = nx.get_node_attributes(self.decoding_graph, 'parent')
        for node in range(self.decoding_graph.number_of_nodes()):
            for nb in self._neighbors_with_fully_grown_edges(node):
                if self.decoding_graph.nodes[nb]['cid'] < node_cids[node]:
                    node_cids[node] = self.decoding_graph.nodes[nb]['cid']

                    node_parents[node] = nb
        nx.set_node_attributes(self.decoding_graph, node_cids, 'cid')
        nx.set_node_attributes(self.decoding_graph, node_parents, 'parent')

        # update st_odd
        node_st_odds = nx.get_node_attributes(self.decoding_graph, 'st_odd')
        for node in range(self.decoding_graph.number_of_nodes()):
            node_st_odds[node] = xor_iterable(self.decoding_graph.nodes[node]['meas'],
                                              *[self.decoding_graph.nodes[child]['st_odd'] for child in
                                                self._children_from_neighbors(node)])
        nx.set_node_attributes(self.decoding_graph, node_st_odds, 'st_odd')

        # update odd
        node_odds = nx.get_node_attributes(self.decoding_graph, 'odd')
        for node in range(self.decoding_graph.number_of_nodes()):
            if self.decoding_graph.nodes[node]['parent'] == node:
                node_odds[node] = self.decoding_graph.nodes[node]['st_odd']
            else:
                node_odds[node] = self.decoding_graph.nodes[self.decoding_graph.nodes[node]['parent']]['odd']
        nx.set_node_attributes(self.decoding_graph, node_odds, 'odd')

    def check(self):
        """
        Check each node properties are consistent or not
        ---
        There are 3 consistency conditions to check between neighbor nodes:
            1) cid and odd must bee the same within one cluster
            2) st_odd must be consistent with parenthood
            3) st_odd must be equal to odd if the node is a root
        """

        def xor_iterable(*args):
            """XOR of an iterable"""
            return sum(args) % 2

        for node in range(self.decoding_graph.number_of_nodes()):
            if any([(self.decoding_graph.nodes[node]['cid'] != self.decoding_graph.nodes[nb]['cid'] or
                     self.decoding_graph.nodes[node]['odd'] != self.decoding_graph.nodes[nb]['odd'])
                    for nb in self._neighbors_with_fully_grown_edges(node)]):
                self.decoding_graph.nodes[node]['consistency'] = False
            elif self.decoding_graph.nodes[node]['st_odd'] != xor_iterable(
                    self.decoding_graph.nodes[node]['meas'],
                    *[self.decoding_graph.nodes[child]['st_odd'] for child in self._children_from_neighbors(node)]):
                self.decoding_graph.nodes[node]['consistency'] = False
            elif (self.decoding_graph.nodes[node]['parent'] == node and
                  self.decoding_graph.nodes[node]['odd'] != self.decoding_graph.nodes[node]['st_odd']):
                self.decoding_graph.nodes[node]['consistency'] = False
            else:
                self.decoding_graph.nodes[node]['consistency'] = True

    def _children_from_neighbors(self, u):
        """Find all v whose parent is u"""
        return [v for v in self.decoding_graph.neighbors(u) if self.decoding_graph.nodes[v]['parent'] == u]

    def _neighbors_with_fully_grown_edges(self, u):
        return [v for v in self.decoding_graph.neighbors(u) if
                self.decoding_graph.edges[(u, v)]['growth'] >= self.decoding_graph.edges[(u, v)]['weight']]

    def span_trees(self):
        """Generate spanning trees by parenthood"""
        parenthood = []
        for edge in self.decoding_graph.edges:
            if (self.decoding_graph.nodes[edge[0]]['parent'] == edge[1] or
                    self.decoding_graph.nodes[edge[1]]['parent'] == edge[0]):
                parenthood.append(edge)
        self.spanning_trees = nx.edge_subgraph(self.decoding_graph.copy(), parenthood)


class DistRepDecoder(DistDecoder, RepDecoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int):
        super().__init__(decoding_graph, code_distance, num_rounds)

    def visualize_intermediate_result_graph(self, with_pseudo_ancilla=True, with_labels=True):
        """
        Visualize the intermediate result graph, i.e., the cluster growing and merging status
        ---
        orange edge: fully grown edge (growth >= weight)
        grey edge: growing edge (growth < weight)
        node label: cluster id (cid)
        green node: odd == 1, and has detection event (meas == 1)
        blue node: odd == 1, but has no detection event (meas == 0)
        lightgreen node: odd == 0, but has detection event (meas == 1)
        lightblue node: odd == 0, and has no detection event (meas == 0)
        node with red edge: inconsistent node
        edge width: if not fully grown: growth / weight * 2; otherwise: 2
        """
        g = self.decoding_graph.copy()
        n = g.number_of_nodes()
        if with_pseudo_ancilla:
            g = replace_rep_pseudo_with_visual(g, self.code_distance)
        else:
            g.remove_node(g.number_of_nodes() - 1)

        fig, _ = plt.subplots(figsize=(5, 0.8 * 5 * self.num_rounds / self.code_distance))
        edge_widths = [min(g.edges[edge]['growth'], g.edges[edge]['weight']) / g.edges[edge]['weight'] * 2
                       for edge in g.edges]
        edge_colors = ['orange' if g.edges[edge]['growth'] >= g.edges[edge]['weight'] else 'grey' for edge in g.edges]
        node_colors = []
        for node in g.nodes:
            if g.nodes[node]['odd'] == 1:
                if g.nodes[node]['meas'] == 1:
                    node_colors.append('green')
                else:
                    node_colors.append('blue')
            else:
                if g.nodes[node]['meas'] == 1:
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightblue')
        edgecolors = [col if g.nodes[node]['consistency'] else 'red' for node, col in zip(g.nodes, node_colors)]
        labels = {node: g.nodes[node]['cid'] for node in g.nodes}
        nx.draw(g, with_labels=with_labels, pos=nx.get_node_attributes(g, 'pos'), node_size=90, font_size=8,
                node_color=node_colors, edge_color=edge_colors, width=edge_widths, labels=labels, edgecolors=edgecolors)
        return fig


class DistSurfDecoder(DistDecoder, SurfDecoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int):
        super().__init__(decoding_graph, code_distance, num_rounds)

    def visualize_intermediate_result_graph(self, with_pseudo_ancilla=True, with_labels=True):
        g = self.decoding_graph.copy()
        if with_pseudo_ancilla:
            g = replace_surf_pseudo_with_visual(g, self.code_distance)
        else:
            g.remove_node(g.number_of_nodes() - 1)

        figsize = np.array(proj_3d_to_2d(self.code_distance, self.code_distance, self.num_rounds))
        figsize = figsize / figsize.min() * 5
        fig, _ = plt.subplots(figsize=figsize)

        edge_widths = [min(g.edges[edge]['growth'], g.edges[edge]['weight']) / g.edges[edge]['weight'] * 2
                       for edge in g.edges]
        edge_colors = ['orange' if g.edges[edge]['growth'] >= g.edges[edge]['weight'] else 'grey' for edge in g.edges]
        node_colors = []
        for node in g.nodes:
            if g.nodes[node]['odd'] == 1:
                if g.nodes[node]['meas'] == 1:
                    node_colors.append('green')
                else:
                    node_colors.append('blue')
            else:
                if g.nodes[node]['meas'] == 1:
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightblue')
        edgecolors = [col if g.nodes[node]['consistency'] else 'red' for node, col in zip(g.nodes, node_colors)]
        labels = {node: g.nodes[node]['cid'] for node in g.nodes}
        nx.draw(g, with_labels=with_labels, pos=nx.get_node_attributes(g, 'pos'), node_size=90, font_size=8,
                node_color=node_colors, edge_color=edge_colors, width=edge_widths, labels=labels, edgecolors=edgecolors)
        return fig
