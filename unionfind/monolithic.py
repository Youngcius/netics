import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set

from . import Decoder, RepDecoder, SurfDecoder
from .utils import replace_rep_pseudo_with_visual, replace_surf_pseudo_with_visual, proj_3d_to_2d


class Node:
    """A class for nodes of cluster trees"""

    def __init__(self, node_id, meas):
        self.id = node_id
        self.meas = meas
        self.parent = self
        self.parity = meas

    def __repr__(self) -> str:
        return 'Node(id={}, meas={}, parity={}, parent={})'.format(self.id, self.meas, self.parity, self.parent.id)

    def is_connected_to(self, node):
        """Check if the node is connected to another node."""
        return self.find_root() == node.find_root()

    def cluster_parity(self):
        """Find the parity of the cluster."""
        return self.find_root().parity

    def find_root(self):
        """Find the root of the node in the cluster (path compression algorithm)"""
        if self.parent != self:
            self.parent = self.parent.find_root()  # TODO: recursive will cause stack overflow when the cluster is large
        return self.parent


class MonoDecoder(Decoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int) -> None:
        super().__init__(decoding_graph, code_distance, num_rounds)
        self.nodes: List[Node] = [Node(node_id, 1) if self.decoding_graph.nodes[node_id]['meas'] else Node(node_id, 0)
                                  for node_id in range(self.decoding_graph.number_of_nodes())]
        self.clusters: Set[Node] = set([node for node in self.nodes if node.meas == 1])
        # fully_grown_edges + growing_edges is the set of all edges
        self.growing_edges: Dict[Tuple[int, int], int] = {edge: 0 for edge in self.decoding_graph.edges}
        self.fully_growth_edges: Dict[Tuple[int, int], int] = {}

    def decode(self):
        self.num_epochs = 0

        while any([node.cluster_parity() for node in self.clusters]):
            self.num_epochs += 1
            self.grow_clusters()

        nx.set_edge_attributes(self.decoding_graph, self.growing_edges, 'growth')
        nx.set_edge_attributes(self.decoding_graph, self.fully_growth_edges, 'growth')

        self.span_trees()

        self.peel_trees()

        self.cal_result()

    def grow_clusters(self):
        """Grow the odd clusters"""
        # grow the edges connected to odd clusters
        for edge in self.growing_edges:
            if self.nodes[edge[0]].cluster_parity():
                self.growing_edges[edge] += 1
            if self.nodes[edge[1]].cluster_parity():
                self.growing_edges[edge] += 1

        # union clusters if satisfied
        for edge, growth in self.growing_edges.items():
            weight = self.decoding_graph.edges[edge]['weight']
            if growth >= weight:
                self.fully_growth_edges.update({edge: weight})
                # print('new fully grown edge:', edge)
                union_clusters(self.nodes[edge[0]], self.nodes[edge[1]])

        for edge in self.fully_growth_edges:
            if edge in self.growing_edges:
                self.growing_edges.pop(edge)

        self.clusters = set([node.find_root() for node in self.clusters])

    def span_trees(self):
        """Generate spanning trees according to self.fully_growth_edges and self.decoding_graph"""
        self.resolved_graph = nx.edge_subgraph(self.decoding_graph.copy(), list(self.fully_growth_edges.keys()))
        self.spanning_trees = nx.minimum_spanning_tree(self.resolved_graph)


class MonoRepDecoder(MonoDecoder, RepDecoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int):
        super().__init__(decoding_graph, code_distance, num_rounds)

    def visualize_intermediate_result_graph(self, with_pseudo_ancilla=True, with_labels=True):
        """
        Visualize the intermediate result graph, i.e., the cluster growing and merging status
        ---
        orange edge: fully grown edge (growth >= weight)
        grey edge: growing edge (growth < weight)
        node label: cluster (root of the cluster) id
        green node: cluster parity is 1, and has detection event (meas == 1)
        blue node: cluster parity is 1, but has no detection event (meas == 0)
        lightgreen node: cluster parity is 0, but has detection event (meas == 1)
        lightblue node: cluster parity is 0, and has no detection event (meas == 0)
        edge width: if not fully grown: growth / weight * 2; otherwise: 2
        """
        # sync dynamic support data to self.decoding_graph first
        nx.set_edge_attributes(self.decoding_graph, self.growing_edges, 'growth')
        nx.set_edge_attributes(self.decoding_graph, self.fully_growth_edges, 'growth')

        g = self.decoding_graph.copy()
        if with_pseudo_ancilla:
            g = replace_rep_pseudo_with_visual(g, self.code_distance)
        else:
            g.remove_node(g.number_of_nodes() - 1)

        fig, _ = plt.subplots(figsize=(5, 0.8 * 5 * self.num_rounds / self.code_distance))
        edge_colors = ['orange' if edge in self.fully_growth_edges else 'grey' for edge in g.edges]
        edge_widths = [min(g.edges[edge]['growth'], g.edges[edge]['weight']) / g.edges[edge]['weight'] * 2
                       for edge in g.edges]
        node_colors = []
        for i in range(g.number_of_nodes()):
            if self.nodes[i].find_root().parity == 1:
                if self.nodes[i].meas == 1:
                    node_colors.append('green')
                else:
                    node_colors.append('blue')
            else:
                if self.nodes[i].meas == 1:
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightblue')

        labels = {node.id: node.find_root().id for node in self.nodes}
        if not with_pseudo_ancilla:
            labels.pop(self.decoding_graph.number_of_nodes() - 1)
        nx.draw(g, with_labels=with_labels, pos=nx.get_node_attributes(g, 'pos'), node_size=90,
                node_color=node_colors, edge_color=edge_colors, width=edge_widths, labels=labels, font_size=8)
        return fig


class MonoSurfDecoder(MonoDecoder, SurfDecoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int):
        super().__init__(decoding_graph, code_distance, num_rounds)

    def visualize_intermediate_result_graph(self, with_pseudo_ancilla=True, with_labels=True):
        # sync dynamic support data to self.decoding_graph first
        nx.set_edge_attributes(self.decoding_graph, self.growing_edges, 'growth')
        nx.set_edge_attributes(self.decoding_graph, self.fully_growth_edges, 'growth')

        g = self.decoding_graph.copy()
        if with_pseudo_ancilla:
            g = replace_surf_pseudo_with_visual(g, self.code_distance)
        else:
            g.remove_node(g.number_of_nodes() - 1)

        figsize = np.array(proj_3d_to_2d(self.code_distance, self.code_distance, self.num_rounds))
        figsize = figsize / figsize.min() * 5
        fig, _ = plt.subplots(figsize=figsize)

        edge_colors = ['orange' if edge in self.fully_growth_edges else 'grey' for edge in g.edges]
        edge_widths = [min(g.edges[edge]['growth'], g.edges[edge]['weight']) / g.edges[edge]['weight'] * 2
                       for edge in g.edges]
        node_colors = []
        for i in range(g.number_of_nodes()):
            if self.nodes[i].find_root().parity == 1:
                if self.nodes[i].meas == 1:
                    node_colors.append('green')
                else:
                    node_colors.append('blue')
            else:
                if self.nodes[i].meas == 1:
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightblue')

        labels = {node.id: node.find_root().id for node in self.nodes}
        if not with_pseudo_ancilla:
            labels.pop(self.decoding_graph.number_of_nodes() - 1)
        nx.draw(g, with_labels=with_labels, pos=nx.get_node_attributes(g, 'pos'), node_size=90,
                node_color=node_colors, edge_color=edge_colors, width=edge_widths, labels=labels, font_size=8)
        return fig


def union_clusters(node1, node2):
    """Union two clusters"""
    root1 = node1.find_root()
    root2 = node2.find_root()
    # print('Unioning clusters', root1, root2)
    if root1 != root2:
        root1.parent = root2
        root2.parity = root1.parity ^ root2.parity
