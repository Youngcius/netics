import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from netics.utils import get_leaf, proj_3d_to_2d
from netics.syndromes import replace_surf_pseudo_with_visual, replace_rep_pseudo_with_visual
from netics.syndromes import visualize_surf_decoding_graph, visualize_rep_decoding_graph
from netics.syndromes import project_surf_errors, project_rep_errors
from netics.stabilizers import surface_code_stabilizer, repetition_code_stabilizer


class Decoder:
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int) -> None:
        self.decoding_graph = decoding_graph
        self.code_distance = code_distance
        self.num_rounds = num_rounds
        self.spanning_trees = nx.Graph()

    def decode(self):
        raise NotImplementedError

    def span_trees(self):
        """Generate self.spanning_trees which is a subgraph of self.decoding_graph"""
        raise NotImplementedError

    def peel_trees(self):
        """Peel the spanning trees to guess error edges"""
        forest = self.spanning_trees.copy()
        while forest.number_of_nodes() > 1:
            leaf = get_leaf(forest)
            if forest.degree(leaf) == 1:  # it might be 0
                parent = list(forest.neighbors(leaf))[0]
                if forest.nodes[leaf]['meas'] == 0:
                    self.spanning_trees.edges[parent, leaf]['error'] = 0
                else:
                    forest.nodes[parent]['meas'] = forest.nodes[parent]['meas'] ^ 1
                    self.spanning_trees.edges[parent, leaf]['error'] = 1
            forest.remove_node(leaf)

    def cal_result(self):
        raise NotImplementedError

    def visualize_decoding_graph(self, with_pseudo_ancilla=True, with_labels=False):
        raise NotImplementedError

    def visualize_intermediate_result_graph(self, with_pseudo_ancilla=True, with_labels=True):
        raise NotImplementedError

    def visualize_result_graph(self, with_pseudo_ancilla=True, with_labels=False):
        raise NotImplementedError


class RepDecoder(Decoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int) -> None:
        super().__init__(decoding_graph, code_distance, num_rounds)
        self.actual_error = project_rep_errors(nx.get_edge_attributes(self.decoding_graph, 'error'),
                                               self.code_distance, self.num_rounds)

    def cal_result(self):
        stabilizer = repetition_code_stabilizer(self.code_distance)
        self.guessed_error = project_rep_errors(nx.get_edge_attributes(self.spanning_trees, 'error'),
                                                self.code_distance, self.num_rounds)
        result = self.guessed_error.dot(self.actual_error)
        assert all(stabilizer.commutes(result)), "!!!!!!!!!!!!!!!!!"

        if result.to_label().count('Z') == self.code_distance:
            self.logical_error = True
        elif result.to_label().count('Z') == 0:
            self.logical_error = False
        else:
            self.logical_error = None
            warnings.warn('Decoding failed')

    def visualize_decoding_graph(self, with_pseudo_ancilla=True, with_labels=False):
        return visualize_rep_decoding_graph(self.decoding_graph, self.code_distance, self.num_rounds,
                                            with_pseudo_ancilla, with_labels)

    def visualize_result_graph(self, with_pseudo_ancilla=True, with_labels=False):
        """
        Visualize the eventual result graph, i.e., the spanning trees and guessed error edges by peeling trees
        ---
        pink edge: edge in the spanning trees
        red edge: guessed error (of course also in the spanning trees)
        lightblue node: node without detection event (meas == 0)
        purple node: node with detection event (meas == 1)
        edge width: if not fully grown: growth / weight * 2; otherwise: 2
        """
        g = self.decoding_graph.copy()
        n = g.number_of_nodes()
        if with_pseudo_ancilla:
            g = replace_rep_pseudo_with_visual(g, self.code_distance)
        else:
            g.remove_node(n - 1)

        fig, _ = plt.subplots(figsize=(5, 0.8 * 5 * self.num_rounds / self.code_distance))
        edge_colors = []
        edge_widths = [min(g.edges[edge]['growth'], g.edges[edge]['weight']) / g.edges[edge]['weight'] * 2
                       for edge in g.edges]
        for edge in g.edges:
            if edge[0] >= n:
                edge = (n - 1, edge[1])
            elif edge[1] >= n:
                edge = (edge[0], n - 1)

            if edge in self.spanning_trees.edges:
                if self.spanning_trees.edges[edge]['error']:
                    edge_colors.append('red')
                else:
                    edge_colors.append('pink')
            else:
                edge_colors.append('grey')
        node_colors = ['purple' if meas == 1 else 'lightblue' for meas in nx.get_node_attributes(g, 'meas').values()]
        nx.draw(g, with_labels=False, node_color=node_colors, pos=nx.get_node_attributes(g, 'pos'),
                edge_color=edge_colors, width=edge_widths, node_size=50)
        return fig


class SurfDecoder(Decoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int) -> None:
        super().__init__(decoding_graph, code_distance, num_rounds)
        self.actual_error = project_surf_errors(nx.get_edge_attributes(self.decoding_graph, 'error'),
                                                self.code_distance, self.num_rounds)

    def cal_result(self):

        stabilizer = surface_code_stabilizer(self.code_distance)

        self.guessed_error = project_surf_errors(nx.get_edge_attributes(self.spanning_trees, 'error'),
                                                 self.code_distance, self.num_rounds)

        result = self.guessed_error.dot(self.actual_error)

        self.logical_error = False

        if not all(stabilizer.commutes(result)):
            self.logical_error = None
            warnings.warn('Decoding failed')

        for x in range(self.code_distance):  # if even-number of Z errors ...
            num_z = 0
            for y in range(self.code_distance):
                if result[x + y * self.code_distance].to_label() == 'Z':
                    num_z += 1
            if num_z % 2 == 1:
                self.logical_error = True
                break

    def visualize_decoding_graph(self, with_pseudo_ancilla=True, with_labels=False):
        return visualize_surf_decoding_graph(self.decoding_graph, self.code_distance, self.num_rounds,
                                             with_pseudo_ancilla, with_labels)

    def visualize_result_graph(self, with_pseudo_ancilla=True, with_labels=False):
        """
        Visualize the eventual result graph, i.e., the spanning trees and guessed error edges by peeling trees
        ---
        pink edge: edge in the spanning trees
        red edge: guessed error (of course also in the spanning trees)
        lightblue node: node without detection event (meas == 0)
        purple node: node with detection event (meas == 1)
        edge width: if not fully grown: growth / weight * 2; otherwise: 2
        """
        g = self.decoding_graph.copy()
        n = g.number_of_nodes()
        if with_pseudo_ancilla:
            g = replace_surf_pseudo_with_visual(g, self.code_distance)
        else:
            g.remove_node(g.number_of_nodes() - 1)

        figsize = np.array(proj_3d_to_2d(self.code_distance, self.code_distance, self.num_rounds))
        figsize = figsize / figsize.min() * 5
        fig, _ = plt.subplots(figsize=figsize)

        edge_colors = []
        edge_widths = [min(g.edges[edge]['growth'], g.edges[edge]['weight']) / g.edges[edge]['weight'] * 2
                       for edge in g.edges]
        for edge in g.edges:
            if edge[0] >= n:
                edge = (n - 1, edge[1])
            elif edge[1] >= n:
                edge = (edge[0], n - 1)

            assert edge in self.decoding_graph.edges, "Edge {} not in decoding graph".format(edge)

            if edge in self.spanning_trees.edges:
                if self.spanning_trees.edges[edge]['error']:
                    edge_colors.append('red')
                else:
                    edge_colors.append('pink')
            else:
                edge_colors.append('grey')
        node_colors = ['purple' if meas == 1 else 'lightblue' for meas in nx.get_node_attributes(g, 'meas').values()]
        nx.draw(g, with_labels=with_labels, node_color=node_colors, pos=nx.get_node_attributes(g, 'pos'),
                edge_color=edge_colors, width=edge_widths, node_size=50)
        return fig
