import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import qiskit.quantum_info as qi
from functools import reduce
from typing import Dict, Tuple
from netics.utils import proj_3d_to_2d


def sample_errors(decoding_graph: nx.Graph):
    """Re-sample errors according to edge attribute "prob" of a decoding graph"""
    error_probs = nx.get_edge_attributes(decoding_graph, 'prob')
    edge_errors = {edge: np.random.choice([0, 1], p=[1 - prob, prob]) for edge, prob in error_probs.items()}
    nx.set_edge_attributes(decoding_graph, edge_errors, 'error')
    node_measurements = np.repeat(0, decoding_graph.number_of_nodes())
    for edge, error in edge_errors.items():
        if error:
            node_measurements[edge[0]] ^= 1
            node_measurements[edge[1]] ^= 1
    nx.set_node_attributes(decoding_graph, dict(zip(range(decoding_graph.number_of_nodes()), node_measurements)),
                           'meas')


def gene_rep_decoding_graph(d, r, w_hori, w_vert, w_hypo, w_hori_upper, w_hori_lower, w_bound_right, w_bound_left,
                            w_bound_right_except, w_bound_left_except):
    """Generate arbitrary Repetition Code decoding graph with circuit-level noise model"""
    decoding_graph = nx.Graph()
    decoding_graph.add_nodes_from(range((d - 1) * r + 1))
    n = decoding_graph.number_of_nodes()
    nx.set_node_attributes(decoding_graph, {i: (i % (d - 1), i // (d - 1)) for i in range(n - 1)}, 'pos')
    nx.set_node_attributes(decoding_graph, {n - 1: (d, r // 2)}, 'pos')

    # add lower horizontal edges
    for i in range(d - 2):
        decoding_graph.add_weighted_edges_from([(i, i + 1, w_hori_lower)])
    decoding_graph.add_weighted_edges_from([(0, n - 1, w_hori_lower)])

    # add upper horizontal edges
    for i in range(n - d, n - 2):
        decoding_graph.add_weighted_edges_from([(i, i + 1, w_hori_upper)])
    decoding_graph.add_weighted_edges_from([(n - 2, n - 1, w_hori_upper)])

    # add horizontal edges
    for i in range(d - 1, n - d - 1):
        if (i + 1) % (d - 1) != 0:
            decoding_graph.add_weighted_edges_from([(i, i + 1, w_hori)])

    # add vertical edges
    for i in range(0, n - d):
        decoding_graph.add_weighted_edges_from([(i, i + d - 1, w_vert)])

    # add hypotenuse edges
    for i in range(0, n - d - 1):
        if (i + 1) % (d - 1) != 0:
            decoding_graph.add_weighted_edges_from([(i, i + d, w_hypo)])

    # add left boundary edges
    for i in range(2 * d - 3, n - d, d - 1):
        decoding_graph.add_weighted_edges_from([(i, n - 1, w_bound_left)])
    decoding_graph.add_weighted_edges_from([(d - 2, n - 1, w_bound_left_except)])

    # add right boundary edges
    for i in range(d - 1, n - d, d - 1):
        decoding_graph.add_weighted_edges_from([(i, n - 1, w_bound_right)])
    decoding_graph.add_weighted_edges_from([(n - d, n - 1, w_bound_right_except)])

    # calculate error probabilities of edges
    edge_weights = nx.get_edge_attributes(decoding_graph, 'weight')
    error_probs = {edge: 1 / (np.exp(weight) + 1) for edge, weight in edge_weights.items()}
    nx.set_edge_attributes(decoding_graph, error_probs, 'prob')

    # normalize and round the edge weights
    all_weights = np.array(list(edge_weights.values()))
    coeff = 2 / all_weights.min()
    edge_weights = {edge: int(weight * coeff) for edge, weight in edge_weights.items()}
    nx.set_edge_attributes(decoding_graph, edge_weights, 'weight')

    # randomly assign the detection events based on edge error probabilities
    sample_errors(decoding_graph)

    return decoding_graph


def gene_surf_decoding_graph(d, r, p_data=0.001, p_meas=0.001):
    """Generate arbitrary Surface Code decoding graph with phenomenonological noise model"""
    w_data, w_meas = np.log((1 - p_data) / p_data), np.log((1 - p_meas) / p_meas)
    coeff = 2 / min(w_data, w_meas)
    w_data_round, w_meas_round = int(w_data * coeff), int(w_meas * coeff)
    gs = []
    pseudo_node = (d ** 2 - 1) // 2 * r
    xdim, ydim = d // 2, d + 1
    for k in range(r):
        g = nx.Graph()
        # add nodes
        for y in range(ydim):
            for x in range(xdim):
                g.add_node((x, y, k))
        # add edges
        for x in range(xdim):
            g.add_edges_from([((x, y, k), (x, y + 1, k)) for y in range(ydim - 1)])
            nx.set_edge_attributes(g, {((x, y, k), (x, y + 1, k)): w_data_round for y in range(ydim - 1)}, 'weight')
            nx.set_edge_attributes(g, {((x, y, k), (x, y + 1, k)): p_data for y in range(ydim - 1)}, 'prob')

        for x in range(1, xdim):
            g.add_edges_from([((x, y, k), (x - 1, y + 1, k)) for y in range(0, ydim, 2)])
            nx.set_edge_attributes(g, {((x, y, k), (x - 1, y + 1, k)): w_data_round for y in range(0, ydim, 2)},
                                   'weight')
            nx.set_edge_attributes(g, {((x, y, k), (x - 1, y + 1, k)): p_data for y in range(0, ydim, 2)}, 'prob')

        for x in range(xdim - 1):
            g.add_edges_from([((x, y, k), (x + 1, y + 1, k)) for y in range(1, ydim - 1, 2)])
            nx.set_edge_attributes(g, {((x, y, k), (x + 1, y + 1, k)): w_data_round for y in range(1, ydim - 1, 2)},
                                   'weight')
            nx.set_edge_attributes(g, {((x, y, k), (x + 1, y + 1, k)): p_data for y in range(1, ydim - 1, 2)},
                                   'prob')
            nx.set_edge_attributes(g, {((x, y, k), (x + 1, y + 1, k)): p_data for y in range(1, ydim - 1, 2)}, 'prob')

        # add pseudo nodes and edges
        for y in range(0, ydim, 2):
            g.add_edge((0, y, k), pseudo_node)
            g.add_edge((xdim - 1, y + 1, k), pseudo_node)
            g.edges[((0, y, k), pseudo_node)]['weight'] = w_meas_round
            g.edges[((xdim - 1, y + 1, k), pseudo_node)]['weight'] = w_meas_round
            g.edges[((0, y, k), pseudo_node)]['prob'] = 2 * p_data * (1 - p_data)
            g.edges[((xdim - 1, y + 1, k), pseudo_node)]['prob'] = 2 * p_data * (1 - p_data)

        # set node positions
        for node in g.nodes:
            if node == pseudo_node:
                continue
            if node[1] % 2 == 0:
                g.nodes[node]['pos'] = proj_3d_to_2d(node[0], node[1] // 2, k)
            else:
                g.nodes[node]['pos'] = proj_3d_to_2d(node[0] + 0.5, node[1] // 2 + 0.5, k)

        gs.append(g)

    decoding_graph = reduce(nx.compose, gs)
    x, y = decoding_graph.nodes[(xdim - 1, ydim // 2, r // 2)]['pos']
    decoding_graph.nodes[pseudo_node]['pos'] = x + 0.5, y

    # add vertical edges
    for k in range(r - 1):
        for x in range(xdim):
            decoding_graph.add_edges_from([((x, y, k), (x, y, k + 1)) for y in range(ydim)])
            nx.set_edge_attributes(decoding_graph, {((x, y, k), (x, y, k + 1)): w_meas_round for y in range(ydim)},
                                   'weight')
            nx.set_edge_attributes(decoding_graph, {((x, y, k), (x, y, k + 1)): p_meas for y in range(ydim)}, 'prob')

    decoding_graph = nx.relabel_nodes(decoding_graph, {node: node[0] + node[1] * xdim + node[2] * xdim * ydim
                                                       for node in decoding_graph.nodes if node != pseudo_node})

    # randomly assign the detection events based on edge error probabilities
    sample_errors(decoding_graph)

    return decoding_graph


def project_rep_errors(edge_errors: Dict[Tuple[int, int], int], code_distance: int, num_rounds: int) -> qi.Pauli:
    """Project space-time Repetition Code error edges to the first time slice

    Args:
        edge_errors: dict of error edges with error value (0/1)
        code_distance: code distance of the Repetition Code
        num_rounds: number of time slices of the Repetition Code
    """
    num_nodes = (code_distance - 1) * num_rounds + 1

    projected_edges = {(i, i + 1): 0 for i in range(code_distance - 1)}
    projected_edges.update({(code_distance - 1, 0): 0})

    def project_node(node):
        if node == num_nodes - 1:
            return code_distance - 1
        return node % (code_distance - 1)

    # collect and project the error edges
    for edge, error in edge_errors.items():
        edge = project_node(edge[0]), project_node(edge[1])
        if error and edge[0] != edge[1]:
            if edge not in projected_edges:
                edge = edge[1], edge[0]
            projected_edges[edge] ^= 1
    # return projected_edges
    opr = ['I'] * code_distance
    for i, error in enumerate(projected_edges.values()):
        if error == 1:
            opr[i] = 'Z'
    return qi.Pauli(''.join(opr))


def project_surf_errors(edge_errors: Dict[Tuple[int, int], int], code_distance: int, num_rounds: int) -> qi.Pauli:
    """
    Project space-time Surface Code error edges to the first time slice
    ---
    In presentation of qiskit.quantum_info.Pauli instance, with d^2 primitive Pauli compositions
    Suppose the decoding graph is on basis of X-stabilizer syndrome, so the inferred error edges are Z-type

    Args:
        edge_errors: dict of error edges with error value (0/1)
        code_distance: code distance of the Surface Code
        num_rounds: number of time slices of the Surface Code
    """
    xdim, ydim = code_distance // 2, code_distance + 1
    num_nodes = xdim * ydim * num_rounds + 1

    #  create a temporary 1-round decoding graph to acquire projected edges
    g = nx.Graph(gene_surf_decoding_graph(code_distance, 1).edges)
    projected_edges = {edge: 0 for edge in g.edges}

    def project_node(node):
        if node == num_nodes - 1:  # pseudo_node index for the original space-time decoding graph
            return (code_distance ** 2 - 1) // 2
        return node % (xdim * ydim)

    for edge, error in edge_errors.items():
        edge = project_node(edge[0]), project_node(edge[1])
        if error and edge[0] != edge[1]:
            if edge not in projected_edges:
                edge = edge[1], edge[0]
            projected_edges[edge] ^= 1

    opr = ['I'] * (code_distance ** 2)
    for edge, error in projected_edges.items():
        if error == 0:
            continue
        u, v = min(*edge), max(*edge)
        if v == (code_distance ** 2 - 1) // 2:  # pseudo_node index for the projected graph
            if u % xdim == 0:
                data_qubit = (u + u // xdim // 2) * 2
            else:
                data_qubit = (u + u // xdim // 2 - (code_distance // 2 - 1)) * 2
        else:
            data_qubit = u + v + u // xdim - (code_distance // 2 - 1)
        opr[data_qubit] = 'Z'

    return qi.Pauli(''.join(opr))


def visualize_rep_decoding_graph(decoding_graph, code_distance, num_rounds, with_pseudo_ancilla=True,
                                 with_labels=False):
    """
    Visualize the Repetition Code decoding graph
    ---
    Convert the single pseudo node to multiple visual nodes
    """
    g = decoding_graph.copy()
    if with_pseudo_ancilla:
        g = replace_rep_pseudo_with_visual(g, code_distance)
    else:
        g.remove_node(g.number_of_nodes() - 1)

    fig, _ = plt.subplots(figsize=(5, 0.8 * 5 * num_rounds / code_distance))
    edge_colors = ['red' if error == 1 else 'grey' for error in nx.get_edge_attributes(g, 'error').values()]
    node_colors = ['purple' if meas == 1 else 'lightblue' for meas in nx.get_node_attributes(g, 'meas').values()]
    nx.draw(g, with_labels=with_labels, pos=nx.get_node_attributes(g, 'pos'), edge_color=edge_colors,
            node_size=50, node_color=node_colors)
    return fig


def visualize_surf_decoding_graph(decoding_graph, code_distance, num_rounds, with_pseudo_ancilla=True,
                                  with_labels=False):
    """
    Visualize the Surface Code decoding graph
    ---
    Convert the single pseudo node to multiple visual nodes
    """
    g = decoding_graph.copy()
    if with_pseudo_ancilla:
        g = replace_surf_pseudo_with_visual(g, code_distance)
    else:
        g.remove_node(g.number_of_nodes() - 1)

    figsize = np.array(proj_3d_to_2d(code_distance, code_distance, num_rounds))
    figsize = figsize / figsize.min() * 5
    fig, _ = plt.subplots(figsize=figsize)
    edge_colors = ['red' if error == 1 else 'grey' for error in nx.get_edge_attributes(g, 'error').values()]
    node_colors = ['purple' if meas == 1 else 'lightblue' for meas in nx.get_node_attributes(g, 'meas').values()]
    nx.draw(g, with_labels=with_labels, pos=nx.get_node_attributes(g, 'pos'), edge_color=edge_colors,
            node_size=50, node_color=node_colors)
    return fig


def replace_rep_pseudo_with_visual(g: nx.Graph, d: int):
    """Convert the single pseudo node to multiple visual nodes for Repetition Code decoding graph visualization"""
    n = g.number_of_nodes()
    pseudo_node = n - 1
    visual_node = n
    for i in g.neighbors(pseudo_node):
        g.add_node(visual_node, **{key: g.nodes[pseudo_node][key] for key in g.nodes[pseudo_node]})
        g.add_edge(i, visual_node, **{key: g.edges[(i, pseudo_node)][key] for key in g.edges[(i, pseudo_node)]})
        if i % (d - 1) == 0:
            g.nodes[visual_node]['pos'] = g.nodes[i]['pos'][0] - 1.2, g.nodes[i]['pos'][1] - 0.2
        else:
            g.nodes[visual_node]['pos'] = g.nodes[i]['pos'][0] + 1.2, g.nodes[i]['pos'][1] + 0.2
        visual_node += 1
    g.remove_node(pseudo_node)
    return g


def replace_surf_pseudo_with_visual(g: nx.Graph, d: int):
    """Convert the single pseudo node to multiple visual nodes for Surface Code decoding graph visualization"""
    n = g.number_of_nodes()
    pseudo_node = n - 1
    xdim, ydim = d // 2, d + 1
    visual_node = n
    for i in g.neighbors(pseudo_node):
        g.add_node(visual_node, **{key: g.nodes[pseudo_node][key] for key in g.nodes[pseudo_node]})
        g.add_edge(i, visual_node, **{key: g.edges[(i, pseudo_node)][key] for key in g.edges[(i, pseudo_node)]})
        x, y, z = (i % (xdim * ydim)) % xdim, (i % (xdim * ydim)) // xdim, i // (xdim * ydim)
        if x == 0:
            g.nodes[visual_node]['pos'] = proj_3d_to_2d(x - 0.5 - 0.2, y // 2 + 0.5 - 0.2, z)
        else:
            g.nodes[visual_node]['pos'] = proj_3d_to_2d(x + 1 + 0.2, y // 2 + 0.2, z)
        visual_node += 1
    g.remove_node(pseudo_node)
    return g
