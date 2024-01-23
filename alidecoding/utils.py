import dgl
import networkx as nx
import rustworkx as rx
import numpy as np


def networkx_to_dgl(g: nx.Graph) -> dgl.DGLGraph:
    """Construct the DGLGraph instance from networkx Graph instance, including node/edge attributes converting"""
    g = g.to_directed()
    node_attrs = list(g.nodes[list(g.nodes)[0]].keys())
    edge_attrs = list(g.edges[list(g.edges)[0]].keys())
    return dgl.from_networkx(g, node_attrs=node_attrs, edge_attrs=edge_attrs)


def dgl_to_networkx(g: dgl.DGLGraph) -> nx.Graph:
    """Construct the networkx Graph instance from DGLGraph instance, including node/edge attributes converting"""
    nx_graph: nx.MultiDiGraph = dgl.to_networkx(g)
    for nfeat, data in g.ndata.items():
        nx.set_node_attributes(nx_graph, dict(zip(range(g.number_of_nodes()), data.numpy())), nfeat)
    for efeat, data in g.edata.items():
        nx.set_edge_attributes(nx_graph, dict(zip(nx_graph.edges, data.numpy())), efeat)
    nx_graph = nx.Graph(nx_graph)  # convert to undirected graph
    return nx_graph


def rustworkx_to_dgl(g: rx.PyGraph) -> dgl.DGLGraph:
    """Construct the DGLGraph instance from rustworkx PyGraph instance, including node/edge attributes converting"""
    raise NotImplementedError


def dgl_to_rustworkx(g: dgl.DGLGraph) -> rx.PyGraph:
    """Construct the rustworkx PyGraph instance from DGLGraph instance, including node/edge attributes converting"""
    raise NotImplementedError


def networkx_to_rustworkx(g: nx.Graph) -> rx.PyGraph:
    """Construct the rustworkx PyGraph instance from networkx Graph instance, including node/edge attributes converting"""
    return rx.networkx_converter(g, keep_attributes=True)


def rustworkx_to_networkx(g: rx.PyGraph) -> nx.Graph:
    """Construct the networkx Graph instance from rustworkx PyGraph instance, including node/edge attributes converting"""
    raise NotImplementedError


def proj_3d_to_2d(x, y, z):
    """Isometric view: convert 3D position in space to 2D position on canvas"""
    theta = np.pi / 9
    phi = 3 * np.pi / 7
    return (x + y * np.cos(phi) / 2) * np.cos(theta), (y * np.sin(phi) / 2 + 2 * z * np.cos(theta))


def get_leaf(tree: nx.Graph):
    """Get one leaf or isolated node from a tree (in type of networkx.Graph)"""
    for node in tree.nodes:
        if tree.degree(node) <= 1:
            return node
    return None
