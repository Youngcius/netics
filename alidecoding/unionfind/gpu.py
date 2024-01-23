"""
Classes defined in this file are GPU-based Union-Find decoders using DGL framework
    GPUDecoder: GPU Union-Find decoder base class
    GPURepDecoder: GPU Union-Find decoder for repetition code
    GPUSurfDecoder: GPU Union-Find decoder for surface code
"""
import time
import torch
import networkx as nx

from alidecoding.utils import networkx_to_dgl, dgl_to_networkx
from alidecoding.unionfind.dist import DistDecoder, DistRepDecoder, DistSurfDecoder


# from rich import console

# console = console.Console()


class GPUDecoder(DistDecoder):
    """Distributed Union-Find decoder running on GPU using DGL framework"""

    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int) -> None:
        super().__init__(decoding_graph, code_distance, num_rounds)
        self.dgl_graph = networkx_to_dgl(self.decoding_graph)
        self.gpu_latency = 0

    def decode(self):
        """If there is GPU available, use GPU to decode, otherwise use CPU to decode"""
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.dgl_graph = self.dgl_graph.to(device)  # move to GPU (if available)
        self.num_epochs = 0
        self.num_inner_epochs = []
        start_time = time.time()

        while True:
            self.num_epochs += 1
            inner_epoch = 0
            # console.rule('Epoch {}'.format(self.num_epochs))
            self.grow()
            while True:
                inner_epoch += 1
                self.merge()
                self.check()
                if torch.all(self.dgl_graph.ndata['consistency']):
                    break
            # console.print('Finished in {} inner epochs'.format(inner_epoch))
            self.num_inner_epochs.append(inner_epoch)
            if not torch.any(self.dgl_graph.ndata['odd']):
                break
        self.gpu_latency = time.time() - start_time

        self.dgl_graph = self.dgl_graph.cpu()  # move back to CPU

        self.decoding_graph = dgl_to_networkx(self.dgl_graph)  # copy node/edge attributes to networkx graph

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

        def update_edge_growth(edges):
            src_odd = ((edges.data['growth'] <= edges.data['weight']) &
                       (edges.src['cid'] != edges.dst['cid']) &
                       (edges.src['odd'] == 1))
            dst_odd = ((edges.data['growth'] <= edges.data['weight']) &
                       (edges.src['cid'] != edges.dst['cid']) &
                       (edges.dst['odd'] == 1))
            growth = edges.data['growth'].clone()
            growth[src_odd] += 1
            growth[dst_odd] += 1
            return {'growth': growth}

        self.dgl_graph.apply_edges(update_edge_growth)

    def merge(self):
        """
        Merging clusters according to fully grown edges
        ---
        There are 3 substages in the merging procedure (corresponding to 3 message-passing periods):
            1. update cid and parent
            2. update st_odd
            3. update odd
        """

        def msg_func_merge_1(edges):
            return {'cid': edges.src['cid'], 'id': edges.src['id'],
                    'is_nb': edges.data['growth'] >= edges.data['weight']}

        def reduce_func_merge_1(nodes):
            mask = (nodes.mailbox['cid'] < nodes.data['cid'].view(-1, 1)) & nodes.mailbox['is_nb']
            min_cid = nodes.mailbox['cid'].clone()
            min_cid[~mask] = self.dgl_graph.num_nodes()
            min_cid, argmin_cid = torch.min(min_cid, 1)
            min_id = torch.gather(nodes.mailbox['id'], 1, argmin_cid.view(-1, 1)).view(-1)
            mask_1d = torch.any(mask, 1)
            cid = nodes.data['cid'].clone()
            parent = nodes.data['parent'].clone()
            cid[mask_1d] = min_cid[mask_1d]
            parent[mask_1d] = min_id[mask_1d]
            return {'cid': cid, 'parent': parent}

        def msg_func_merge_2(edges):
            return {'st_odd': edges.src['st_odd'], 'is_child': edges.src['parent'] == edges.dst['id']}

        def reduce_func_merge_2(nodes):
            st_odd = ((nodes.mailbox['st_odd'] * nodes.mailbox['is_child']).sum(1, dtype=nodes.mailbox[
                'st_odd'].dtype) + nodes.data['meas']) % 2
            return {'st_odd': st_odd}

        def msg_func_merge_3(edges):
            return {'id': edges.src['id'], 'odd': edges.src['odd']}

        def reduce_func_merge_3(nodes):
            self_parent = nodes.data['parent'] == nodes.data['id']
            odd = nodes.data['odd'].clone()
            odd[self_parent] = nodes.data['st_odd'][self_parent]
            other_parent = nodes.mailbox['id'] == nodes.data['parent'].view(-1, 1)
            assert torch.count_nonzero(self_parent) + torch.count_nonzero(other_parent) == self_parent.size()[
                0], "Parenthood error"
            assert torch.allclose(self_parent | other_parent.any(1), torch.ones_like(self_parent)), "Parenthood error"
            odd[~self_parent] = nodes.mailbox['odd'][other_parent]
            return {'odd': odd}

        self.dgl_graph.update_all(msg_func_merge_1, reduce_func_merge_1)
        self.dgl_graph.update_all(msg_func_merge_2, reduce_func_merge_2)
        self.dgl_graph.update_all(msg_func_merge_3, reduce_func_merge_3)

    def check(self):
        """
        Check each node properties are consistent or not
        ---
        There are 3 consistency conditions to check between neighbor nodes:
            1) cid and odd must bee the same within one cluster
            2) st_odd must be consistent with parenthood
            3) st_odd must be equal to odd if the node is a root
        """

        def msg_func_check_1(edges):
            return {'cid': edges.src['cid'], 'odd': edges.src['odd'],
                    'is_nb': edges.data['growth'] >= edges.data['weight']}

        def reduce_func_check_1(nodes):
            unsatisfied = (((nodes.mailbox['cid'] != nodes.data['cid'].view(-1, 1)) |
                            (nodes.mailbox['odd'] != nodes.data['odd'].view(-1, 1))) &
                           nodes.mailbox['is_nb']).any(1)
            consistency = nodes.data['consistency'].clone()
            consistency[unsatisfied] = False
            return {'consistency': consistency}

        def msg_func_check_2(edges):
            return {'st_odd': edges.src['st_odd'], 'is_child': edges.src['parent'] == edges.dst['id']}

        def reduce_func_check_2(nodes):
            unsatisfied = nodes.data['st_odd'] != ((nodes.mailbox['st_odd'] * nodes.mailbox['is_child']).sum(1) +
                                                   nodes.data['meas']) % 2
            consistency = nodes.data['consistency'].clone()
            consistency[unsatisfied] = False
            return {'consistency': consistency}

        def check_3(nodes):
            unsatisfied = (nodes.data['parent'] == nodes.data['id']) & (nodes.data['odd'] != nodes.data['st_odd'])
            consistency = nodes.data['consistency'].clone()
            consistency[unsatisfied] = False
            return {'consistency': consistency}

        self.dgl_graph.ndata['consistency'].fill_(True)
        self.dgl_graph.update_all(msg_func_check_1, reduce_func_check_1)
        self.dgl_graph.update_all(msg_func_check_2, reduce_func_check_2)
        self.dgl_graph.apply_nodes(check_3)


class GPURepDecoder(GPUDecoder, DistRepDecoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int):
        super().__init__(decoding_graph, code_distance, num_rounds)

    def visualize_intermediate_result_graph(self, with_pseudo_ancilla=True, with_labels=True):
        # sync dynamic support data to self.decoding_graph first
        self.dgl_graph = self.dgl_graph.cpu()
        self.decoding_graph = dgl_to_networkx(self.dgl_graph)
        return super().visualize_intermediate_result_graph(with_pseudo_ancilla, with_labels)


class GPUSurfDecoder(GPUDecoder, DistSurfDecoder):
    def __init__(self, decoding_graph: nx.Graph, code_distance: int, num_rounds: int):
        super().__init__(decoding_graph, code_distance, num_rounds)

    def visualize_intermediate_result_graph(self, with_pseudo_ancilla=True, with_labels=True):
        self.dgl_graph = self.dgl_graph.cpu()
        self.decoding_graph = dgl_to_networkx(self.dgl_graph)
        return super().visualize_intermediate_result_graph(with_pseudo_ancilla, with_labels)
