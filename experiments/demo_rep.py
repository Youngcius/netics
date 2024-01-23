import sys

sys.path.append('..')

# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from alidecoding.unionfind import mono, dist, gpu
from alidecoding import syndromes
from rich import console

console = console.Console()

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--distance', type=int, default=11, help='code distance (default 11)')
parser.add_argument('-r', '--rounds', type=int, default=20, help='measurement rounds (default 20)')
parser.add_argument('-seed', '--seed', type=int, default=None,
                    help='random seed for sampling syndromes')
args = parser.parse_args()

##################################################
# Generate decoding graph
with open('../bench/edge_weights.json', 'r') as f:
    weights = json.load(f)
d = args.distance
r = args.rounds
np.random.seed(args.seed)
g = syndromes.gene_rep_decoding_graph(d, r, **weights)

console.print('Decoding settings (Repetition Code):', style='bold red')
console.print('code distance: {}'.format(d))
console.print('measurement rounds: {}'.format(r))
console.print('number of nodes: {}'.format(g.number_of_nodes()))
console.print('number of edges: {}'.format(g.number_of_edges()))
print()


##################################################
# Monolithic decoding
def monolithic_decoding():
    console.rule('Monolithic decoding')
    decoder = mono.MonoRepDecoder(g, d, r)
    start_time = time.time()
    decoder.decode()
    latency = time.time() - start_time

    console.print('Monolithic decoding finished in {} epochs'.format(decoder.num_epochs))
    console.print('Number of fully grown edges: {}/{}'.format(len(decoder.fully_growth_edges), g.number_of_edges()))
    console.print('Logical error:', decoder.logical_error)
    console.print('Guess errors:\t', decoder.guessed_error)
    console.print('Actual errors:\t', decoder.actual_error)
    console.print('Latency: {:.4f}s'.format(latency))
    print()

    _ = decoder.visualize_result_graph()
    plt.show()


##################################################
# Distributed decoding
def distributed_decoding():
    console.rule('Distributed decoding')
    decoder = dist.DistRepDecoder(g, d, r)
    start_time = time.time()
    decoder.decode()
    latency = time.time() - start_time

    console.print('Distributed decoding finished in {} epochs'.format(decoder.num_epochs))
    console.print('Inner epochs: {}'.format(decoder.num_inner_epochs))
    console.print('Number of fully grown edges: {}/{}'.format(
        len([edge for edge in g.edges if decoder.decoding_graph.edges[edge]['growth'] >= g.edges[edge]['weight']]),
        g.number_of_edges()))
    console.print('Logical error: {}'.format(decoder.logical_error))
    console.print('Guess errors:\t', decoder.guessed_error)
    console.print('Actual errors:\t', decoder.actual_error)
    console.print('Latency: {:.4f}s'.format(latency))
    print()
    _ = decoder.visualize_result_graph()
    plt.show()


##################################################
# GPU decoding
def gpu_decoding():
    console.rule('GPU decoding')
    decoder = gpu.GPURepDecoder(g, d, r)
    start_time = time.time()
    decoder.decode()
    latency = time.time() - start_time

    console.print('GPU decoding finished in {} epochs'.format(decoder.num_epochs))
    console.print('Inner epochs: {}'.format(decoder.num_inner_epochs))
    console.print('Number of fully grown edges: {}/{}'.format(
        len([edge for edge in g.edges if decoder.decoding_graph.edges[edge]['growth'] >= g.edges[edge]['weight']]),
        g.number_of_edges()))
    console.print('Logical error: {}'.format(decoder.logical_error))
    console.print('Guess errors:\t', decoder.guessed_error)
    console.print('Actual errors:\t', decoder.actual_error)
    console.print('Latency: {:.4f}s'.format(latency))
    _ = decoder.visualize_result_graph()
    plt.show()


if __name__ == '__main__':
    monolithic_decoding()
    distributed_decoding()
    gpu_decoding()