import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import matplotlib.pyplot as plt
from unionfind import utils, monolithic, distributed, gpu

from rich import console

console = console.Console()

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--distance', type=int, default=11, help='code distance (default 21)')
args = parser.parse_args()

##################################################
# Generate decoding graph
d = args.distance
r = d
p_data = 0.01
p_meas = 0.05
g = utils.gene_surf_decoding_graph(d, r, p_data, p_meas)

console.rule('Decoding settings (Surface Code)')
console.print('code distance: {}'.format(d))
console.print('measurement rounds: {}'.format(r))
console.print('number of nodes: {}'.format(g.number_of_nodes()))
console.print('number of edges: {}'.format(g.number_of_edges()))
print()

##################################################
# Monolithic decoding
def monolithic_decoding():
    console.rule('Monolithic decoding')
    decoder = monolithic.MonoSurfDecoder(g, d, r)
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

##################################################
# Distributed decoding
def distributed_decoding():
    console.rule('Distributed decoding')
    decoder = distributed.DistSurfDecoder(g, d, r)
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

##################################################
# GPU decoding
def gpu_decoding():
    console.rule('GPU decoding')
    decoder = gpu.GPUSurfDecoder(g, d, r)
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


if __name__ == '__main__':
    monolithic_decoding()
    if d <= 21:
        distributed_decoding()
    gpu_decoding()
