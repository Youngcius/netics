import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt

from unionfind import utils, monolithic

from rich import console
import json

with open('../bench/edge_weights.json', 'r') as f:
    weights = json.load(f)

console = console.Console()

# d = 5
# r = 300

d = 7
r = 100
p_data = 0.01
p_meas = 0.05

# d = 11
# r = 100
# p_data = 0.001
# p_meas = 0.001

trial = 0
logical_errors = []
for _ in range(10):
    trial += 1
    # g = utils.gene_rep_decoding_graph(d, r, **weights)
    # decoder = monolithic.MonoRepDecoder(g, d, r)

    g = utils.gene_surf_decoding_graph(d, r, p_data, p_meas)
    decoder = monolithic.MonoSurfDecoder(g, d, r)

    console.rule('Trial {}'.format(trial))

    decoder.decode()

    if decoder.logical_error is None:
        console.print('Decoding failed', style='bold blue')
    #     _ = decoder.visualize_decoding_graph(True, True)
    #     plt.title('Decoding Graph')
    #     plt.savefig('decoding_graph.png', dpi=350)
    #
    #     _ = decoder.visualize_result_graph(True, True)
    #     plt.title('Result graph')
    #     plt.savefig('result_graph.png', dpi=350)
    #
    #     break

    logical_errors.append(decoder.logical_error)

    console.print('Decoding finished in {} epochs'.format(decoder.num_epochs))
    console.print('Number of fully grown edges: {}/{}'.format(
        len([edge for edge in g.edges if decoder.decoding_graph.edges[edge]['growth'] >= g.edges[edge]['weight']]),
        g.number_of_edges()))
    console.print('Logical error: {}'.format(decoder.logical_error))
    console.print('Guess errors:', decoder.guessed_error)
    console.print('Actual errors:', decoder.actual_error)
    console.print('XOR result:', decoder.guessed_error.dot(decoder.actual_error))
    console.print()

console.print(logical_errors)
