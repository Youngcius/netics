import matplotlib.pyplot as plt
import networkx as nx
import qiskit.quantum_info as qi

from unionfind import utils, monolithic
from rich import console

console = console.Console()

d = 5
r = 3
g = utils.gene_surf_decoding_graph(d, r, 0.01, 0.05)

_ = utils.visualize_surf_decoding_graph(g, d, r, with_pseudo_ancilla=True, with_labels=True)
plt.show()

error = utils.project_surf_errors(nx.get_edge_attributes(g, 'error'), d, r)

console.print(error)

stabilizer = utils.surface_code_stabilizer(d)

console.print(stabilizer)
console.print('decoding result')
console.print(error)
console.print(stabilizer.commutes(error))
console.print(stabilizer.anticommutes(error))

opr = ['I'] * d ** 2
for i in [0, 1, 10, 6]:
    opr[i] = 'Z'
s = qi.Pauli(''.join(opr))  # stabilizer element

opr = ['I'] * d ** 2
for i in [0, 1, 2, 3, 4]:
    opr[i] = 'Z'
# l = qi.Pauli(''.join(opr))  # logical operator
l = qi.Pauli('ZZZZIIZZZIZIZZZIZIIIZIIII')

def is_logical_error(opr):
    logical_error = False
    for x in range(d):
        num_z = 0
        for y in range(d):
            if opr[x + y * d].to_label() == 'Z':
                num_z += 1
        if num_z % 2 == 1:
            logical_error = True
            break
    return logical_error


console.rule('stabilizer element')
console.print(stabilizer.commutes(s))
console.print('logical error', is_logical_error(s))

console.rule('logical operator')
console.print(stabilizer.commutes(l))
console.print('logical error', is_logical_error(l))