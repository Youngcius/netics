from qiskit import quantum_info as qi


def repetition_code_stabilizer(d: int) -> qi.PauliList:
    """Generate the stabilizer of the repetition code of distance d."""
    ops = []
    for i in range(d - 1):
        opr = ['I'] * d
        opr[i], opr[i + 1] = 'Z', 'Z'
        ops.append(''.join(opr))
    return qi.PauliList(ops)


def surface_code_stabilizer(d: int) -> qi.PauliList:
    """Generate the stabilizer of the surface code of distance d."""

    ops = []

    def convert_2d_to_int(x, y):
        return x + y * d

    ##################################################
    ####### set Z-type stabilizer generators #########
    # set intra stabilizer generators
    start = 0
    for y in range(d - 1):
        if start == 0:
            for x in range(0, d - 1, 2):
                opr = ['I'] * d ** 2
                opr[convert_2d_to_int(x, y)] = 'Z'
                opr[convert_2d_to_int(x + 1, y)] = 'Z'
                opr[convert_2d_to_int(x, y + 1)] = 'Z'
                opr[convert_2d_to_int(x + 1, y + 1)] = 'Z'
                ops.append(''.join(opr))
        else:
            for x in range(1, d, 2):
                opr = ['I'] * d ** 2
                opr[convert_2d_to_int(x, y)] = 'Z'
                opr[convert_2d_to_int(x + 1, y)] = 'Z'
                opr[convert_2d_to_int(x, y + 1)] = 'Z'
                opr[convert_2d_to_int(x + 1, y + 1)] = 'Z'
                ops.append(''.join(opr))
        start ^= 1

    # set boundary stabilizer generators
    for y in range(1, d, 2):
        opr = ['I'] * d ** 2
        opr[convert_2d_to_int(0, y)] = 'Z'
        opr[convert_2d_to_int(0, y + 1)] = 'Z'
        ops.append(''.join(opr))

    for y in range(0, d - 1, 2):
        opr = ['I'] * d ** 2
        opr[convert_2d_to_int(d - 1, y)] = 'Z'
        opr[convert_2d_to_int(d - 1, y + 1)] = 'Z'
        ops.append(''.join(opr))

    ##################################################
    ####### set X-type stabilizer generators #########
    # set intra stabilizer generators
    start = 1
    for y in range(d - 1):
        if start == 0:
            for x in range(0, d - 1, 2):
                opr = ['I'] * d ** 2
                opr[convert_2d_to_int(x, y)] = 'X'
                opr[convert_2d_to_int(x + 1, y)] = 'X'
                opr[convert_2d_to_int(x, y + 1)] = 'X'
                opr[convert_2d_to_int(x + 1, y + 1)] = 'X'
                ops.append(''.join(opr))
        else:
            for x in range(1, d, 2):
                opr = ['I'] * d ** 2
                opr[convert_2d_to_int(x, y)] = 'X'
                opr[convert_2d_to_int(x + 1, y)] = 'X'
                opr[convert_2d_to_int(x, y + 1)] = 'X'
                opr[convert_2d_to_int(x + 1, y + 1)] = 'X'
                ops.append(''.join(opr))
        start ^= 1

    # set boundary stabilizer generators
    for x in range(0, d - 1, 2):
        opr = ['I'] * d ** 2
        opr[convert_2d_to_int(x, 0)] = 'X'
        opr[convert_2d_to_int(x + 1, 0)] = 'X'
        ops.append(''.join(opr))

    for x in range(1, d, 2):
        opr = ['I'] * d ** 2
        opr[convert_2d_to_int(x, d - 1)] = 'X'
        opr[convert_2d_to_int(x + 1, d - 1)] = 'X'
        ops.append(''.join(opr))

    return qi.PauliList(ops)
