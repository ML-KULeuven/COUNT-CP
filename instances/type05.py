import glob
import json

import numpy as np
from cpmpy import *
import learner
from learn import learn
from instance import Instance

"""
    Graph coloring

    Arguments:
    - formatTemplate: dict as in the challenge, containing:
        "list": list of dicts of 'high', 'low', 'type'
    - inputData: dict as in challenge, containing:
        "list": list of dicts with 'nodeA', 'nodeB'
"""



def model_type05(instance: Instance):
    puzzle = instance.cp_vars['array']
    s = instance.input_data['size']
    n = s*s

    model = Model(
        # Constraints on rows and columns
        [AllDifferent(row) for row in puzzle],
        [AllDifferent(col) for col in puzzle.T],  # numpy's Transpose
    )

    # Constraints on blocks
    for i in range(0, n, s):
        for j in range(0, n, s):
            model += AllDifferent(puzzle[i:i + s, j:j + s])  # python's indexing

    # Constraints on values (cells that are not empty)
    given = np.zeros((n,n), dtype=int)
    for d in instance.input_data['preassigned']:  # [{'column': int, 'row': int, 'value': int}]
        given[d['row'], d['column']] = d['value']
    model += (puzzle[given != 0] == given[given != 0])  # numpy's indexing

    return model

if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 5
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (sudoku pre-assgn)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type05(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type05(inst)))

