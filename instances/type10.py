import glob
import json

import numpy as np
from cpmpy import *

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

# Magic Square --- CSPLib prob019

# A Magic Square of order N is an N x N matrix of values from 1 to N^2, where
# each run, column, and diagonal sum to the same value. This value can be
# calculated as N * (N^2 + 1) / 2.
def model_type10(instance: Instance):
    square = instance.cp_vars['list']
    N = instance.constants['size']

    sum_val = int(N * (N * N + 1) / 2)  # This is what all the columns, rows and diagonals must add up to

    model = Model(
        AllDifferent(square),

        [sum(row) == sum_val for row in square],
        [sum(col) == sum_val for col in square.T],

        sum([square[a, a] for a in range(N)]) == sum_val,  # diagonal TL - BR
        sum([square[a, N - a - 1] for a in range(N)]) == sum_val  # diagonal TR - BL
    )

    return model


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 10
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (magic squares)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type10(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type10(inst)))
