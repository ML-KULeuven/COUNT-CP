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


def model_type11(instance: Instance):
    square1 = instance.cp_vars['square1']
    square2 = instance.cp_vars['square2']
    N = instance.constants['size']

    # latin square has rows/cols permutations (alldifferent)
    def latin_sq(square):
        return [[AllDifferent(row) for row in square],
                [AllDifferent(col) for col in square.T]]

    model = Model()
    # each is a latin square
    model += latin_sq(square1)
    model += latin_sq(square2)

    # orthogonal (all pairs distinct)
    model += AllDifferent([square1[i,j]*N + square2[i,j] for i in range(N) for j in range(N)])

    return model


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 11
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (Orth Latin Sq)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type11(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type11(inst)))
