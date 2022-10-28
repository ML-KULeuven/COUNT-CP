import glob
import json
from math import floor

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


# Balanced Incomplete Block Design (BIBD) --- CSPLib prob028

# A BIBD is defined as an arrangement of v distinct objects into b blocks such
# that each block contains exactly k distinct objects, each object occurs in
# exactly r different blocks, and every two distinct objects occur together in
# exactly lambda blocks. Another way of defining a BIBD is in terms of its
# incidence matrix, which is a v by b binary matrix with exactly r ones per row,
# k ones per column, and with a scalar product of lambda 'l' between any pair of
# distinct rows.

def get_model(v, b, r, k, l):
    matrix = Matrix(v, b)
    model = Model(
        [Sum(row) == r for row in matrix.row],  # every row adds up to r
        [Sum(col) == k for col in matrix.col],  # every column adds up to k

        # the scalar product of every pair of columns adds up to l
        [Sum([(row[col_i] * row[col_j]) for row in matrix.row]) == l
         for col_i in range(v) for col_j in range(col_i)],
    )
    return matrix, model

def model_type12(instance: Instance):
    matrix = instance.cp_vars['matrix']
    l,v,k = (instance.input_data[key] for key in ['lambda','v','k'])
    b = instance.constants['matrix_dim1']
    r = l+k
    #print(b, r)

    model = Model(
        [sum(row) == r for row in matrix],  # every row adds up to r
        [sum(col) == k for col in matrix.T],  # every column adds up to k

        # the scalar product of every pair of columns adds up to l
        #[sum([(row[col_i] * row[col_j]) for row in matrix]) == l
        # for col_i in range(v) for col_j in range(col_i)],
    )

    raise NotImplementedError("Ground truth model not known")
    return model


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 12
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    print("WARNING, SKIPPING THE LEARNING")
    #bounding_expressions = learn(instances)
    #for k, v in bounding_expressions.items():
    #    print(k, v)


    print("Ground-truth model (BIBD)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type12(inst)
    print(m)

    # wtf does it model
    #for i,inst in enumerate(instances):
    #    print(i, "constants:", inst.constants, "sum fst row", sum(inst.pos_data[0]['matrix'][0]))

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type12(inst)))
