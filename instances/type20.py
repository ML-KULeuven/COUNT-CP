import glob
import json

import numpy as np
from cpmpy import *

from learn import learn
from instance import Instance

"""
    N-queens, binary
"""


def model_type20(instance: Instance):
    board = instance.cp_vars['board']
    N = len(board)

    model = Model(
        # each row exactly one
        [sum(board[i,:]) == 1 for i in range(N)],
        # each col exactly one
        [sum(board[:,j]) == 1 for j in range(N)],
        # one for each "\"-diagonal
        [sum(board[i,j] for i in range(N) for j in range(N) if i-j==k) <= 1 for k in range(1-N, N-1)],
        # one for each "/"-diagonal
        [sum(board[i,j] for i in range(N) for j in range(N) if i+j==k) <= 1 for k in range(1,N+N-1)],
    )
    return model

if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 20
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    # skip learning
    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (n-queens, Boolean)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type20(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type20(inst)))
