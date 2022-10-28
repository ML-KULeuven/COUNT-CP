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


def model_type15(instance: Instance):
    queens = instance.cp_vars['list']
    N = len(queens)

    model = Model(
        AllDifferent(queens),
        #AllDifferent([queens[i] + i for i in range(N)]),
        #AllDifferent([queens[i] - i for i in range(N)])
    )

    att = []
    for i in range(N):
        i_att = []
        for j in range(N):
            i_att.append( (queens[i]+i == queens[j]+j) | (queens[i]-i == queens[j]-j) )
        att.append(sum(i_att))
    model += sum(att) >= 12

    raise NotImplementedError("Ground truth model not known")
    return model

if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 15
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (n-queens)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type15(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type15(inst)))
