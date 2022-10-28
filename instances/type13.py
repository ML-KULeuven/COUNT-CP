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

# From MiniZinc challenge, links to http: // mathworld.wolfram.com / CostasArray.html
# alldiff and each row of distance matrix alldiff
# actually from a NumberJack paper

def model_type13(instance: Instance):
    costas = instance.cp_vars['list']
    N = instance.constants['size']

    model = Model()
    # costas array
    model += AllDifferent(costas)

    # triang diff matrix, each row all different
    for y in range(N - 2):
        model += AllDifferent([costas[i] - costas[i + y + 1] for i in range(N - y - 1)])

    return model


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 13
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (Costas Array)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type13(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type13(inst)))
