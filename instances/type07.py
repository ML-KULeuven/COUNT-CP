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


def model_type07(instance: Instance):
    balls = instance.cp_vars['list']
    N = instance.constants['size']

    model = Model()

    # the 'not (x + y = z)' of the lemma is offset 1
    for x in range(1, N + 1):
        for y in range(1, N - x + 1):
            z = x + y
            model += ~((balls[x-1] == balls[y-1]) &
                       (balls[x-1] == balls[z-1]) &
                       (balls[y-1] == balls[z-1]))

    return model


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 7
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (schurs lemma)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type07(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type07(inst)))
