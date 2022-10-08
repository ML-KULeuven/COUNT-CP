import glob

from cpmpy import *
import json

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


def model_type01(instance: Instance):
    m = Model()

    list_vars = instance.cp_vars["list"]

    # arcs diff color
    for pdict in instance.input_data["list"]:
        m += (list_vars[pdict['nodeA']] != list_vars[pdict['nodeB']])

    # m.maximize(max(list_vars))

    return m


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 1
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (Graph coloring)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type01(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type01(inst)))
