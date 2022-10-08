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


def model_type03(instance: Instance):
    warehouses = instance.cp_vars['warehouses']
    customers = instance.cp_vars['customers']

    model = Model(
        # warehouse=1 iff used
        [(sum(customers == i) > 0) == (warehouses[i] == 1) for i in range(instance.input_data['nrWarehouses'])]
    )

    wCosts = np.zeros(instance.input_data['nrWarehouses'], dtype=int)
    for d in instance.input_data['warehouseCost']:  # [{'cost': int, 'warehouse': int}]
        wCosts[d['warehouse']] = d['cost']
    wCost = sum(wCosts * warehouses)

    cCosts = np.zeros((instance.input_data['nrWarehouses'], instance.input_data['nrCustomers']), dtype=int)
    for d in instance.input_data['customerCost']:  # [{'cost': int, 'warehouse': int, 'customer': int}]
        cCosts[d['warehouse'], d['customer']] = d['cost']
    cCost = sum(cCosts[w, c] * (customers[c] == w) for w in range(instance.input_data['nrWarehouses']) for c in
                range(instance.input_data['nrCustomers']))

    # model.minimize(wCost + cCost)
    return model


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 3
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)

    print("Ground-truth model (warehouse location)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type03(inst)
    print(m)
    # tst
    print(m.solve())
    for k, v in inst.cp_vars.items():
        print(k, v.value())

    # sanity check ground truth
    for i, inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type03(inst)))
