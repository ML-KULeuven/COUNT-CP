import itertools
import logging
import random
from functools import reduce

#import cpmpy
from cpmpy import *

import numpy as np

logger = logging.getLogger(__name__)


def nested_map(f, tensor):
    if isinstance(tensor, (list, tuple)):
        return [nested_map(f, st) for st in tensor]
    else:
        return f(tensor)


def load_input_partitions(type_number, input_data, constants):

    if type_number == 1:

        return {
            "edges": [
                [("list", d["nodeA"]), ("list", d["nodeB"])]
                for d in input_data["list"]
            ]
        }

    elif type_number == 5 or type_number == 6:

        partitions = {"blocks": []}
        size = constants["size"]

        for i in range(size):
            for j in range(size):

                partition = []

                for k in range(i * size, (i + 1) * size):
                    for l in range(j * size, (j + 1) * size):
                        partition.append(("array", k, l))

                partitions["blocks"].append(partition)

        return partitions

    elif type_number == 20:

        partitions = {"diagonals": []}
        size = constants["size"]
        # grid = np.indices((size, size))
        matrix = np.empty([size, size], dtype=object)

        for i in range(size):
            for j in range(size):
                matrix[i, j] = (i, j)

        diags = [matrix[::-1, :].diagonal(i) for i in range(1-size, size)]
        diags.extend(matrix.diagonal(i) for i in range(size-1, -size, -1))

        for partition in diags:

            partition = sorted([("board", i, j) for i,j in partition])
            # partition = sorted(partition)

            if len(partition)==1:
                continue

            partitions["diagonals"].append(partition)

        return partitions

    return {}


def load_input_assignments(type_number, input_data):
    if type_number == 5:
        return {("array", d["row"], d["column"]): d["value"] for d in input_data["preassigned"]}


class Instance:

    def __init__(self, number, json_data, problem_type):

        tensors_lb = {}
        tensors_ub = {}

        self.number = number
        self._cp_vars = None

        self.problem_type = problem_type
        self.jsonSeq = None
        self.input_data = json_data.get("inputData", {})
        self.constants = {k: v for k, v in self.input_data.items() if isinstance(v, (int, float))}

        if "size" in json_data:
            self.constants["size"] = json_data["size"]

        self.input_partitions = load_input_partitions(problem_type, self.input_data, self.constants)
        self.input_assignments = load_input_assignments(problem_type, self.input_data)
        self.formatTemplate = json_data["formatTemplate"]

        for k, v in json_data["formatTemplate"].items():
            if k != "objective":
                tensors_lb[k] = np.array(nested_map(lambda d: d["low"], v))
                tensors_ub[k] = np.array(nested_map(lambda d: d["high"], v))

        self.tensors_dim = {k: v.shape for k, v in tensors_ub.items()}
        for k, shape in self.tensors_dim.items():
            for i, v in enumerate(shape):
                self.constants[f"{k}_dim{i}"] = v

        self.var_lbs = tensors_lb
        self.var_ubs = tensors_ub

        self.var_bounds = {
            k: list(zip(tensors_lb[k].flatten(), tensors_ub[k].flatten()))
            for k in self.tensors_dim
        }

        self.objective = json_data["formatTemplate"].get("objective", None)

        def import_objectives(_l):
            return np.array([d["objective"] for d in _l])

        if self.objective:
            self.pos_data_obj = import_objectives(json_data["solutions"])
            self.neg_data_obj = import_objectives(json_data["nonSolutions"])
            self.test_obj = import_objectives(json_data["tests"])
        else:
            self.pos_data_obj = self.neg_data_obj = self.test_obj = None

        def import_data(_l):
            return [
                {_k: np.array(_e[_k]) for _k in self.tensors_dim}
                for _e in _l
            ]

        def import_data__flattened(_l):
            return {
                _k: np.array([np.array(d[_k]).flatten() for d in _l])
                for _k in self.tensors_dim
            }

        self.pos_data = self.neg_data = self.test_data = self.training_data = None

        if json_data["solutions"]:
            self.pos_data = import_data(json_data["solutions"])
            self.neg_data = import_data(json_data["nonSolutions"])
            self.training_data = {
                k: np.array([d[k] for d in self.pos_data])
                for k in self.tensors_dim
            }

        self.test_data = import_data(json_data["tests"])

        if problem_type == 3:

            inputData = json_data["inputData"]
            customerCost = np.zeros(
                [inputData["nrWarehouses"], inputData["nrCustomers"]]
            )

            for v in inputData["customerCost"]:
                customerCost[v["warehouse"], v["customer"]] = v["cost"]

            warehouseCost = np.zeros(inputData["nrWarehouses"])

            for v in inputData["warehouseCost"]:
                warehouseCost[v["warehouse"]] = v["cost"]
            # self.inputData = [warehouseCost, customerCost]

        if problem_type == 1:

            inputData = self.input_data["list"]
            lst = []

            for d in inputData:
                lst.append(tuple(sorted(d.values())))
            self.jsonSeq = lst

    @property
    def cp_vars(self):

        if self._cp_vars is None:

            self._cp_vars = dict()
            for k in self.tensors_dim:

                indices = np.array(["-".join(map(str, i)) for i in np.ndindex(*self.tensors_dim[k])])
                index_iterable = np.reshape(np.array(indices), self.tensors_dim[k])

                self._cp_vars[k] = cpm_array(
                    np.vectorize(lambda _i, _lb, _ub: intvar(
                        _lb, _ub, name=f"{k}-{_i}"
                    ))(index_iterable, self.var_lbs[k], self.var_ubs[k])
                )

        return self._cp_vars

    def has_solutions(self):
        return self.pos_data is not None

    def flatten_data(self, data):
        return [np.hstack([list(d[k].flatten()) for k in self.tensors_dim]) for d in data]
        # all_data = None
        # for k in self.tensors_dim:
        #     if all_data is None:
        #         all_data = data[k]
        #     else:
        #         all_data = np.hstack([all_data, data[k]])
        # return all_data

    def unflatten_data(self, data):

        d = dict()
        offset = 0

        for k, dims in self.tensors_dim.items():
            length = reduce(lambda a, b: a * b, dims)
            d[k] = data[offset:offset + length].reshape(dims)
            offset += length
        return d

    def all_local_indices(self, arity):

        for name in self.tensors_dim:

            index_pool = [
                (name,) + indices
                for indices in np.ndindex(*self.tensors_dim[name])
            ]

            yield from itertools.combinations(index_pool, arity)

    def example_count(self, positive):

        data = self.pos_data if positive else self.neg_data

        for k in self.tensors_dim:
            return data[k].shape[0]

        raise RuntimeError("Tensor dimensions are empty")

    def objective_function(self, data):

        if self.problem_type == 3:

            data = self.unflatten_data(data)
            sum = 0
            tmp = np.zeros([len(data["warehouses"]), len(data["customers"])])

            for i, c in enumerate(data["customers"]):
                tmp[c][i] = 1

            # print(self.input_data, data["customers"], data["warehouses"])

            warehouseCost = [d['cost'] for d in self.input_data['warehouseCost']]
            customerCost = np.reshape([d['cost'] for d in self.input_data['customerCost']], tmp.shape)
            sum += np.sum(np.multiply(warehouseCost, data["warehouses"]))
            sum += np.sum(np.multiply(customerCost, tmp))

            return sum

        return max(data)

    def check(self, model):

        model_vars = np.hstack([self.cp_vars[k].flatten() for k in self.cp_vars])

        percentage_pos, cnt, co, total = check_solutions_fast(
            model,
            cpm_array(model_vars),
            self.flatten_data(self.pos_data),
            self.objective_function,
            self.pos_data_obj,
        )

        percentage_neg, cnt, co, total = check_solutions_fast(
            model,
            cpm_array(model_vars),
            self.flatten_data(self.neg_data),
            self.objective_function,
            self.neg_data_obj,
        )

        percentage_neg = 100 - percentage_neg

        return percentage_pos, percentage_neg, cnt, co, total


def check_solutions_fast(m: Model, m_vars, sols, objective_exp, objective_values):
    if sols is None:
        print("No solutions to check")
        return 100
    correct_objective = sols

    # remove duplicates, if any (happens for type06)
    for i in reversed(range(len(sols))):  # backward, for del
        for j in range(i):  # forward up to and without i
            if np.array_equal(sols[i], sols[j]):
                # sols are equal, check to drop 'i' (at back)
                if objective_values is None:
                    del sols[i]
                    break
                elif objective_values[i] == objective_values[j]:
                    del sols[i]
                    del objective_values[i]
                    break

    # filter out based on objective values, if present
    if objective_values is not None:
        correct_objective = []
        for i, sol in enumerate(sols):
            if objective_exp(sol) == objective_values[i]:
                correct_objective.append(sol)

    # print(len(sols), len(correct_objective))
    s = SolverLookup.get("ortools", m)
    s += Table(
        m_vars,
        correct_objective
    )
    cnt = solveAll(s)
    # print(cnt, len(correct_objective))
    logger.info(f"{cnt} satisfied out of {len(sols)}")
    return cnt * 100.0 / len(sols), cnt, len(correct_objective), len(sols)
