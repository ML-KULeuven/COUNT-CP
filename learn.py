from collections import defaultdict

import sympy
from sympy import symbols, lambdify, Symbol

from instance import Instance
import itertools as it
import numpy as np
import enum
from cpmpy import *


# def unary_operators():
#     def modulo(x):
#         return x % 2
#
#     def power(x):
#         return x * x
#
#     def identity(x):
#         return x
#
#     for f in [identity, abs]:
#         yield f


def generate_unary_exp(x):
    yield x
    # yield abs(x) for negative data


def generate_binary_expr(x, y):
    yield x + y
    yield x - y
    yield abs(x - y)


class Expression:
    @property
    def arity(self):
        raise NotImplementedError()

    def evaluate(self, *args):
        raise NotImplementedError()

    def bounds(self, *args):
        vals = self.evaluate(*args)
        return min(vals), max(vals)


class SympyExpression(Expression):
    def __init__(self, exp, exp_symbols):
        self.exp = exp
        self.exp_symbols = exp_symbols

    @property
    def f(self):
        return lambdify(self.exp_symbols, self.exp, "math")

    @property
    def arity(self):
        return len(self.exp_symbols)

    def evaluate(self, *args):
        return self.f(*args)

    def __str__(self):
        return str(self.exp)


class AggregateExpression(Expression):
    def __init__(self, f):
        self.f = f

    @property
    def arity(self):
        return None

    def evaluate(self, *args):
        return [self.f(assignment) for assignment in zip(*args)]

    def __str__(self):
        return self.f.__name__


def gen_expressions():
    x, y = symbols("x y")

    for u in generate_unary_exp(x):
        yield SympyExpression(u, [x])

    for b in generate_binary_expr(x, y):
        yield SympyExpression(b, [x, y])

    yield AggregateExpression(sum)


# model = []
# for expr in unary_grammar:
#     1. make expression
#     1b ?? check for (syntactic) equivalence with previous expressions?
#     for each inst:
#         2. compute bounds over all vars
#         3. search for grouped constraints over these bounds of this expression
#     4a: symbolize bounds across instances and only keep general ones
#     4. remove trivial bounds/constraints (both general and remaining single)
#     5. check that the found constraints are not implied by current model
#     6. add remaining to model
# same for expr in binary_grammar:

class PartitionType(enum.Enum):
    row = "row", 0
    column = "column", 1
    full = "full", None


class Partition:
    def generate_partition_indices(self, instance: Instance):
        raise NotImplementedError()


class GenericPartition(Partition):
    def __init__(self, tensor_name, partition_type):
        self.tensor_name = tensor_name
        self.partition_type = partition_type

    def generate_partition_indices(self, instance: Instance):
        index_pool = []
        if self.partition_type.value[1] is not None:
            for i in range(instance.tensors_dim[self.tensor_name][self.partition_type.value[1]]):
                indices = [
                    (self.tensor_name,) + indices
                    for indices in np.ndindex(*instance.tensors_dim[self.tensor_name])
                    if indices[self.partition_type.value[1]] == i
                ]
                index_pool.append(indices)
        else:
            indices = [
                (self.tensor_name,) + indices
                for indices in np.ndindex(*instance.tensors_dim[self.tensor_name])
            ]
            index_pool.append(indices)

        return index_pool

    def __repr__(self):
        return f"Partition({self.tensor_name}, {self.partition_type.value[0]})"


class InputPartition(Partition):
    def __init__(self, input_key):
        self.input_key = input_key

    def generate_partition_indices(self, instance: Instance):
        return instance.input_partitions[self.input_key]

    def __repr__(self):
        return f"InputPartition({self.input_key})"


def gen_partitions(type_shape: dict[str, int], input_keys: list[str]) -> list[Partition]:
    """
    Generates partitions for a given shape
    :param type_shape: Maps tensor names to the number of dimensions they have
    :param input_keys: The keys of (custom) input partitions provided to the learning algorithms
    :return:  A list of partitions
    """
    partitions = []
    for name, n_dims in type_shape.items():
        partitions.append(GenericPartition(name, PartitionType.full))
        if n_dims == 2:
            partitions.append(GenericPartition(name, PartitionType.row))
            partitions.append(GenericPartition(name, PartitionType.column))
        elif n_dims > 2:
            raise NotImplementedError()
    for key in input_keys:
        partitions.append(InputPartition(key))
    return partitions


class Sequence(enum.Enum):
    ALL_UNARY = "all_unary", 1
    EVEN_UNARY = "even_unary", 1
    ODD_UNARY = "odd_unary", 1
    FULL_UNARY = "full_unary", 1

    ALL_PAIRS = "all_pairs", 2
    # SEQUENCE_PAIRS = "sequence_pairs", 2


def gen_sequences(arity):
    if arity is None:
        return [Sequence.FULL_UNARY]

    return [
        s
        for s in Sequence
        if (s.value[1] == arity) and s not in (Sequence.EVEN_UNARY, Sequence.ODD_UNARY, Sequence.FULL_UNARY)
    ]


def gen_index_groups(sequence, partition_indices):
    # print(partition_indices)
    if sequence == Sequence.ALL_UNARY:
        return [(i,) for i in partition_indices]
    if sequence == Sequence.EVEN_UNARY:
        return [(partition_indices[i],) for i in range(0, len(partition_indices), 2)]
    if sequence == Sequence.ODD_UNARY:
        return [(partition_indices[i],) for i in range(1, len(partition_indices), 2)]
    if sequence == Sequence.FULL_UNARY:
        return [tuple(i for i in partition_indices)]

    if sequence == Sequence.ALL_PAIRS:
        return list(it.combinations(partition_indices, r=2))
    # if sequence == Sequence.SEQUENCE_PAIRS:
    #     return [(partition_indices[i], partition_indices[i + 1]) for i in range(len(partition_indices) - 1)]


def filter_partition_bounds(partition_bounds, threshold=1):
    # cv = lambda x: np.std(x, ddof=1) / np.mean(x)
    #
    # for sequence in all_sequences:
    #     lbs = [lb for lb, _ in partition_bounds[sequence]]
    #     ubs = [ub for _, ub in partition_bounds[sequence]]
    #
    #     if cv(lbs) > threshold:
    #         partition_bounds[sequence]

    return partition_bounds  # Filter using STD-DEV


def compute_shape(instances: list[Instance]):
    return {key: len(shape) for key, shape in instances[0].tensors_dim.items()}


def compute_input_keys(instances: list[Instance]):
    return [key for key in instances[0].input_partitions]


FeatureName = str
InstanceNumber = Value = int


def compute_candidate_features(instances: list[Instance]) -> dict[FeatureName, dict[InstanceNumber, Value]]:
    return {k: {instance.number: instance.constants[k] for instance in instances} for k in instances[0].constants}


def fit_feature_expressions_single(
        is_lb,
        bounds: dict[InstanceNumber, tuple],
        candidate_features: dict[FeatureName, dict[InstanceNumber, Value]],
        threshold=1
):
    if is_lb:
        bound = min([bounds[instance_number][0] for instance_number in bounds])
        if bound == float("-inf"):
            return None
        min_error = [bounds[instance_number][0] - bound for instance_number in bounds]
    else:
        bound = max([bounds[instance_number][1] for instance_number in bounds])
        if bound == float("inf"):
            return None
        min_error = [bound - bounds[instance_number][1] for instance_number in bounds]

    for f, values in candidate_features.items():
        if is_lb:
            bias = min(
                [bounds[instance_number][0] - values[instance_number]
                 for instance_number in bounds]
            )
            error = [bounds[instance_number][0] - values[instance_number] - bias for instance_number in bounds]

        else:
            bias = max(
                [bounds[instance_number][1] - values[instance_number]
                 for instance_number in bounds]
            )
            error = [values[instance_number] + bias - bounds[instance_number][1] for instance_number in bounds]

        if sum(error) < sum(min_error):
            min_error = error
            bound = bias + sympy.S(f)

    if sum(min_error) / len(min_error) > threshold:
        return None

    return bound


def fit_feature_expressions(
        bounds: dict[InstanceNumber, tuple],
        candidate_features: dict[FeatureName, dict[InstanceNumber, Value]],
        threshold=1
):
    return (
        fit_feature_expressions_single(True, bounds, candidate_features, threshold),
        fit_feature_expressions_single(False, bounds, candidate_features, threshold)
    )


def learn_for_instance(instance: Instance, expression, training_size=None):
    print("expression", expression)
    if not instance.has_solutions():
        return

    return learn_local_bounds(instance, expression, training_size)


def learn_propositional(instance):
    bounding_expressions = dict()

    for exp in gen_expressions():
        for key, val in learn_for_instance(instance, exp).items():
            # noinspection PyTypeChecker
            bounding_expressions[(exp,) + (key,)] = val

    return bounding_expressions


def find_bounds(indices, expression, instance: Instance):
    cp_vars = [np.array([instance.cp_vars[index[0]][index[1:]]]) for index in indices]
    objective = expression.evaluate(*cp_vars)[0]

    def solve(is_lb):
        m = Model()
        if is_lb:
            m.minimize(objective)
        else:
            m.maximize(objective)
        m.solve()
        return m.objective_value()

    return solve(True), solve(False)


def learn_local_bounds(instance, expression, training_size=None):
    local_bounds = dict()

    def compute_bounds(_indices):
        # print(_indices)
        # print(instance.training_data)
        args = [instance.training_data[ind[0]][(slice(0, training_size),) + ind[1:]] for ind in _indices]
        lb, ub = expression.bounds(*args)
        min_lb, max_ub = find_bounds(_indices, expression, instance)
        if lb == min_lb:
            lb = float("-inf")
        if ub == max_ub:
            ub = float("inf")
        local_bounds[_indices] = lb, ub

    if expression.arity is not None:
        for indices in instance.all_local_indices(expression.arity):
            compute_bounds(indices)
    else:
        type_shape = compute_shape([instance])
        input_keys = compute_input_keys([instance])
        all_partitions = gen_partitions(type_shape, input_keys)

        for partition in all_partitions:
            for partition_indices in partition.generate_partition_indices(instance):
                compute_bounds(tuple(partition_indices))

    return local_bounds


def learn_for_expression(instances: list[Instance], expression, symbolic, training_size=None):
    bounds_over_partitions_across_instances = dict()
    type_shape = compute_shape(instances)
    input_keys = compute_input_keys(instances)
    candidate_features = {}
    if symbolic:
        candidate_features = compute_candidate_features(instances)
    #
    all_partitions = gen_partitions(type_shape, input_keys)
    all_sequences = gen_sequences(expression.arity)

    print("expression", expression)

    for instance in instances:
        if not instance.has_solutions():
            continue

        local_bounds = learn_local_bounds(instance, expression, training_size)

        bounds_over_partitions = dict()

        # columns of a matrix, rows of a matrix, all values of a list
        for partitions in all_partitions:
            partition_bounds = defaultdict(list)

            # all indices in a specific column
            for partition_indices in partitions.generate_partition_indices(instance):
                for sequence in all_sequences:  # all-pairs, sequential values
                    partition_sequence_bounds = [
                        local_bounds[index_group]
                        for index_group in gen_index_groups(sequence, partition_indices)
                        # one pair of a specific column
                    ]
                    partition_bounds[sequence].append((
                        min([lb for lb, _ in partition_sequence_bounds]),
                        max([ub for _, ub in partition_sequence_bounds])
                    ))

            partition_bounds = filter_partition_bounds(partition_bounds)

            bounds_over_partitions[partitions] = {
                seq:
                    (min([lb for lb, _ in partition_bounds[seq]]), max([ub for _, ub in partition_bounds[seq]]))
                for seq in all_sequences
            }

            # print(expression, partitions, bounds_over_partitions[partitions])

        bounds_over_partitions_across_instances[instance.number] = bounds_over_partitions

    # Symbolic

    bounding_expressions = dict()

    # print([i for i, ins in enumerate(instances) if ins.has_solutions()])

    for i, partitions in enumerate(all_partitions):
        for sequence in all_sequences:
            bounds = {instance.number:
                          bounds_over_partitions_across_instances[instance.number][partitions][sequence]
                      for instance in instances if instance.has_solutions()}
            # print(bounds)
            symbolic_bounds = fit_feature_expressions(bounds, candidate_features)
            ##### for filtering redundant partitions and sequences ####
            if i > 0 and (all_partitions[0], sequence) in bounding_expressions and \
                    bounding_expressions[
                        (all_partitions[0], sequence)] == symbolic_bounds:
                continue
            # if sequence == Sequence.SEQUENCE_PAIRS and (partitions, Sequence.ALL_PAIRS) in bounding_expressions and \
            #         bounding_expressions[
            #             (partitions, Sequence.ALL_PAIRS)] == symbolic_bounds:
            #     continue
            # if sequence == Sequence.SEQUENCE_PAIRS and (all_partitions[0], Sequence.ALL_PAIRS) in bounding_expressions and \
            #         bounding_expressions[
            #             (all_partitions[0], Sequence.ALL_PAIRS)] == symbolic_bounds:
            #     continue
            ###################
            bounding_expressions[(partitions, sequence)] = symbolic_bounds
            print("\t", partitions, sequence, bounding_expressions[(partitions, sequence)])

    # print(bounding_expressions)
    return bounding_expressions


def learn(instances, training_size=None, symbolic=True):
    bounding_expressions = dict()

    for exp in gen_expressions():
        for key, val in learn_for_expression(instances, exp, symbolic, training_size).items():
            # noinspection PyTypeChecker
            bounding_expressions[(exp,) + key] = val

    return bounding_expressions


def ground_bound(instance, bound):
    # print(instance.constants, bound)
    try:
        for k, v in instance.constants.items():
            bound = bound.subs(Symbol(k), v)
        return int(bound)
    except AttributeError:
        return bound


def encode_constraint(m, instance, indices, expression, bounds):
    lb, ub = bounds
    # print(indices)
    cp_vars = [np.array([instance.cp_vars[index[0]][index[1:]]]) for index in indices]
    # print(*cp_vars)
    # m += [sum([*cp_vars]) >= 0]
    cpm_e = expression.evaluate(*cp_vars)[0]
    if lb is not None:
        m += [cpm_e >= ground_bound(instance, lb)]
    if ub is not None:
        m += [cpm_e <= ground_bound(instance, ub)]
    # print(m)


def create_model(general_bounds, instance: Instance, propositional):
    m = Model()
    total_constraints = 0

    if propositional:
        for (expr, indices), bounds in general_bounds.items():
            encode_constraint(m, instance, indices, expr, bounds)
            total_constraints += 2
    else:
        for (expr, partitions, sequences), bounds in general_bounds.items():
            for partition_indices in partitions.generate_partition_indices(instance):
                for indices in gen_index_groups(sequences, partition_indices):
                    encode_constraint(m, instance, indices, expr, bounds)
                    total_constraints += 2

    if instance.input_assignments:
        for k, v in instance.input_assignments.items():
            m += [instance.cp_vars[k[0]][k[1:]] == v]

    return m, total_constraints
