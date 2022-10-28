from cpmpy import *
# from cpmpy.solvers import CPM_ortools
from cpmpy.transformations.get_variables import *
from instance import Instance
import numpy as np
# from learner import solutions

def nurse_rostering_instance(nurses=7, days=7, minNurses=5, maxNurses=7):
    schedule_instance = Instance(nurses, {"inputData":{}, "formatTemplate":{}, "solutions": None, "tests":{}}, 22)
    schedule_instance.input_data = {'minNurses': minNurses, 'maxNurses': maxNurses}
    schedule_instance.constants = {'minNurses': minNurses, 'maxNurses': maxNurses}
    schedule_instance.tensors_dim = {'array': (nurses, days)}
    schedule_instance.var_lbs = {'array': np.zeros([nurses, days]).astype(int)}
    schedule_instance.var_ubs = {'array': np.ones([nurses, days]).astype(int)}
    schedule_instance.var_bounds = {
        k: list(zip(schedule_instance.var_lbs[k].flatten(), schedule_instance.var_ubs[k].flatten()))
        for k in schedule_instance.tensors_dim
    }
    # print(schedule_instance.var_lbs, schedule_instance.var_ubs, schedule_instance.tensors_dim)
    m = nurse_rostering_model(schedule_instance)
    schedule_instance.pos_data = []
    for solution in solutions(m, schedule_instance, 100):
        solution = np.reshape(solution, (nurses, days))
        schedule_instance.pos_data.append({'array': solution})
    if not schedule_instance.pos_data:
        print(nurses, days, minNurses, maxNurses)
    schedule_instance.training_data = {
        k: np.array([d[k] for d in schedule_instance.pos_data])
        for k in schedule_instance.tensors_dim
    }
    return schedule_instance

def nurse_rostering_model(instance:Instance):
    minNurses = instance.input_data['minNurses']
    maxNurses = instance.input_data['maxNurses']
    schedule = instance.cp_vars["array"]
    m = Model()
    m += [sum(schedule[i, :]) >= 0 for i in range(len(schedule))]
    m += [sum(schedule[i, :]) <= 5 for i in range(len(schedule))]
    m += [sum(schedule[:, i]) >= minNurses for i in range(len(schedule[0]))]
    m += [sum(schedule[:, i]) <= maxNurses for i in range(len(schedule[0]))]
    return m


if __name__ == "__main__":
    train_instance = nurse_rostering_instance(2, 5)
    m = nurse_rostering_model(train_instance)
    cp_vars = get_variables_model(m)
    s = SolverLookup.get("ortools", m)
    while s.solve():
        print(cp_vars.value())
        s += ~all([var == var.value() for var in cp_vars])


def solutions(model: Model, instance: Instance, size):
    rng = np.random.RandomState(111)
    s = SolverLookup.get("ortools", model)
    # model = Model([c for c in model.constraints])
    # model = CPM_ortools(model)
    vars = np.hstack([instance.cp_vars[k].flatten() for k in instance.cp_vars])
    s += sum(vars) >= 0
    vars_lb = np.hstack([instance.var_lbs[k].flatten() for k in instance.var_lbs])
    vars_ub = np.hstack([instance.var_ubs[k].flatten() for k in instance.var_ubs])

    sols = []
    sol_count = 0
    while s.solve() and sol_count < size:
        sols.append([var.value() for var in vars])
        s += ~all([var == var.value() for var in vars])
        initial_point = []
        for i, v in enumerate(vars):
            initial_point.append(rng.randint(vars_lb[i], vars_ub[i]))
        s.solution_hint(vars, initial_point)
        sol_count += 1
    return sols



