import numpy as np
from cpmpy.solvers.ortools import OrtSolutionCounter
from cpmpy.expressions.core import Expression
from instance import Instance

from cpmpy import *
from cpmpy.transformations.get_variables import get_variables


class OrtSolutionPrinter2(OrtSolutionCounter):
    def __init__(self, solver, display=None, solution_limit=None, verbose=False):
        super().__init__(verbose)
        self._solution_limit = solution_limit
        # we only need the cpmpy->solver varmap from the solver
        self._varmap = solver.varmap
        # identify which variables to populate with their values
        self._cpm_vars = []
        self._display = display
        if isinstance(display, (list,Expression)):
            self._cpm_vars = get_variables(display)
        elif callable(display):
            # might use any, so populate all (user) variables with their values
            self._cpm_vars = solver.user_vars

    def on_solution_callback(self):
        """Called on each new solution."""
        super().on_solution_callback()
        if len(self._cpm_vars):
            # populate values before printing
            for cpm_var in self._cpm_vars:
                # it might be an NDVarArray
                if hasattr(cpm_var, "flat"):
                    for cpm_subvar in cpm_var.flat:
                        cpm_subvar._value = self.Value(self._varmap[cpm_subvar])
                else:
                    cpm_var._value = self.Value(self._varmap[cpm_var])
            
            
            if isinstance(self._display, Expression):
                print(self._display.value())
            elif isinstance(self._display, list):
                # explicit list of expressions to display
                print([v.value() for v in self._display])
            else: # callable
                self._display()

            # check for count limit
            if self._solution_limit is not None and \
               self.solution_count() == self._solution_limit:
                self.StopSearch()



def solveAll(s, display=None, solution_limit=None):
    # XXX: check that no objective function??
    cb = OrtSolutionPrinter2(s, display=display, solution_limit=solution_limit)
    s.solve(enumerate_all_solutions=True, solution_callback=cb)
    return cb.solution_count()

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

def statistic(model1, model2, instance: Instance, size=100):
    sols = solutions(model1, instance, size)
    print(f"Number of solutions: {len(sols)}")
    if len(sols) == 0:
        return 0
    # print(len(sols), type(sols), type(sols[0]), type(sols[0][0]))
    vars = np.hstack([instance.cp_vars[k].flatten() for k in instance.cp_vars])
    s = SolverLookup.get("ortools", model2)
    s += Table(vars, sols)
    cnt = solveAll(s)
    # print(f"Number of solutions: {len(sols)}")
    return cnt * 100 / len(sols)
