# from a Future release... work in progress
# in the future, you can do s.solveAll() with a signature like below, and documentation ; )
from cpmpy.solvers.ortools import OrtSolutionCounter
from cpmpy.expressions.core import Expression

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
