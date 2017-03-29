import cplex
import sys
from cplex.callbacks import IncumbentCallback
from cplex.callbacks import LazyConstraintCallback

class CheckSolutuionCallback(IncumbentCallback):
    def __call__(self):
        
def solveFile(filename):
  c = cplex.Cplex(filename)
  
  try:
    c.solve()
  except CplexSolverError:
    print "Exception raised during solve"
    return
  
  status = c.solution.get_status()
  print "Solution status = ", status, ":",
  print c.solution.status[status]
  
  print "Objective value = ", c.solution.get_objective_value()

if __name__ == "__main__":
    solveFile("/var/www/html/euf2030/frontend/mips/job_32.lp")