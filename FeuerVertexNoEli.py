from __future__ import print_function
import cplex
#import sys
#from cplex.callbacks import IncumbentCallback
#from cplex.callbacks import LazyConstraintCallback
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.io
import itertools
from cplex.exceptions import CplexError
from cplex.callbacks import LazyConstraintCallback
from cplex.callbacks import BranchCallback

matlabData=scipy.io.loadmat('data/feuerDataNoElimination%d_%d_%d.mat' % (10,30,2))
Amip=scipy.sparse.lil_matrix(matlabData['A'])
b_L=matlabData['b_L']
b_U=matlabData['b_U']
c=matlabData['c']
xn=matlabData['xn'][0,0]
tn=matlabData['tn'][0,0]
intVarN=matlabData['intVarN']
contVarN=matlabData['contVarN']

Arow=Amip.shape[0]
Acol=Amip.shape[1]
#define reduced ones
names=['']*Acol

modelFull=cplex.Cplex()
model = cplex.Cplex()
for i in range(1,contVarN+1):
    names[i-1]="cont"+str(i-1)
    model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1000.0],types=["C"])
    modelFull.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1000.0],types=["C"])

for i in range(contVarN+1,intVarN+contVarN+1):
    names[i-1]="bin"+str(i-1)
    model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["B"])
    modelFull.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["B"])

Acoo=Amip.tocoo()

print(contVarN)
print(intVarN)
rows=[]
for i in range(0,Arow):
    rows.append([[],[]])

for i,j,v in itertools.izip(Acoo.row, Acoo.col, Acoo.data):
    rows[i][0].append(names[j])
    rows[i][1].append(v)

model.linear_constraints.add(lin_expr = rows, senses = ["L"]*Arow, rhs = np.transpose(b_U).tolist()[0])
model.linear_constraints.add(lin_expr = rows, senses = ["G"]*Arow, rhs = np.transpose(b_L).tolist()[0])



  
try:
    start=model.get_time()
    model.parameters.mip.tolerances.mipgap.set(0.1)
    #model.parameters.dettimelimit.set(50000.0)
    model.solve()
    end=model.get_time()
    duration=end-start
    #model2.solve()
except CplexError as exc:
    print(exc)


print(model.solution.status[model.solution.get_status()])
numrows = model.linear_constraints.get_num()
numcols = model.variables.get_num()
print("Solution status = ", model.solution.get_status(), ":", end=' ')

print(model.solution.status[model.solution.get_status()])
print("Solution value  = ", model.solution.get_objective_value())
slack = model.solution.get_linear_slacks()
x = model.solution.get_values()
x_k=np.transpose(np.array([x]))

scipy.io.savemat('/home/fabian/MIPDECO/Feuerprojekt/Results/FeuerprojektstateNoElixn%dtn%ds%d.mat' % (xn,tn,2),  dict([('x_k',x),('duration',duration)]))


 


