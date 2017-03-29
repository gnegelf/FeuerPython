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
#class CheckSolutuionCallback(IncumbentCallback):
#    def __call__(self):
#        a=9
class StateconstraintCallback(LazyConstraintCallback):
    def __call__(self):
        global b_Uext
        global Amipext
        global tn
        global xn
        global names
        
        x_k=np.transpose(np.array([self.get_values()]))
        self.number_of_calls+=1
        #state=b_Uext-np.matmul(Amipext,x_k)
        state=Amipext*x_k-b_Lext
        #print(state)
        for i in range(1,tn+2):
            timestate=state[(i-1)*(xn+1)*(xn+1):i*(xn+1)*(xn+1)]
            minIdx=np.argmin(timestate)
            minIdx=(i-1)*(xn+1)*(xn+1)+minIdx
            #print(minIdx)
            if state[minIdx,0]<-0.0001:
                thevars=[]
                thecoefs=[]
                for j in range (0,Amipext.shape[1]):
                    if Amipext[minIdx,j]!=0:
                        thevars.append(names[j])
                        thecoefs.append(Amipext[minIdx,j])
                #print("Adding constraint with min Idx %i",minIdx)
                #print(thevars)
                self.add(constraint=cplex.SparsePair(thevars,thecoefs), sense = "G", rhs = float(b_Lext[minIdx]))
        print("callbacks: ",self.number_of_calls)
        
matlabData=scipy.io.loadmat('feuerData12_30.mat')

Amipred=scipy.sparse.lil_matrix(matlabData['A'])
b_Lred=matlabData['b_L']
b_Ured=matlabData['b_U']
c=matlabData['c']
xn=matlabData['xn'][0,0]
tn=matlabData['tn'][0,0]
Amipext=scipy.sparse.lil_matrix(matlabData['Aext'])
b_Uext=matlabData['b_Uext']
b_Lext=matlabData['b_Lext']
intVarN=matlabData['intVarN']
contVarN=matlabData['contVarN']
Amip=scipy.sparse.lil_matrix(matlabData['Afull'])
b_U=matlabData['b_Ufull']
b_L=matlabData['b_Lfull']
Arow=Amipred.shape[0]
Acol=Amipred.shape[1]
#define reduced ones
names=['']*Acol


lpFileName="feuer.lp"
lpFullFileName="feuerFull.lp"


write = 1
full = 0
write = 1
modelFull=cplex.Cplex()
model = cplex.Cplex()
for i in range(1,contVarN+1):
    names[i-1]="cont"+str(i)
    if write:
        model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["C"])
        modelFull.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["C"])
print("Hmm")
for i in range(contVarN+1,intVarN+contVarN+1):
    names[i-1]="bin"+str(i)
    if write:
        model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["B"])
        modelFull.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["B"])
print("hier")
Aredcoo=Amipred.tocoo()
Acoo=Amip.tocoo()
Aextcoo=Amipext.tocoo()

if write:
    if not full:
        rows=[]
        for i in range(0,Arow):
            rows.append([[],[]])
        
        for i,j,v in itertools.izip(Aredcoo.row, Aredcoo.col, Aredcoo.data):
            rows[i][0].append(names[j])
            rows[i][1].append(v)
        
        
        print("Huya")
        model.linear_constraints.add(lin_expr = rows, senses = ["L"]*Arow, rhs = np.transpose(b_Ured).tolist()[0])
        model.linear_constraints.add(lin_expr = rows, senses = ["G"]*Arow, rhs = np.transpose(b_Lred).tolist()[0])
        model.write(lpFileName)
        model2=cplex.Cplex(lpFileName)
    print("her")
    if full:
        rows2=[]
        for i in range(0,Amip.shape[0]):
            rows2.append([[],[]])
        
        for i,j,v in itertools.izip(Acoo.row, Acoo.col, Acoo.data):
            rows2[i][0].append(names[j])
            rows2[i][1].append(v)
        modelFull.linear_constraints.add(lin_expr = rows2, senses = ["L"]*Amip.shape[0], rhs = np.transpose(b_U).tolist()[0])
        modelFull.linear_constraints.add(lin_expr = rows2, senses = ["G"]*Amip.shape[0], rhs = np.transpose(b_L).tolist()[0])
        modelFull.write(lpFullFileName)
else:
    if full:
        modelFull=cplex.Cplex(lpFullFileName)
    else:
        model=cplex.Cplex(lpFileName)

  
if not full:
    lazy_cb = model.register_callback(StateconstraintCallback)
    lazy_cb2 = model2.register_callback(StateconstraintCallback)
    lazy_cb.number_of_calls = 0
    lazy_cb2.number_of_calls = 0

print("yay")
try:
    if not full:
        model.solve()
        #model2.solve()
    else:
        modelFull.solve()
except CplexError as exc:
    print(exc)
if not full:
    #print("LazyCbcalls: %i",lazy_cb.number_of_calls)
    print(model.solution.status[model.solution.get_status()])
    #print(modelFull.solution.status[modelFull.solution.get_status()]) 
    numrows = model.linear_constraints.get_num()
    numcols = model.variables.get_num()
    print("Solution status = ", model.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(model.solution.status[model.solution.get_status()])
    print("Solution value  = ", model.solution.get_objective_value())
    slack = model.solution.get_linear_slacks()
    x = model.solution.get_values()
    #x2= model2.solution.get_values()
    x_k=np.transpose(np.array([x]))
else:
    print(modelFull.solution.status[modelFull.solution.get_status()])
    #print(modelFull.solution.status[modelFull.solution.get_status()]) 
    numrows = modelFull.linear_constraints.get_num()
    numcols = modelFull.variables.get_num()
    print("Solution status = ", modelFull.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(modelFull.solution.status[modelFull.solution.get_status()])
    print("Solution value  = ", modelFull.solution.get_objective_value())
    slack = modelFull.solution.get_linear_slacks()
    x = modelFull.solution.get_values()
    x_k=np.transpose(np.array([x]))
#state=b_Uext-np.matmul(Amipext,x_k)
state2=Amipext*x_k-b_Lext
if full:
    scipy.io.savemat('stateFull.mat', dict(x_k=x))
    
else:
    scipy.io.savemat('state.mat', dict(x_k=x))
    #scipy.io.savemat('state2.mat', dict(x_k=x2))
    #for i in range(numrows):
#    print("Row %d:  Slack = %10f" % (i, slack[i]))
#for j in range(numcols):
#    print("Column %d:  Value = %10f" % (j, x[j]))

 


