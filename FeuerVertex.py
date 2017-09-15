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
#class CheckSolutuionCallback(IncumbentCallback):
#    def __call__(self):
#        a=9
class DummyCallback(BranchCallback):
    def __call__(self):
        return 0

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
            storeIdx=minIdx
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
                #print("Adding cut: %d" % (storeIdx))
                self.add(constraint=cplex.SparsePair(thevars,thecoefs), sense = "G", rhs = float(b_Lext[minIdx]))
        print("callbacks: ",self.number_of_calls)

for timeVar in range(30,31,10):
    for countVar in range(35,36,5):
        matlabData=scipy.io.loadmat('data/feuerData%d_%d_%d.mat' % (countVar,timeVar,2))
        Amipred=scipy.sparse.lil_matrix(matlabData['A2'])
        b_Lred=matlabData['b_L2']
        b_Ured=matlabData['b_U2']
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
        full = 0
        order = 0
        
        model = cplex.Cplex()
        for i in range(1,contVarN+1):
            names[i-1]="cont"+str(i-1)
            model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["C"])
        print("Finished adding continuous variables")
        for i in range(contVarN+1,intVarN+contVarN+1):
            names[i-1]="bin"+str(i-1)
            model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["B"])
        print("Finished adding integer variables")
        Aredcoo=Amipred.tocoo()
        Acoo=Amip.tocoo()
        Aextcoo=Amipext.tocoo()
        rows=[]
        if not full:
            for i in range(0,Arow):
                rows.append([[],[]])
            for i,j,v in itertools.izip(Aredcoo.row, Aredcoo.col, Aredcoo.data):
                rows[i][0].append(names[j])
                rows[i][1].append(v)  
            model.linear_constraints.add(lin_expr = rows, senses = ["L"]*Arow, rhs = np.transpose(b_Ured).tolist()[0])
            model.linear_constraints.add(lin_expr = rows, senses = ["G"]*Arow, rhs = np.transpose(b_Lred).tolist()[0])
        else:
            for i in range(0,Amip.shape[0]):
                rows.append([[],[]])
            for i,j,v in itertools.izip(Acoo.row, Acoo.col, Acoo.data):
                rows[i][0].append(names[j])
                rows[i][1].append(v)
            model.linear_constraints.add(lin_expr = rows, senses = ["L"]*Amip.shape[0], rhs = np.transpose(b_U).tolist()[0])
            model.linear_constraints.add(lin_expr = rows, senses = ["G"]*Amip.shape[0], rhs = np.transpose(b_L).tolist()[0])
        if not full:
            lazy_cb = model.register_callback(StateconstraintCallback)
            lazy_cb.number_of_calls = 0
        else:
            dummy_cb = model.register_callback(DummyCallback)
            dummy_cb.number_of_calls = 0
        print("Finished adding constraints")
        print(model.get_version())
        try:
            start=model.get_time()
            #model.parameters.mip.tolerances.mipgap.set(0.00001)
            model.parameters.timelimit.set(5000.0)
            model.parameters.mip.display.set(2)
            if not full:
                print("Starting to solve model with callbacks")
            else:
                print("Starting to solve model without callbacks")
            #model.set_warning_stream('feuerWarning%d_%d' % (countVar,timeVar))
            model.set_log_stream('feuerLogfile%d_%d.log' % (countVar,timeVar))
            model.set_results_stream('feuerResult%d_%d' % (countVar,timeVar))
            model.solve()
            status=model.solution.get_status()
            print(model.solution.MIP.get_mip_relative_gap())
            #model.set_log_stream('feuerLogfile%d_%d.log' % (countVar,timeVar))
            end=model.get_time()
            duration=end-start
            #model2.solve()
        except CplexError as exc:
            print(exc)
            #print("LazyCbcalls: %i",lazy_cb.number_of_calls)
        if (status == 101 or status== 102):
            print(model.solution.status[model.solution.get_status()])
            #print(modelFull.solution.status[modelFull.solution.get_status()]) 
            numrows = model.linear_constraints.get_num()
            numcols = model.variables.get_num()
            print("Solution status = ", model.solution.get_status(), ":", end=' ')
            # the following line prints the corresponding string
            print(model.solution.status[model.solution.get_status()])
            #print("Solution value  = ", model.solution.get_objective_value())
            
            slack = model.solution.get_linear_slacks()
            x = model.solution.get_values()
            #x2= model2.solution.get_values()
            x_k=np.transpose(np.array([x]))
            #state=b_Uext-np.matmul(Amipext,x_k)
            state2=Amipext*x_k-b_Lext
            print("The duration = ", duration)
            print("saving result and duration")
            print(model.solution.get_status_string())
            if full:
                scipy.io.savemat('/home/fabian/MIPDECO/Feuerprojekt/Results/stateFullxn%dtn%ds%d.mat' % (xn,tn,2), dict([('x_k',x),('duration',duration),('objective',model.solution.get_objective_value())]))    
            else:
                scipy.io.savemat('/home/fabian/MIPDECO/Feuerprojekt/Results/statexn%dtn%ds%d.mat' % (xn,tn,2),  dict([('x_k',x),('duration',duration),('objective',model.solution.get_objective_value())]))
        if (status == 107):
            if full:
                scipy.io.savemat('/home/fabian/MIPDECO/Feuerprojekt/Results/stateFullxn%dtn%ds%d.mat' % (xn,tn,2), dict([('x_k',x),('duration',duration),('objective',model.solution.get_objective_value()),('gap',model.solution.MIP.get_mip_relative_gap())]))      
            else:
                scipy.io.savemat('/home/fabian/MIPDECO/Feuerprojekt/Results/statexn%dtn%ds%d.mat' % (xn,tn,2),  dict([('x_k',x),('duration',duration),('objective',model.solution.get_objective_value(),('gap',model.solution.MIP.get_mip_relative_gap()))]))





