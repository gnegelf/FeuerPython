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
        state=b_Uext-np.matmul(Amipext,x_k)
        #print(state)
        for i in range(1,tn+2):
            timestate=state[(i-1)*(xn+1)*(xn+1):i*(xn+1)*(xn+1)]
            minIdx=np.argmin(timestate)
            minIdx=(i-1)*(xn+1)*(xn+1)+minIdx
            #print(minIdx)
            if state[minIdx]<-0.0001:
                thevars=[]
                thecoefs=[]
                for j in range (0,Amipext.shape[1]):
                    if Amipext[minIdx][j]!=0:
                        thevars.append(names[j])
                        thecoefs.append(Amipext[minIdx][j])
                #print("Adding constraint with min Idx %i",minIdx)
                #print(thevars)
                self.add(constraint=cplex.SparsePair(thevars,thecoefs), sense = "L", rhs = float(b_Uext[minIdx]))
                #(lin_expr = [cplex.SparsePair(thevars,thecoefs)], senses = ["G"], rhs = [float(b_Lred[i])])
        print("callbacks: ",self.number_of_calls)
            
        """
        b_U=0
        x_k=0#read Data
        state=np.matmul(A,x_k)
        state=np.multiply(-1.0,state)#calculate State
        for i in range(1,tn+1+1):
            ASorted, AIdx = sort(state[(i-1)*(xn+1)^2+1:i*(xn+1)*(xn+1)])#find index of minimal state for each time interval
            if state((i-1)*(xn+1)^2+addI)<-0.0001:#if that minimum is less than 0 add correct constraint
                iList=[iList (i-1)*(xn+1)^2+addI]

                Ared=[Ared, A[1+ishift+(i-1)*(xn+1)^2+addI,:]
                b_Lred=[b_Lred, b_L[1+ishift+(i-1)*(xn+1)*(xn+1)+addI]
                b_Ured=[b_Ured, b_U(1+ishift+(i-1)*(xn+1)*(xn+1)+addI]
                """

        
def constrFD(xn,tn,dx,dt,a1,a2,D,hL,hR):
    xn1=xn+3
    #A = scipy.sparse.lil_matrix((xn1*xn1*(tn+1),xn1*xn1*(tn+1)))
    difM=scipy.sparse.lil_matrix((xn1*xn1,2*xn1*xn1))
    R=lambda x,y: D(x,y)*dt/dx/dx

    #A[1-1:xn1*xn1,1-1:xn1*xn1]=scipy.sparse.eye(xn1*xn1)
    for i in range(2,xn+3):
        for j in range(2,xn+3):
            nDiag=-0.5*R((i-2)*dx,(j-2)*dx)
            rx=dt/dx*a1((i-2)*dx,(j-2)*dx)
            ry=dt/dx*a2((i-2)*dx,(j-2)*dx)
            difM[(i-1)*xn1+j-1,(i-1)*xn1+j+xn1*xn1  -1]=1.0+2.0*R((i-2)*dx,(j-2)*dx)
            difM[(i-1)*xn1+j-1,(i-2)*xn1+j+xn1*xn1  -1]=nDiag 
            difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1+xn1*xn1-1]=nDiag
            difM[(i-1)*xn1+j-1,(i-1)*xn1+j+1+xn1*xn1-1]=nDiag
            difM[(i-1)*xn1+j-1, i   *xn1+j+xn1*xn1  -1]=nDiag
            difM[(i-1)*xn1+j-1,(i-1)*xn1+j  -1]=-1.0+2.0*R((i-2)*dx,(j-2)*dx)
            if rx>0:
                difM[(i-1)*xn1+j-1,(i-2)*xn1+j  -1]=nDiag-rx
                difM[(i-1)*xn1+j-1, i   *xn1+j  -1]=nDiag
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j  -1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1]+rx
            else:
                difM[(i-1)*xn1+j-1,(i-2)*xn1+j -1]=nDiag
                difM[(i-1)*xn1+j-1, i   *xn1+j-1]=nDiag+rx
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j  -1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1]-rx
            if ry>0:
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1-1]=nDiag
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j+1-1]=nDiag-ry
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j  -1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1]+ry
            else:
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1-1]=nDiag+ry
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j+1-1]=nDiag
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j  -1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1]-ry
    for j in range(2,xn+3):
        difM[j-1,j+xn1*xn1  -1]=1.0/(2.0*dx)
        difM[j-1,2*xn1+j+xn1*xn1  -1]=-1.0/(2.0*dx)
        difM[j-1,xn1+j+xn1*xn1  -1]=-hL
        difM[(xn1-1)*xn1+j-1,(xn1-1)*xn1+j+xn1*xn1  -1]=+1.0/(2.0*dx)
        difM[(xn1-1)*xn1+j-1,(xn1-3)*xn1+j+xn1*xn1  -1]=-1.0/(2.0*dx)
        difM[(xn1-1)*xn1+j-1,(xn1-2)*xn1+j+xn1*xn1  -1]=hR
    
    for i in range(2,xn+3):
        difM[(i-1)*xn1+1-1,(i-1)*xn1+1+xn1*xn1  -1]=1.0/(2.0*dx)
        difM[(i-1)*xn1+xn1-1,(i-1)*xn1+xn1+xn1*xn1  -1]=1.0/(2.0*dx)
        difM[(i-1)*xn1+1-1,(i-1)*xn1+2+xn1*xn1  -1]=1.0/-(2.0*dx)
        difM[(i-1)*xn1+xn1-1,(i-1)*xn1+xn1-1+xn1*xn1  -1]=1.0/-(2.0*dx)
    difM[1              -1,1+xn1*xn1 -1]=1.0
    difM[(xn1-1)*xn1+1  -1,(xn1-1)*xn1+1+xn1*xn1  -1]=1.0
    difM[xn1            -1,xn1+xn1*xn1            -1]=1.0
    difM[(xn1-1)*xn1+xn1-1,(xn1-1)*xn1+xn1+xn1*xn1-1]=1.0
    Aaddbase=scipy.sparse.kron(scipy.sparse.diags([1]+[0]*(tn),0),scipy.sparse.eye(xn1*xn1))
    Aadd=scipy.sparse.kron(scipy.sparse.diags([1]*(tn),-1),difM[:,1-1:xn1*xn1])
    Aadd2=scipy.sparse.kron(scipy.sparse.diags([0]+[1]*(tn),0),difM[:,xn1*xn1+1-1:xn1*xn1*2])

    A=Aaddbase+Aadd+Aadd2
    #print B.toarray()
    return A
    #A[xn1*xn1+1-1:xn1*xn1*(tn+1),1-1:xn1*xn1*(tn)]=Aadd
    #A[xn1*xn1+1-1:xn1*xn1*(tn+1),xn1*xn1+1-1:xn1*xn1*(tn+1)]=A[xn1*xn1+1-1:xn1*xn1*(tn+1),xn1*xn1+1-1:xn1*xn1*(tn+1)]+Aadd2
    return A

def constrAbc(i3, tn,xn,contN,uInhom,maxCont,z1):
    i1=contN*contN
    i2=(tn+1)*i1
    A=np.zeros((1+i2+i3,i1+i2))
    b_L=np.multiply(-1000000,np.ones((1+i2+i3,1)))
    b_U=np.zeros((1+i2+i3,1))
    c=np.zeros((i1+i2,1))
    b_U[1-1]=maxCont
    A[1-1,1-1:i1]=np.ones((1,i1))
    for i in range(1,tn+1+1):
        for j in range(1,i1+1):
            A[1+(i-1)*i1+j-1,i1+(i-1)*contN*contN+j-1]=1
            A[1+(i-1)*i1+j-1,j-1]=-1
    shiftR=1+i2
    for t in range(1,tn+1+1):
        for x in range(1,xn+1+1):
            for y in range(1,xn+1+1):
                b_U[shiftR+(t-1)*(xn+1)*(xn+1)+(x-1)*(xn+1)+y-1]=uInhom[x-1,y-1,t-1]
                for t2 in range(1,tn+1+1):
                    for cont in range(1,i1+1):
                        if x==1:
                            c[i1+(t2-1)*contN*contN+cont-1]=c[i1+(t2-1)*contN*contN+cont-1]+z1[(t-1)*(xn+3)*(xn+3)+(x)*(xn+3)+y+1-1,cont-1,t2-1];
                        A[shiftR+(t-1)*(xn+1)*(xn+1)+(x-1)*(xn+1)+y-1,i1+(t2-1)*contN*contN+cont-1]=A[shiftR+(t-1)*(xn+1)*(xn+1)+(x-1)*(xn+1)+y-1,i1+(t2-1)*contN*contN+cont-1]-z1[(t-1)*(xn+3)*(xn+3)+(x)*(xn+3)+y+1-1,cont-1,t2-1];
    return A,b_L,b_U,c

def calcInit(xn,dx,a1,a2,D,hL,hR,fR):
    xn1=xn+3
    difM=scipy.sparse.lil_matrix((xn1*xn1,xn1*xn1))
    R= lambda x,y: D(x,y)/dx/dx
    for i in range(2,xn+3):
        for j in range(2,xn+3):
            nDiag=-R((i-2)*dx,(j-2)*dx)
            rx=1.0/dx*a1((i-2)*dx,(j-2)*dx)
            ry=1.0/dx*a2((i-2)*dx,(j-2)*dx)
            difM[(i-1)*xn1+j-1,(i-1)*xn1+j  -1]=4*R((i-2)*dx,(j-2)*dx)
            difM[(i-1)*xn1+j-1,(i-2)*xn1+j  -1]=nDiag
            difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1-1]=nDiag
            difM[(i-1)*xn1+j-1,(i-1)*xn1+j+1-1]=nDiag
            difM[(i-1)*xn1+j-1, i   *xn1+j -1 ]=nDiag
            if rx>0:
                difM[(i-1)*xn1+j-1,(i-2)*xn1+j -1]=difM[(i-1)*xn1+j-1,(i-2)*xn1+j -1]-rx
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j -1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j -1]+rx
            else:
                difM[(i-1)*xn1+j-1, i   *xn1+j-1]=difM[(i-1)*xn1+j-1, i   *xn1+j -1]+rx
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j -1]-rx
            if ry>0:
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j+1-1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j+1-1]-ry
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1  ]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1]+ry
            else:
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1-1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1-1]+ry
                difM[(i-1)*xn1+j-1,(i-1)*xn1+j  -1]=difM[(i-1)*xn1+j-1,(i-1)*xn1+j-1]-ry
    for j in range(2,xn+3):
        difM[j-1,j-1]=1.0/(2.0*dx)
        difM[j-1,2*xn1+j-1]=-1.0/(2.0*dx)
        difM[j-1,xn1+j-1  ]=-hL
        difM[(xn1-1)*xn1+j-1,(xn1-1)*xn1+j  -1]=+1.0/(2.0*dx)
        difM[(xn1-1)*xn1+j-1,(xn1-3)*xn1+j  -1]=-1.0/(2.0*dx)
        difM[(xn1-1)*xn1+j-1,(xn1-2)*xn1+j  -1]=hR
    for i in range(2,xn+3):
        difM[(i-1)*xn1+1-1,(i-1)*xn1+1  -1]=1.0/(2.0*dx)
        difM[(i-1)*xn1+xn1-1,(i-1)*xn1+xn1  -1]=1.0/(2.0*dx)
        difM[(i-1)*xn1+1-1,(i-1)*xn1+2 -1]=1.0/-(2.0*dx)
        difM[(i-1)*xn1+xn1-1,(i-1)*xn1+xn1-1-1]=1.0/-(2.0*dx)   
    difM[0,0]=1.0
    difM[(xn1-1)*xn1+1-1,(xn1-1)*xn1+1 -1]=1.0
    difM[xn1-1,xn1 -1]=1.0
    difM[(xn1-1)*xn1+xn1-1,(xn1-1)*xn1+xn1 -1]=1.0
    b=scipy.sparse.lil_matrix((xn1*xn1,1))
    for j in range(2,xn1):
        b[(xn1-1)*xn1+j-1]=fR((j-2)*dx)
    zInit=scipy.sparse.linalg.spsolve(difM,b)
    return zInit

def runTest(contN,T,L,a1,a2,D,hL,hR,fR,fC,u,full):
    global b_Uext
    global Amipext
    global tn
    global xn
    global names
    xn1=xn+3
    tn1=tn+1
    i1=contN*contN
    dx=L/xn
    dt=T/tn
    #ishift=tn1*i1
    difM=constrFD(xn,tn,dx,dt,a1,a2,D,hL,hR)
    #base=scipy.sparse.csr_matrix([[0,0],[1,0]])
    #kron1=scipy.sparse.kron(scipy.sparse.eye(4),base)
    A=difM.toarray()
    #for i in contPos.rows:
    contPos = [[0. for xx in range(2)] for yy in range(i1)] 
    b = np.array([[0.0 for xx in range(contN*contN)] for yy in range(xn1*xn1*tn1)])
    bInhom = np.array([0.0 for xx in range(xn1*xn1*tn1)])
    zInit=calcInit(xn,dx,a1,a2,D,hL,hR,fR)
    bInhom[0:xn1*xn1]=np.transpose(zInit)
    for i in range(1,contN+1):
        for j in range(1,contN+1):
            contPos[(i-1)*contN+j-1][0]=i/(contN+1.)
            contPos[(i-1)*contN+j-1][1]=j/(contN+1.)
    
     
    for k in range(1,contN*contN+1):
        t=2
        for j in range(2,xn1):
            bInhom[(t-1)*xn1*xn1+(xn1-1)*xn1+j-1]=fR((j-2)*dx)
            for i in range(2,xn1):
               b[(t-1)*xn1*xn1+(i-1)*xn1+j-1,k-1]=fC((i-2.0)*dx,(j-2.0)*dx,contPos[k-1][0],contPos[k-1][1])
    
    for t in range(2,tn1+1):
        for j in range(2,xn1):
            bInhom[(t-1)*xn1*xn1+(xn1-1)*xn1+j-1]=fR((j-2)*dx)
    z=scipy.sparse.linalg.spsolve(A,b)
    zInhom=scipy.sparse.linalg.spsolve(A,bInhom)
    z1=np.array([[[0.0 for xx in range(tn1)] for yy in range(contN*contN)] for rr in range(tn1*xn1*xn1)])
    z1[:,:,2]=z
    for t in range(3,tn+2):
        z1[:,:,t-1]=np.concatenate((np.zeros(((t-1)*xn1*xn1,contN*contN)),z[xn1*xn1:(tn1-t+2)*xn1*xn1,:]),axis=0);
    
    zMat=zInhom.reshape(xn1,xn1,tn1,order='F')
    zMat=zMat[1:xn1-1,1:xn1-1,:]
    
    vec=np.multiply(2.0,np.ones((xn1,1)))
    vec[0]=0.0
    vec[xn1-1]=0.0
    vec[1]=1.0
    vec[xn1-2]=1.0
    
    forF=np.concatenate((vec,np.zeros((xn1,1)),np.multiply(-1.0,vec),np.zeros((xn1*(xn1-3),1))),axis=0)
    forF=scipy.sparse.lil_matrix(forF)
    F=scipy.sparse.kron(np.ones((tn1,1)),forF)
    F=F.toarray()
    coeff=np.multiply(-1.0*dt,np.matmul(np.transpose(F),z))
    uBasis=np.zeros((xn1-2,xn1-2,tn1,contN*contN))
    uInhom=np.zeros((xn1-2,xn1-2,tn1))
    
    for t1 in range(1,tn1+1):
        for x in range(2,xn1-1+1):
            for y in range(2,xn1-1+1):
                for i in range(1,contN*contN+1):
                    if abs(z[(t1-1)*xn1*xn1+(x-1)*xn1+y-1,i-1])>0.00001:
                        uBasis[x-2,y-2,t1-1,i-1]=z[(t1-1)*xn1*xn1+(x-1)*xn1+y-1,i-1]
    
    for t1 in range (1,tn1+1):
        for x in range(2,xn1-1+1):
            for y in range(2,xn1-1+1):
                if abs(zInhom[(t1-1)*xn1*xn1+(x-1)*xn1+y-1])>0.00001:
                    uInhom[x-1-1,y-1-1,t1-1]=zInhom[(t1-1)*xn1*xn1+(x-1)*xn1+y-1]
    
    maxCont=5
    Amip,b_L,b_U,c=constrAbc((xn+1)*(xn+1)*tn1, tn,xn,contN,uInhom,maxCont,z1)
    Amipred=Amip[0:1+(tn+1)*i1,:]
    b_Lred=b_L[0:1+(tn+1)*i1]
    b_Ured=b_U[0:1+(tn+1)*i1]
    Amipext=Amip[1+(tn+1)*i1:,:]
    b_Lext=b_L[1+(tn+1)*i1:]
    b_Uext=b_U[1+(tn+1)*i1:]
    Arow=Amipred.shape[0]
    Acol=Amipred.shape[1]
    #define reduced ones
    names=['']*Acol
    model = cplex.Cplex()
    modelFull=cplex.Cplex()
    
    for i in range(1,contN*contN+1):
        names[i-1]="bin"+str(i)
        if not full:
            model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["B"])
        else:
            modelFull.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["B"])

    for i in range(contN*contN+1,Acol+1):
        names[i-1]="cont"+str(i)
        if not full:
            model.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["C"])
        else:
            modelFull.variables.add(obj=[float(c[i-1])],names=[names[i-1]],lb=[0.0],ub=[1.0],types=["C"])
    if not full:
        for i in range(0,Arow):
            thevars=[]
            thecoefs=[]
            for j in range (0,Acol):
                if Amipred[i][j]!=0:
                    thevars.append(names[j])
                    thecoefs.append(Amipred[i][j])
            #print thevars
            #print thecoefs
            model.linear_constraints.add(lin_expr = [cplex.SparsePair(thevars,thecoefs)], senses = ["L"], rhs = [float(b_Ured[i])])
            model.linear_constraints.add(lin_expr = [cplex.SparsePair(thevars,thecoefs)], senses = ["G"], rhs = [float(b_Lred[i])])
        lazy_cb = model.register_callback(StateconstraintCallback)
        lazy_cb.number_of_calls = 0   
    else:
        for i in range(0,Amip.shape[0]):
            thevars=[]
            thecoefs=[]
            for j in range(0,Amip.shape[1]):
                if Amip[i][j]!=0:
                    thevars.append(names[j])
                    thecoefs.append(Amip[i][j])
            modelFull.linear_constraints.add(lin_expr = [cplex.SparsePair(thevars,thecoefs)], senses = ["L"], rhs = [float(b_U[i])])
            modelFull.linear_constraints.add(lin_expr = [cplex.SparsePair(thevars,thecoefs)], senses = ["G"], rhs = [float(b_L[i])])
    
    
    
    
    try:
        if not full:
            start=model.get_time()
            model.parameters.timelimit.set(1000)
            model.solve()
            end=model.get_time()
        else:
            start=modelFull.get_time()
            modelFull.solve()
            modelFull.parameters.timelimit.set(1000)
            end=modelFull.get_time()
        duration=end-start
    except CplexError as exc:
        print(exc)
    #print("LazyCbcalls: %i",lazy_cb.number_of_calls)
    #print(model.solution.status[model.solution.get_status()])
    #print(modelFull.solution.status[modelFull.solution.get_status()]) 
    #numrows = model.linear_constraints.get_num()
    #numcols = model.variables.get_num()
    
    # solution.get_status() returns an integer code
    #print("Solution status = ", model.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    #print(model.solution.status[model.solution.get_status()])
    #print("Solution value  = ", model.solution.get_objective_value())
    #slack = model.solution.get_linear_slacks()
    #x = model.solution.get_values()
    #x_k=np.transpose(np.array([x]))
    #state=b_Uext-np.matmul(Amipext,x_k)
    #scipy.io.savemat('stateConta.mat', dict(x=x))
    return duration
    
#for i in range(numrows):
#    print("Row %d:  Slack = %10f" % (i, slack[i]))
#for j in range(numcols):
#    print("Column %d:  Value = %10f" % (j, x[j]))

#xn=10
#tn=30
contN=5
T=18.0
L=1.0
a1 = lambda x,y:  -0.02-0.04*(y<0.7)
a2 = lambda x,y:  0.0*y
D= lambda x,y: 0.001+0.004*(y>0.3 and y < 0.7)
hL=-20.0
hR=1.0
fR=lambda x: 0.0*x+hR
fC=lambda x,y,xc,yc: -0.4*np.exp(-20.0*((x-xc)*(x-xc)+(y-yc)*(y-yc)))
u= lambda x,y: 0.0
durations=[]
durationsFull=[]

for tn in range(60,100,5):
    durations.append([])
    #durationsFull.append([])
    for xn in range(12,24,2):
        durations[-1].append(runTest(contN,T,L,a1,a2,D,hL,hR,fR,fC,u,0))
        print("runTestDone")
        #durationsFull[-1].append(runTest(contN,T,L,a1,a2,D,hL,hR,fR,fC,u,1))
print(durations)
print(durationsFull)

