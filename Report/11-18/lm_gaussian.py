import numpy as np
from matplotlib import pyplot as plt

def gauss(m,x):
    #get y=a+b*exp((x-x0)^2/2sig^2)
    a=m[0]
    b=m[1]
    x0=m[2]
    sig=m[3]
    expvec=np.exp(-0.5*(x-x0)**2/sig**2)
    y=a+b*expvec
    derivs=np.empty([len(x),len(m)])
    derivs[:,0]=1
    derivs[:,1]=expvec
    derivs[:,2]=b*(x-x0)*2/(2*sig**2)*expvec
    derivs[:,3]=b*(0.5*(x-x0)**2)*2/sig**3*expvec

    return y,derivs

def update_lamda(lamda, success):
    if success:
        lamda = lamda/1.5
        if lamda<0.5:
            lamda=0
    else:
        if lamda==0:
            lamda=1
        else:
            #lamda = lamda*1.5**2
            lamda = lamda*2
    return lamda

def get_matrices(m, fun, x, y, Ninv=None):
    model, derivs = fun(m, x)
    r = y-model
    if Ninv is None:
        lhs = derivs.T@derivs
        rhs = derivs.T@r
        chisq = np.sum(r**2)
    else: 
        lhs=derivs.T@Ninv@derivs
        rhs=derivs.T@(Ninv@r)
        chisq=r.T@Ninv@r
    return chisq, lhs, rhs

def linv(mat, lamda):
    mat=mat+lamda*np.diag(np.diag(mat))
    return np.linalg.pinv(mat)

def LM(m, fun, x, y, Ninv=None, niter=20, chitol= 1):
#levenberg-Marquardt Fitter 
    lamda=0
    lamda_trend = []
    lamda_trend.append(lamda)
    chisq, lhs, rhs = get_matrices(m, fun, x, y, Ninv)
    for i in range(niter):
        lamda_trend.append(lamda)
        lhs_inv = linv(lhs, lamda)
        dm = lhs_inv@rhs
        #result = check_limits(m+dm, low_bound, up_bound)
        #print('m: ', m)
        #print('dm: ', dm)
        #if (result==False):
         #   print('I am here 1!')
        """
        while (True): # to chack if  all the parameters are in the reasonable limits
            lhs_inv = linv(lhs, lamda)
            dm = lhs_inv@rhs
            result = check_limits(m+dm, low_bound, up_bound)           
            if (result == True):
                break
            else:
                lamda = update_lamda(lamda, False)
                print('I am here 1!')
                #print('out of range params: ', m+dm)
                lamda_trend.append(lamda)
        """      
        chisq_new, lhs_new, rhs_new = get_matrices(m+dm, fun, x, y, Ninv)
        if chisq_new<chisq:  
            #accept the step
            #check if we think we are converged - for this, check if 
            #lamda is zero (i.e. the steps are sensible), and the change in 
            #chi^2 is very small - that means we must be very close to the
            #current minimum
            if lamda==0:
                if (np.abs(chisq-chisq_new)<chitol):
                    print(np.abs(chisq-chisq_new))
                    print('Converged after ', i, ' iterations of LM')
                    return m+dm
            chisq = chisq_new
            lhs = lhs_new
            rhs = rhs_new
            m = m+dm
            lamda = update_lamda(lamda,True)
            #lamda_trend.append(lamda)
            
        else:
            lamda = update_lamda(lamda, False)
            #print('I am here 2!')
            #lamda_trend.append(lamda)
        print('\n', 'on iteration ', i, ' chisq is ', chisq, ' with step ', dm, ' and lamda ', lamda)
        #print('params are: ', m)
    return m, lamda_trend

x=np.linspace(-5,5,1001)
m_true=np.asarray([0.5,1.5,-0.5,1])
y_true,derivs=gauss(m_true,x)

plt.ion()
plt.clf()
plt.plot(x,y_true)
plt.show()
y=y_true+np.random.randn(len(x))
plt.plot(x,y,'.')

m0=m_true+np.random.randn(len(m_true))*2.0
m_fit = LM(m0,gauss,x,y,niter=20)
y_fit,derivs=gauss(m_fit,x)
plt.plot(x,y_fit)