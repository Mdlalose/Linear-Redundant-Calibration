import numpy as np, math
import matplotlib.pyplot as plt
from scipy.optimize import minimize as min_newton_cg
import get_chi2_func



xdim =3
ydim = 3
    
xx=np.arange(xdim)
yy=np.arange(ydim)
x,y=np.meshgrid(xx,yy) # put antennnas position into a xdim by ydim grid
x=np.reshape(x,[x.size,1])
y=np.reshape(y,[y.size,1])
ant=x*xdim + y     # making a 1d list of antenna indexes from 0 to 48
x1,x2=np.meshgrid(x,x) # putting  position x into a N by N grid
ant1,ant2=np.meshgrid(ant,ant) # putting antennas into a grid
y1,y2=np.meshgrid(y,y) 
# computing baseline in x and y direction

q=y2-y1
u=x2-x1
q=np.reshape(q,[q.size,1])
u=np.reshape(u,[u.size,1])
ant1=np.reshape(ant1,[ant1.size,1]) # reshaping ant1 into 1d of len 4
ant2=np.reshape(ant2,[ant2.size,1]) #reshaping ant2 into 1d of len 4
# removing repeated antennas
isgood=ant1>ant2
#selecting q,u are not repeated and corresponding ant1 and ant2
q=q[isgood]
u=u[isgood]
ant1=ant1[isgood]
ant2=ant2[isgood]

        


qu=q+np.complex(0,1)*u
qu=np.unique(qu) # selecting unique baselines

n_unique=(xdim)**2-1+(xdim-1)**2

#print   repr(n_unique)
    
    
q_unique=qu.real
u_unique=qu.imag
 # this is map of visibilities which group them  according to their redundant unique group set"
vis_map=0*q
for ind in range(q.size):
          #print ind, q_unique, q[ind]
          unique_ind=np.where( (q[ind]==q_unique) & (u[ind]==u_unique))
          
          #print myind, ind
          vis_map[[ind]]=unique_ind


gains = np.zeros(ydim**2,dtype='complex')

for g in range(ydim**2):
     eta= np.random.normal(0.0,1.0)
     amp = np.exp(eta)
     phase = np.random.uniform(0.0,2.0*math.pi)
     gains[g]=amp*(np.cos(phase)+ 1j*np.sin(phase))






    
sky = np.zeros(n_unique,dtype='complex')

for s in range(n_unique):
     amp= np.random.normal(0.0,0.5)
     phase = np.random.uniform(0.0,2.0*math.pi)
     sky[s]=amp*(np.cos(phase)+ 1j*np.sin(phase))
     


data =np.conj(gains[ant1])*gains[ant2]*sky[vis_map] + 0.01*np.random.randn(q.size)

##################################################################################################################################################################################
########################################## finding the best fit parameters (antenna gain factors and true sky signal) using Newton Multivaraite Methods #########################
#################################################################################################################################################################################

gains = gains #+ 0.000001*np.random.randn(gains.size)
sky = sky #+ 0.01*np.random.randn(sky.size)

#print nw.get_fits_param(100,np.power(10,-6),data,gains,sky,ant1,ant2,vis_map)


##################################################################################################################################################################################
############################################# Gradient and Curvature Test ########################################################################################################
##################################################################################################################################################################################

#print  get_chi2_func.get_grad_chisqd(data,gains,sky,ant1,ant2,vis_map).shape, get_chi2_func.get_curv_chisqd(data,gains,sky,ant1,ant2,vis_map).shape

"""
#chi2 test


CHI_CALC=[]
CHI_PRED_GRAD=[]
CHI_PRED_CURV=[]
GAINS=[]
SKY=[]
gains_old=gains
chi2_old = get_chi2_func.get_chisqd(data,gains_old,sky,ant1,ant2,vis_map)
grad_old = get_chi2_func.get_grad_chisqd(data,gains_old,sky,ant1,ant2,vis_map)
curv_old = get_chi2_func.get_curv_chisqd(data,gains_old,sky,ant1,ant2,vis_map)
for t in range(1000):
      rand = np.zeros(gains.size,dtype='complex')
      rand[0] = 0.1*np.random.randn()
      gains_new = gains_old #+ rand
      #sky
      rand_sky = np.zeros(sky.size,dtype='complex')
      rand_sky[0] = 0.1*np.random.randn()
      sky_new = sky + rand_sky
      param_new = np.concatenate((gains_new,sky_new))
      param_old = np.concatenate((gains_old,sky))
      
      #print gains_old[0],gains_new[0],rand[0]
      CHI_CALC.append(get_chi2_func.get_chisqd(data,gains_new,sky_new,ant1,ant2,vis_map))
      chi_pred = chi2_old + (param_new-param_old).dot(grad_old)
      chi_pred_curv = chi2_old + (param_new-param_old).dot(grad_old) + 0.5*(param_new-param_old).T.dot(curv_old).dot(param_new-param_old)
      CHI_PRED_CURV.append(chi_pred_curv)
      CHI_PRED_GRAD.append(chi_pred)
      GAINS.append(gains_new[0])
      SKY.append(sky_new[0])




plt.plot(SKY,CHI_CALC, '.',label= '$\chi^2$')
#plt.plot(GAINS,CHI_PRED_CURV,'.',label='$\chi^2$ Pred Curv')
#plt.plot(GAINS,np.array(CHI_CALC)-np.array(CHI_PRED_CURV),'.',label ='$\chi^2$ Residual')
plt.plot(SKY,CHI_PRED_GRAD,'.', label= '$\chi^2$ Pred')
plt.plot(SKY,np.array(CHI_CALC)-np.array(CHI_PRED_GRAD),'.',label ='$\chi^2$ Residual')
plt.plot(sky[0],np.min(np.array(CHI_CALC)-np.array(CHI_PRED_CURV)),'*',label='$s_0$')
plt.plot(sky[0],chi2_old,'*',label='$\chi_{0}^2$, $s_{0}$')
plt.plot([sky[0],sky[0]],[chi2_old,np.min(np.array(CHI_CALC)-np.array(CHI_PRED_CURV))])
plt.xlabel('Sky')
plt.ylabel('$\chi^2$')
plt.legend(loc='best')
plt.show()
"""

def chi_func(x):
     
     gains_p = x[0:gains.size]
     sky_p = x[gains.size:x.size]
     return get_chi2_func.get_chisqd(data,gains_p,sky_p,ant1,ant2,vis_map)
    

def chi_func_grad(x):
     
     gains_p = x[0:gains.size]
     sky_p = x[gains.size:x.size]
     return get_chi2_func.get_grad_chisqd(data,gains_p,sky_p,ant1,ant2,vis_map)

def chi_func_curv(x):
     
     gains_p = x[0:gains.size]
     sky_p = x[gains.size:x.size]
     return get_chi2_func.get_curv_chisqd(data,gains_p,sky_p,ant1,ant2,vis_map)


x0= np.concatenate((gains,sky))


x_fits = min_newton_cg(chi_func,x0,method='Newton-CG',jac=chi_func_grad,hess=chi_func_curv,options={'xtol': 1e-1, 'disp': True})

