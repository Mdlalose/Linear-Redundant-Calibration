import numpy as np, math
import matplotlib.pyplot as plt
from scipy.optimize import minimize as min_newton_cg
import get_chi2_func
import lincal

ant1,ant2,vis_map = np.load('ants_data_1_point_src_0.3_dev.npy')[0], np.load('ants_data_1_point_src_0.3_dev.npy')[1],np.load('ants_data_1_point_src_0.3_dev.npy')[2]
sky,gains =  np.load('data_vis_uniq_1_point_src_0.3_dev.npy'),np.load('data_gains_100_point_src_0.3_dev.npy') 
data = np.load('data_vis_1_point_src_0.3_dev.npy') 

def get_lin_chi(x):
    gains_0 = x[0:gains.size]					
    gains_new = gains_0 + 0.00001*np.random.randn(gains_0.size)
    sky_0 = x[gains.size:x.size]
    sky_new = sky_0 + 0.00001*np.random.randn(sky_0.size)					
    return lincal.get_lincal_chisqd(data,gains_0,sky_0,gains_new,sky_new,ant1,ant2,vis_map)

def get_lin_grad(x):
    gains_0 = x[0:gains.size]
    gains_new = gains_0 + 0.00001*np.random.randn(gains.size)
    sky_0 = x[gains.size:x.size]
    sky_new = sky_0 + 0.00001*np.random.randn(sky_0.size)
    return lincal.get_lincal_grad(data,gains_0,sky_0,gains_new,sky_new,ant1,ant2,vis_map)

def get_lin_curv(x):
    gains_0 =x[0:gains.size]
    gains_new = gains_0 + 0.00001*np.random.randn(gains_0.size)
    sky_0 = x[gains_0.size:x.size]
    sky_new = sky_0 + 0.00001*np.random.randn(sky_0.size)
    return lincal.get_lincal_curv(data,gains_0,sky_0,gains_new,sky_new,ant1,ant2,vis_map)


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


### ##################################### data test#############################################

"""
xdim =4
ydim = 4
    
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
     


data =np.conj(gains[ant1])*gains[ant2]*sky[vis_map] #+ 0.1*np.random.randn(q.size)
"""
gains_0 =gains + 0.001*np.random.randn(gains.size)
sky_0 = sky + 0.001*np.random.randn(sky.size)

x0= np.concatenate((gains_0,sky_0))
Nfeval =1
def callbackF(x):
    global Nfeval
    print '{0:4d} {1:3.6f}'.format(Nfeval,chi_func(x))
    Nfeval +=1

print 'full chi square'
x_fits = min_newton_cg(chi_func,x0,callback=callbackF,method='Newton-CG',jac=chi_func_grad,hess=chi_func_curv,options={'xtol':1e-10,'disp': True})
print 'linearize chi sqaured'

x_lin_fits = min_newton_cg(get_lin_chi,x0,callback=callbackF,method='Newton-CG',jac=get_lin_grad,hess=get_lin_curv,options={'xtol':1e-10,'disp':True})
print get_lin_chi(x0)
#plt.ion()
"""
plt.plot(gains.real,gains.imag,'.', label='input gains')
plt.plot(x_fits.x[0:gains.size].real,x_fits.x[0:gains.size].imag,'*', label='output gains')
plt.xlabel('real part')
plt.ylabel('imaginary part')
plt.legend(loc='best')
plt.show()
		

plt.plot(sky.real,sky.imag,'.', label='input true sky')
plt.plot(x_fits.x[gains.size:x_fits.x.size].real,x_fits.x[gains.size:x_fits.x.size].imag,'.', label='output true sky')
plt.xlabel('real part')
plt.ylabel('imaginary part')
plt.legend(loc='best')
plt.show()
"""
best_fit_gains =x_fits.x[0:gains.size]
best_fit_sky = x_fits.x[gains.size:x_fits.x.size]
best_fits_data = np.conj(best_fit_gains[ant1])*best_fit_gains[ant2]*best_fit_sky[vis_map]
							
		
plt.title(r'Exact $\chi^2$')
plt.plot(best_fits_data.real,best_fits_data.imag,'.', label='NW-CG Best fits visibility')
plt.plot(data.real,data.imag,'*', label='Simulsted visibility')
plt.xlabel('real part')
plt.ylabel('imaginary part')
plt.legend(loc='best')
plt.grid()
plt.show()
																																																																																													
plt.plot(best_fits_data.real/data.real,'.')
plt.ylabel('Visibility Residual Fraction')
plt.title('Real Part')
plt.legend(loc='best')
plt.show()
											
plt.plot(best_fits_data.imag/data.imag,'.')
plt.ylabel('Visibility Residual Fraction')
plt.title('Imaginary Part')
plt.legend(loc='best')
plt.show()


							
