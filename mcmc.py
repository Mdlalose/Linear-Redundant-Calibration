import numpy as np
import math
import  matplotlib.pyplot  as plt
## this is an array of frequencies form 1-100MHz
import get_chi2_func
import sys
#simulated data

xdim = int(sys.argv[1])
ydim = int(sys.argv[2])
#print ydim, xdim

xx=np.arange(int(xdim))
yy=np.arange(int(ydim))

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



# fake antenna gains
eta= np.random.normal(0.0,1.0)*np.random.randn(xdim*ydim)
amp = np.exp(eta)
phase = np.random.uniform(0.0,math.pi)*np.random.randn(xdim*ydim)
gains= amp*(np.cos(phase)+ 1j*np.sin(phase))

#gains = np.ones(xdim*ydim,dtype='complex')
#gains[0]= gains_s[0]

# fake true sky signal for n_unique baselines

amp= np.random.normal(0.0,0.5)*np.random.randn(n_unique)
phase = np.random.uniform(0.0,math.pi)*np.random.randn(n_unique)
sky_true =amp*(np.cos(phase)+ 1j*np.sin(phase))
     

# fake visibilities
data =np.conj(gains[ant1])*gains[ant2]*sky_true[vis_map] # + 0.01*np.random.randn(q.size)

def model_vis(g_0,s_0):
     return np.conj(g_0[ant1])*g_0[ant2]*s_0[vis_map]


noise_frac_gains = float(sys.argv[3])
noise_frac_sky = float(sys.argv[4])


gains = gains  + noise_frac_gains*np.random.randn(gains.size) # gain_0 
#g_0 = g_0/np.mean(g_0) 
g_0= gains
s_0 = sky_true + noise_frac_sky*np.random.randn(sky_true.size)


def get_chisqd(data,theta):
    chi=0.0
    for vis in range(data.size):
            error= data[vis] - np.conj(theta[0][ant1[vis]])*theta[0][ant2[vis]]*theta[1][vis_map[vis]] 
            chi += (np.conj(error)*error)
    
    return chi

theta = np.array([g_0,s_0])



###mcmc

N_steps= int(sys.argv[5])
Gains=[]
Sky=[]
N1=0
count=[]
for i in range(N_steps):
    delta_s_0 = np.zeros(n_unique,dtype="complex") 
    delta_g_0 = np.zeros(xdim*ydim,dtype="complex")

    for i in range(xdim*ydim):
           delta_g_0[i] = np.random.uniform(0.1,1.0)
    for j in range(n_unique):
               delta_s_0[j] = np.random.normal(0.0,0.5)


    delta_theta = np.array([delta_g_0,delta_s_0])
    u=theta + delta_theta
    
    R= (0.5*np.exp((get_chisqd(data,theta)-get_chisqd(data,u))))
    r=np.random.uniform(0,1)
    print R

    if R>1:
        N1+=1
        count.append(N1)
        theta = u
        print u
        Gains.append(theta[0])
        Sky.append(theta[1])
    elif R>r:
        N1+=1
        count.append(N1)
        theta=u
    
        Gains.append(theta[0])
        Sky.append(theta[1])
        
        
    else:
        theta=theta
        





