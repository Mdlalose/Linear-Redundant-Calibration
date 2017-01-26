# test script
import numpy as np, math 
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy import linalg as lin
import lincal as omnical
import get_chi2_func
import sys

# 1st & 2nd  argv  dimensions of an array
# 3rd & 4th argv are the noise added to simulated gains nd true sky
# 5th arg is number of iteration

# simulate fake data nxn array element
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
phase = np.random.uniform(0.0,math.pi/20.0)*np.random.randn(xdim*ydim)
gains= amp*(np.cos(phase)+ 1j*np.sin(phase))

#gains = np.ones(xdim*ydim,dtype='complex')
#gains[0]= gains_s[0]

# fake true sky signal for n_unique baselines

amp= np.random.normal(0.0,0.5)*np.random.randn(n_unique)
phase = np.random.uniform(0.0,math.pi/20.0)*np.random.randn(n_unique)
sky_true =amp*(np.cos(phase)+ 1j*np.sin(phase))
     

# fake visibilities
Vis_data =np.conj(gains[ant1])*gains[ant2]*sky_true[vis_map] # + 0.01*np.random.randn(q.size)
"""

ant1,ant2,vis_map = np.load('ants_data_100_point_src_0.1_pos_dev.npy')[0], np.load('ants_data_100_point_src_0.1_pos_dev.npy')[1],np.load('ants_data_100_point_src_0.1_pos_dev.npy')[2]
sky_true, gains =  np.load('data_vis_uniq_100_point_src_0.1_pos_dev.npy'),np.load('data_gains_100_point_src-0.1_pos_dev.npy') 
Vis_data = np.load('data_vis_100_point_src_0.1_pos_dev.npy') 

"""
noise_frac_gains = np.random.normal(0.0,float(sys.argv[3]))
noise_frac_sky = np.random.normal(0.0,float(sys.argv[4]))


gains = gains  + noise_frac_gains*np.ones(gains.size) # gain_0 
#g_0 = g_0/np.mean(g_0) 
g_0= gains
s_0 = sky_true + noise_frac_sky*np.ones(sky_true.size)

def  model_vis(g,sky,vis_map):
  
    return np.conj(g[ant1])*g[ant2]*sky[vis_map]


# lincal function


def linncal_func(Vis,g_0,s_0,N_steps):
    Parameter ={}
    #chi_lin =np.zeros(N_steps)
    chi=[]
 
    for j in range(N_steps):
        P = omnical.lincal(Vis,g_0,s_0,ant1,ant2,vis_map)
        Parameter[j] =P
        
       
        s_0 = P[1]
        g_0 = P[0]
        
        chi.append(get_chi2_func.get_chisqd(Vis,P[0],P[1],ant1,ant2,vis_map))
        
     
    return np.array([Parameter,chi ])


n_iter =  int(sys.argv[5])			
#data_inter = linncal_func(Vis_data,g_0,s_0,n_iter)[0] 
#chi_sqd  = np.array(linncal_func(Vis_data,g_0,s_0,n_iter)[1])

data_iter = omnical.get_lincal(Vis_data,g_0,s_0,n_iter,ant1,ant2,vis_map)
 
chi_sqd  = data_iter[2]







# residual fraction


#REAL PART
#plt.plot(model_vis(data_inter[n_iter-1][0],data_inter[n_iter-1][1],vis_map).real/Vis_data.real,'.',label='100 iteration ')
plt.plot(model_vis(data_iter[0][data_iter[3]],data_iter[1][data_iter[3]],vis_map).real/Vis_data.real,'.',label='100 iteration ')

#plt.plot(np.ones(len(model_vis(data_inter[n_iter-1][0],data_inter[n_iter-1][1],vis_map))),'r',label='VRF=1.0')

plt.ylabel(r'Visibility Residual Fraction ')

plt.title('Real Part')
plt.legend(loc = 'best')
plt.show()
"""
# Imaginary part
plt.plot(model_vis(data_inter[n_iter-1][0],data_inter[n_iter-1][1],vis_map).imag/Vis_data.imag,'.',label='3rd Iteration')


plt.plot(np.ones(len(model_vis(data_inter[n_iter-1][0],data_inter[n_iter-1][1],vis_map))),'r',label='VRF=1.0')

plt.ylabel(r'Visibility Residual Fraction')
plt.title('Imaginary Part')
#plt.legend(loc = 'best')
plt.show()
"""


plt.plot(Vis_data.real,Vis_data.imag,'>r',label='Simulated Visibilities')
#plt.plot(model_vis(data_inter[n_iter-1][0],data_inter[n_iter-1][1],vis_map).real,model_vis(data_inter[n_iter-1][0],data_inter[n_iter-1][1],vis_map).imag,'.k',label='Best Fit Visibilities')

plt.plot(model_vis(data_iter[0][data_iter[3]],data_iter[1][data_iter[3]],vis_map).real,model_vis(data_iter[0][data_iter[3]],data_iter[1][data_iter[3]],vis_map).imag,'.k',label='Best Fit Visibilities')
print ' std ', np.std(model_vis(data_iter[0][data_iter[3]],data_iter[1][data_iter[3]],vis_map)- Vis_data)/np.std(model_vis(data_iter[0][data_iter[3]],data_iter[1][data_iter[3]],vis_map))
plt.ylabel(r'Imaginary Part ')
plt.xlabel('Real Part')
#plt.title('3rd Iteration')

plt.legend(loc = 'best')
plt.grid(True)
plt.axis('equal')
plt.show()
"""
plt.plot(Vis_data.real, model_vis(data_iter[0][data_iter[3]],data_iter[1][data_iter[3]],vis_map).real, '*')
plt.xlabel('input visibilities')
plt.ylabel('output visibilties')
plt.show()


plt.plot(gains.real,data_iter[0][data_iter[3].real, '*')
plt.xlabel('input gains')
plt.ylabel('output output')
plt.show()

plt.plot(sky_true.real,.real, '*')
plt.xlabel('input sky')
plt.ylabel('output sky')
plt.show()
"""
##############################################################################
####################### chi squared plots ##################################

plt.plot(chi_sqd)
plt.xlabel('N_iterations')
plt.ylabel(r'$log(\chi^2)$')

#plt.axis([-10,150, -10.0, chisqd[-1]])
plt.show()

plt.plot(gains.real, gains.imag,'<', label ='Simulated Input gains factors')
plt.plot(data_iter[0][data_iter[3]].real,data_iter[0][data_iter[3]].imag,'.', label ='Best Fit gain factors')

plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend(loc="best")
plt.show()


plt.plot(sky_true.real,sky_true.imag,'*',label='Simulated True Sky')
plt.plot(data_iter[1][data_iter[3]].real,data_iter[1][data_iter[3]].imag,'v',label='Best Fit True Sky')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend(loc='best')
plt.show()


			
