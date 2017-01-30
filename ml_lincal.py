import numpy as np, math 
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy import linalg as lin
import get_chi2_func
import sys


def get_config_matrix(case,n_ants,ant1,ant2,vis_map,gain_param,sky_param,q,n_unique):
    # real vis, gains and sky'R==1'
    # configaration matrix for real and complex cases
    if case is 1:
        # real data
        A = np.zeros((q.size, n_ants + n_unique))
        for vis_ in range(q.size):
            A[vis_][n_unique + ant1[vis_]]   = gain_param[ant2[vis_]]*sky_param[vis_map[vis_]]
            A[vis_][n_unique + ant2[vis_]]  = gain_param[ant1[vis_]]*sky_param[vis_map[vis_]]
            A[vis_][vis_map[vis_]]          = gain_param[ant1[vis_]]*gain_param[ant2[vis_]]
        return A
        
    else:
        #complex vis, gains and sky
        A = np.zeros((q.size, n_ants + n_unique),dtype ="complex")
        for vis_ in range(q.size):
            A[vis_][ n_unique + ant1[vis_]]   = gain_param[ant2[vis_]]*sky_param[vis_map[vis_]]
            A[vis_][n_unique + ant2[vis_]]  = np.conj(gain_param[ant1[vis_]])*sky_param[vis_map[vis_]]
            A[vis_][vis_map[vis_]]          =      np.conj(gain_param[ant1[vis_]])*gain_param[ant2[vis_]]
        return A


def get_ml_func(data,g,s,ant1,ant2,vis_map,n_steps,lambda_0=0.01,eps=0.1,l_up= 11.0,l_down =9.0):
	S=[]
        G=[]
        Chi2=[]
	for step in range(n_steps):
		B = get_config_matrix(2,g.size,ant1,ant2,vis_map,g,s,data,s.size)
                Curv = np.conj(B).T.dot(np.conj(B))
                lambda_0 = lambda_0*np.max(np.diag(Curv))
		#print lambda_0,np.max(np.diag(Curv))
                
                mod_curv = Curv + lambda_0*np.diag(Curv)
           
                h_lm = np.linalg.pinv(mod_curv).dot(B.T).dot(data -np.conj(g[ant1])*g[ant2]*s[vis_map])
          
                g_1 = g + h_lm[s.size:len(h_lm)]
		s_1 = s + h_lm[0:s.size]
                #print h_lm
                
		delta_chi2 = (get_chi2_func.get_chisqd(data,g,s,ant1,ant2,vis_map)-get_chi2_func.get_chisqd(data,g_1,s_1,ant1,ant2,vis_map))
		delta_h_lm = h_lm.T.dot(lambda_0*np.diag(Curv).dot(h_lm) + B.T.dot(data-np.conj(g[ant1])*g[ant2]*s[vis_map]))
		
		rho_h_lm = delta_chi2/delta_h_lm
                #print rho_h_lm
		
                
		alpha_up = (B.T.dot(data-np.conj(g[ant1])*g[ant2]*s[vis_map])).T.dot(h_lm)
                
                
		alpha_d = (get_chi2_func.get_chisqd(data,g_1,s_1,ant1,ant2,vis_map) -get_chi2_func.get_chisqd(data,g,s,ant1,ant2,vis_map))/2.0 + 2.0*(B.T.dot(data-np.conj(g[ant1])*g[ant2]*s[vis_map])).T.dot(h_lm)
                 
                
		alpha =  alpha_up/alpha_d
                #print alpha
		
                if rho_h_lm> eps:
			g= g + alpha*h_lm[s.size:len(h_lm)]
			s = s + alpha*h_lm[0:s.size]
                        S.append(s)
                        G.append(g)
                        Chi2.append(get_chi2_func.get_chisqd(data,g,s,ant1,ant2,vis_map))

			lambda_new = np.maximum(lambda_0/(1.0 + alpha),10**-7)
			#lambda_0 = np.maximum(lambda_0/l_down, 10**-7)
                        
		else:
			g_1= g + alpha*h_lm[s.size:len(h_lm)]
			s_1 = s + alpha*h_lm[0:s.size]

			lambda_0 = lambda_0 + np.linalg.norm(get_chi2_func.get_chisqd(data,g_1,s_1,ant1,ant2,vis_map) -get_chi2_func.get_chisqd(data,g,s,ant1,ant2,vis_map))/(2.0*alpha)
                        #lambda_0 = np.minimum(lambda_0*l_up,10**7)

                
		
                     
               

                     

      

        return [G,S,Chi2] 

def  model_vis(g,sky,vis_map):
  
    return np.conj(g[ant1])*g[ant2]*sky[vis_map]
###########################testing ml ##############################################################

ant1,ant2,vis_map = np.load('ants_data_100_point_src_0.1_pos_dev.npy')[0], np.load('ants_data_100_point_src_0.1_pos_dev.npy')[1],np.load('ants_data_100_point_src_0.1_pos_dev.npy')[2]
sky_true, gains =  np.load('data_vis_uniq_100_point_src_0.1_pos_dev.npy'),np.load('data_gains_100_point_src-0.1_pos_dev.npy') 
Vis_data = np.load('data_vis_100_point_src_0.1_pos_dev.npy') 

noise_frac_gains =float(sys.argv[3])
noise_frac_sky = float(sys.argv[4])


gains = gains  + noise_frac_gains*np.ones(gains.size) # gain_0 
#g_0 = g_0/np.mean(g_0) 
g_0= gains
s_0 = sky_true + noise_frac_sky*np.ones(sky_true.size)
  
n_iter = int(sys.argv[5])
fit_p = get_ml_func(Vis_data,gains,sky_true,ant1,ant2,vis_map,n_iter)

plt.plot(Vis_data.real,Vis_data.imag,'>r',label='Simulated Visibilities')


plt.plot(model_vis(fit_p[0][-1],fit_p[1][-1],vis_map).real,model_vis(fit_p[0][-1],fit_p[1][-1],vis_map).imag,'.k',label='LM Best Fit Visibilities')
print ' std ', np.std(model_vis(fit_p[0][-1],fit_p[1][-1],vis_map)- Vis_data)/np.std(model_vis(fit_p[0][-1],fit_p[1][-1],vis_map))
plt.ylabel(r'Imaginary Part ')
plt.xlabel('Real Part')
#plt.title('3rd Iteration')

plt.legend(loc = 'best')
plt.grid(True)
plt.axis('equal')
plt.show()


plt.plot(fit_p[2])
plt.xlabel('N_iterations')
plt.ylabel(r'$\chi^2$')
plt.title('Levenberg-Marqaudt-Lincal Method')

#plt.axis([-10,150, -10.0, chisqd[-1]])
plt.show()

plt.plot(gains.real, gains.imag,'<', label ='Simulated Input gains factors')
plt.plot(fit_p[0][-1].real,fit_p[0][-1].imag,'.', label ='LM Best Fit gain factors')

plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend(loc="best")
plt.show()


plt.plot(sky_true.real,sky_true.imag,'*',label='Simulated True Sky')
plt.plot(fit_p[1][-1].real,fit_p[1][-1].imag,'v',label='LM Best Fit True Sky')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend(loc='best')
plt.show()


