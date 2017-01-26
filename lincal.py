#Linear Rendundant calibration schem

import numpy as np, math 
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy import linalg as lin
import get_chi2_func


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




def Pinv(Matrix,eigen_threshold=10**-6):
    #Matrix = Matrix.conjugate()
    #this function compute the psuedo invirse
    U,s_value,V= lin.svd(Matrix)
    S_inv= np.zeros((len(s_value),len(s_value)))
    
    for ss in range(len(s_value)):
        #print s_value[ss]
        if s_value[ss] <= eigen_threshold :
            #print s_value[ss]
            S_inv[ss][ss] = 0.0

        else:
            s_inv =1.0/s_value[ss]
            #print s_inv
            S_inv[ss][ss]= s_inv
            

    


    return V.T.dot(S_inv).dot(U.T)

def  model_vis(g,sky,ant1,ant2,vis_map):
    return np.conj(g[ant1])*g[ant2]*sky[vis_map]

# lincal function

def lincal(data,g_0,s_0,ant1,ant2,vis_map):
     # linear Redundant calibration algorithm
     # this function compute deviations in gains & sky and then update the guess paramters
 
     #for k in range(N_step):
     B= get_config_matrix(2,g_0.size,ant1,ant2,vis_map,g_0,s_0,data,s_0.size)
     #B = np.matrix(B)
     Curvature_matrix = np.diag(2.0*np.conj(B).T.dot(B))
     c
     #Curvature_matrix = B.getH().dot(B)

     #delta_X = Pinv(Curvature_matrix).dot(A.T).dot(data-model_vis(g_0,s_0,ant1,ant2,vis_map)
     delta_X = lin.pinv(Curvature_matrix).dot(np.conj(B).T).dot(data-model_vis(g_0,s_0,ant1,ant2,vis_map))
     #delta_X = lin.pinv(Curvature_matrix).dot(B.getH()).dot(data-model_vis(g_0,s_0,ant1,ant2,vis_map))
     g_1 = g_0 + delta_X[s_0.size:len(delta_X)]
     s_1 = s_0 + delta_X[0:s_0.size]
     #chi= get_chisqd(,data,s_1,ant1,ant2,vis_map)
     return np.array([g_1,s_1,delta_X])
				
def get_lincal(data,g_0,s_0,N_steps,ant1,ant2,vis_map):
        """ This function compute the best fit paramters given intial guuesses"""
	Gains ={}
	Sky ={}
        chi2 =[]
        N= 0
	for k in range(0,N_steps):
	                B= get_config_matrix(2,g_0.size,ant1,ant2,vis_map,g_0,s_0,data,s_0.size)
     			#B = get_chi2_func.get_grad_chisqd(data,g_0,s_0,ant1,ant2,vis_map)
			Curvature_matrix = np.conj(B).T.dot(B)
			#Curvature_matrix = np.diag(Curvature_matrix)
                        #Curvature_matrix = get_chi2_func.get_curv_chisqd(data,g_0,s_0,ant1,ant2,vis_map)
 			#Curvature_matrix = np.diag(Curvature_matrix)

     			delta_X = lin.pinv(Curvature_matrix).dot(np.conj(B).T).dot(data-model_vis(g_0,s_0,ant1,ant2,vis_map))
                        #delta_X = lin.pinv(Curvature_matrix).dot(B)
                        
			     			
			if k%2 ==0 :
                       		g_1 = g_0 + delta_X[s_0.size:len(delta_X)]
     		       		s_1 = s_0 + delta_X[0:s_0.size]
                      		Gains[k] = g_1
                       		Sky[k] = s_1
                       		
                       		
                                #print s_0, s_1
                                #print k
				
                                if get_chi2_func.get_chisqd(data,g_1,s_1,ant1,ant2,vis_map) - get_chi2_func.get_chisqd(data,g_0,s_0,ant1,ant2,vis_map) >0.0:
                                   
					print "Reach minimum point"
                                        break 

				else:
                                   N +=1
                                   #print get_chi2_func.get_chisqd(data,g_0,s_0,ant1,ant2,vis_map)
				
				chi2.append(get_chi2_func.get_chisqd(data,g_0,s_0,ant1,ant2,vis_map))
				g_0=g_1
				s_0 =s_1
     
     			else:
			
                     		g_1 = g_0 + 0.5*delta_X[s_0.size:len(delta_X)]
     		     		s_1 = s_0 + 0.5*delta_X[0:s_0.size]
                     		Gains[k] = g_1
                     		Sky[k] = s_1
                     		
                  	
                                #print k
                                if get_chi2_func.get_chisqd(data,g_1,s_0,ant1,ant2,vis_map) - get_chi2_func.get_chisqd(data,g_0,s_0,ant1,ant2,vis_map) >0.0:  
                                        print "Reached minimum chi squared"
                                        break
					

				else:
                                    N +=1
                                    #print get_chi2_func.get_chisqd(data,g_0,s_0,ant1,ant2,vis_map)
			 
				    chi2.append(get_chi2_func.get_chisqd(data,g_0,s_0,ant1,ant2,vis_map))
				
                        	g_0= g_1
		        	s_0 =s_1
  
        print "Lincal best fit Performance"           
	print "chi squared value", chi2[-1]
        print "Number of iterations",N
	return [Gains,Sky,chi2,N]            
     
			    




            		
            
             


    
def B_matrix_lua(n_ants,ant1,ant2,vis_map,gain_param,sky_param,q,n_unique): 
        A = np.zeros((q.size, 2.0*n_ants.size + n_unique),dtype ="complex")
        for vis_ in range(q.size):
            A[vis_][vis_map[vis_]]          =  np.conj(gain_param[ant1[vis_]])*gain_param[ant2[vis_]]
            A[vis_][n_unique +ant1[vis_]]   = np.conj(gain_param[ant1[vis_]])*gain_param[ant2[vis_]]*sky_param[vis_map[vis_]]
            
            A[vis_][n_unique + ant2[vis_]]  = np.conj(gain_param[ant1[vis_]])*gain_param[ant2[vis_]]*sky_param[vis_map[vis_]]
            A[vis_][n_unique + n_ants +ant1[vis_]]   = -1j*np.conj(gain_param[ant1[vis_]])*gain_param[ant2[vis_]]*sky_param[vis_map[vis_]]
            A[vis_][n_unique + n_ants +ant2[vis_]]   = 1j*np.conj(gain_param[ant1[vis_]])*gain_param[ant2[vis_]]*sky_param[vis_map[vis_]]
            
            
           
        return A
    
    
    
def lincal_lua(data,eta_0,phi_0,s_0,vis_u_map,ant1,ant2,n_ants):
    n_unique = s_0.size
    gain_0 = np.exp(eta_0)*(np.cos(phi_0)+1j*np.sin(phi_0))
    #gain_0 = gain_0/np.mean(gain_0)
    model =  np.conj(gain_0[ant1])*gain_0[ant2]*s_0[vis_u_map]
    
    B = B_matrix(n_ants,ant1,ant2,vis_u_map,gain_0,s_0,data,n_unique)
    Curvature = B.T.dot(B)
    
    delta_x = np.linalg.pinv(Curvature).dot(B.T).dot(data-model)
    s_1 = s_0 + delta_x[0:s_0.size]
    eta_1 = eta_0 + delta_x[s_0.size: s_0.size+ n_ants.size]
    phi_1 = phi_0 + delta_x[s_0.size +n_ants.size:len(delta_x)]
    
    return np.array([eta_1,phi_1,s_1,delta_x])
    
      			
