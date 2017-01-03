#Newton multavaraite method
import numpy as np, math
import matplotlib.pyplot as plt
import get_chi2_func


def get_fits_param(n_steps,tol,data,gains,sky,ant1,ant2,vis_map):
    """ Netown's Multivaraite Method"""
    p=[]
    for step in range(n_steps):
              param_old= np.concatenate((gains,sky))
              #print param_old[0]
              
              grad = get_chi2_func.get_grad_chisqd(data,param_old[0:gains.size],param_old[gains.size:len(param_old)],ant1,ant2,vis_map)
              curv = get_chi2_func.get_curv_chisqd(data,param_old[0:gains.size],param_old[gains.size:len(param_old)],ant1,ant2,vis_map)
              param_new = param_old - np.linalg.pinv(curv).dot(grad)
              if np.linalg.norm(param_new-param_old) <= tol:
                  p.append(param_new)
                  #print param_new[0], 'good'
                  break
              else:
                  param_old = param_new
                  #print param_new[0], 'bad'
   
          
    return p   
 

