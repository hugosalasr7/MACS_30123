from numba.pycc import CC
import numpy as np

cc = CC('pset1_30123')
@cc.export('sim_loop_jit_vrho', 
           'f4, i4, i4, f8[:,:], f8[:,:], f4, f4, f8[:]')

def sim_loop_jit_vrho(z_0_, S_, T_, eps_mat_, z_mat_, 
                      mu_, sigma_, grid_rho_ccore_):
  
  avg_days_b4_sick = [] #Empty list for avg sick days
  for crho in grid_rho_ccore_:
    days_b4_sick = []
    for s_ind in range(S_):
      z_tm1 = z_0_    
      for t_ind in range(T_):
          e_t = eps_mat_[t_ind, s_ind]
          z_t = crho * z_tm1 + (1 - crho) * mu_ + e_t
          z_mat_[t_ind, s_ind] = z_t
          if z_t < 0 or t_ind == (T_ - 1):
            days_b4_sick.append(t_ind)
            break
          z_tm1 = z_t
    
    avg_days_b4_sick.append( np.mean(np.asarray(days_b4_sick)) )

  return avg_days_b4_sick

cc.compile()