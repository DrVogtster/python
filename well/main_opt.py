
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import linalg
from kron import *


def scaled_trun_func(input,a,b,mean,std,scale):
    myclip_a =a
    myclip_b = b
    my_mean = mean
    my_std = std

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    return scale*truncnorm.pdf(input, a, b, loc = my_mean, scale = my_std)
def plot_trunc(a,b,mean,std,scale):
    x_range = np.linspace(a,b,1000)
    output=[]
    for i in range(0,len(x_range)):
        output.append(scaled_trun_func(x_range[i],a,b,mean,std,scale))
    plt.plot(x_range,output)
    plt.savefig("trun.pdf")


#plot_trunc(0,3,1.5,1,math.pi)

def state_solver(v1,v2,a,b,mean,std,dim,,a,b,mean,std,dim,T_max,nt,g_u,g_v)
    (S,K) = produce_real_and_imag_ham_funcs(v1,v2,amp_list,dim,a,b,mean,std,dim):
    (sol_history,final_sol) = stromer_verlet(S,K,T_max,nt,g_u,g_v,dim)
    return (sol,final_sol)
def ham_helper():
    sigma_x,sigma_y,sigma_z = generate_sigmas_xyz()
    S12=s_one(1,2,sigma_x,sigma_y,sigma_z)
    S23 = s_one(2,3,sigma_x,sigma_y,sigma_z)


    return (S12.real,S12.imag,S23.real,S23.imag)



def produce_real_part_func(t,v1,v2,amp_list, s12r, s23r,a,b,mean,std,dim)
    
    m,n = s12r.shape

    s12_part = zeros((m,m))
    s23_part = zeros((m,m))
    for i in range(0,len(v1)):
        s12_part = s12_part -v1[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s12r
        s23_part = s23_part -v2[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s23r
    return s12_part+s23_part

def produce_imag_part_func(t,v1,v2,amp_list, s12i, s23i,a,b,mean,std,dim)
    
    m,n = s12i.shape

    s12_part = zeros((m,m))
    s23_part = zeros((m,m))
    for i in range(0,len(v1)):
        s12_part = s12_part -v1[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s12i
        s23_part = s23_part -v2[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s23i
    return s12_part+s23_part
    

    
        



   

def produce_real_and_imag_ham_funcs(v1,v2,amp_list,dim,a,b,mean,std,dim):
    s12r,s12i,s23r,s23i = ham_helper()
    S = lambda t: produce_real_part_func(t,v1,v2,amp_list,s12r,s23r,a,b,mean,std,dim)
    K = lambda t: produce_imag_part_func(t,v1,v2,amp_list,s12i,s23i,a,b,mean,std,dim)

    return (S,K)




def stromer_verlet(S,K,T_max,nt,g_u,g_v,dim):
    t_list,dt= np.linspace(0,T_max,nt,retstep=True)
    
    solution_list =np.zeros((dim,dim,len(t_list)), dtype=complex)
    solution_list[:,:,0] = gu-1j*gv 
    for i in range(1,len(t_list)):
        u_n = solution_list[:,:,i-1].real
        v_n = -solution_list[:,:,i-1].imag
        tn_1=t_list[i]
        tn_p5 = t_list[i] - (dt/2.0)
        tn = t_list[i-1]
        
        Sn = S(tn)
        Sn5 = S(tn_p5)
        Sn1 = S(tn_1)
        
        Kn = K(tn)
        Kn5 = K(tn_p5)
        Kn1 = K(tn_1)

        U_n1 = u_n
        V_n1 = linalg.solve(np.eye(dim) - Sn5, v_n +(dt/2.0)*Kn5*U_n1)
        kap_n1 = Sn*U_n1 - Kn*V_n1
        l_n1 = Kn5*U_n1 + Sn5*V_n1
        V_n2 = v_n +(h/2.0)*l_n1
        U_n2 = linalg.solve(np.eye(dim) - Sn1, u_n +(dt/2.0)*(kap_n1 - Kn1*V_n2))

        kap_n2 = Sn1*U_n2 - Kn1*V_n2
        l_n2 = Kn5*U_n2 + Sn5*V_n2

        u_np1 = u_n + (dt/2.0)*(kap_n1+kap_n2)
        v_np1 = v_n + (dt/2.0)*(l_n1+l_n2)
        solution_list[:,:,i] = u_np1-1j*v_np1 
    return (solution_list,solution_list[:,:,-1])

