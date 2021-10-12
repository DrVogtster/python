
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import linalg
from kron import *
accum_list=[]
mydic={}

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

def S_test(t):
    n=5
    return math.sin(t)*np.eye(n)

def K_test(t):
    n=5
    return math.cos(t)*np.eye(n)

def f_u_s_test(t,h):
    n=5
    u=np.eye(n)*math.cos(t)
    v = np.eye(n)*math.sin(t)
    return -np.eye(n)*math.sin(t) - (S_test(t)*u)/h +  (K_test(t)*v)/h

def f_v_s_test(t,h):
    n=5
    u=np.eye(n)*math.cos(t)
    v = np.eye(n)*math.sin(t)
    return np.eye(n)*math.cos(t) - (K_test(t)*u)/h -  (S_test(t)*v)/h

def true_u(t):
    n=5
    return np.eye(n)*math.cos(t)
def true_v(t):
    n=5
    return -np.eye(n)*math.sin(t)

def creat_matrix_from_vector(vector,p,m,n):
    matrix_list=[]
    for i in range(0,p):
        v_i  = vector[i*m*n:(i+1)*m*n]
        v_i_mat = np.reshape(v_i,(m,n))
        matrix_list.append(v_i_mat)
    return matrix_list  

def convergence_test():

    convergence_list_re=[]
    maxerror_list_re=[]

    convergence_list_im=[]
    maxerror_list_im=[]

    # n_list =[100000,200000,400000,800000,1600000]
    n_dim =5
    n_list=[]
    T_max = 10.0
    for i in range(5,17):
        n_list.append(2**i)
    h_plank=10.0
    S=S_test
    K=K_test
    for i in range(0,len(n_list)):
        (sol_list,final_sol) =stromer_verlet(S,K,T_max,n_list[i],np.eye(n_dim),np.zeros((n_dim,n_dim)),n_dim,h_plank,f_u_s_test,f_v_s_test)
        final_sol_re = final_sol.real
        final_sol_im = final_sol.imag
        max_re = np.linalg.norm(true_u(T_max) - final_sol_re,np.inf)
        max_im = np.linalg.norm(true_v(T_max) - final_sol_im,np.inf)
        maxerror_list_re.append(max_re)
        maxerror_list_im.append(max_im)
        

    for i in range(1,len(n_list)-1):
            
        convergence_list_re.append(math.log(maxerror_list_re[i]/maxerror_list_re[i+1])/math.log(2.0))
        convergence_list_im.append(math.log(maxerror_list_im[i]/maxerror_list_im[i+1])/math.log(2.0))
    
    print("Error re:" +str(maxerror_list_re))
    print("Error im:" +str(maxerror_list_im))
    print("Convergence re:" +str(convergence_list_re))
    print("Convergence im:" +str(convergence_list_im))
        




def state_solver(v1,v2,a,b,mean,std,dim,T_max,nt,g_u,g_v):
    (S,K) = produce_real_and_imag_ham_funcs(v1,v2,amp_list,dim,a,b,mean,std,dim)
    (sol_history,final_sol) = stromer_verlet(S,K,T_max,nt,g_u,g_v,dim)
    return (sol,final_sol)
def ham_helper():
    sigma_x,sigma_y,sigma_z = generate_sigmas_xyz()
    S12=s_one(1,2,sigma_x,sigma_y,sigma_z)
    S23 = s_one(2,3,sigma_x,sigma_y,sigma_z)


    return (S12.real,S12.imag,S23.real,S23.imag)

def ham_helper_general(ne):
    S_list_real=[]
    S_list_imag=[]
    sigma_x,sigma_y,sigma_z = generate_sigmas_xyz()
    for i in range(0,ne-1):
        s_iip1 = s_one(i+1,i+2,sigma_x,sigma_y,sigma_z)
        S_list_real.append(s_iip1.real)
        S_list_imag.append(s_iip1.imag)

    return S_list_real,S_list_imag



def produce_real_part_func_general(t,time_step,v_list,amp_list,s_real_list,dim,time_list):
    m,n = s_real_list[0].shape
    total_sum_real = zeros((m,m))
    k_val = int(math.floor(t/time_step))
    a = time_list[k_val][0]
    b = time_list[k_val][1]
    mean = (a+b)/2.0
    std=1.0
    for i in range(0,len(s_list)):
        u_i = 0
        for n in range(0,len(amp_list)):
            u_i = u_i + scaled_trun_func(t,a,b,mean,std,amp_list[n])*v_list[i][k_val,n]
        total_sum_real = total_sum_real - u_i*s_real_list[i]
       
    return total_sum_real

def produce_imag_part_func_general(t,time_step,v_list,amp_list,s_imag_list,dim,time_list):
    m,n = s_imag_list[0].shape
    total_sum_imag = zeros((m,m))
    k_val = int(math.floor(t/time_step))
    a = time_list[k_val][0]
    b = time_list[k_val][1]
    mean = (a+b)/2.0
    std=1.0
    for i in range(0,len(s_list)):
        u_i = 0
        for n in range(0,len(amp_list)):
            u_i = u_i + scaled_trun_func(t,a,b,mean,std,amp_list[n])*v_list[i][k_val,n]
        total_sum_imag = total_sum_real - u_i*s_imag_list[i]
       
    return total_sum_imag



def produce_real_part_func(t,v1,v2,amp_list, s12r, s23r,a,b,mean,std,dim):
    
    m,n = s12r.shape

    s12_part = zeros((m,m))
    s23_part = zeros((m,m))
    for i in range(0,len(v1)):
        s12_part = s12_part -v1[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s12r
        s23_part = s23_part -v2[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s23r
    return s12_part+s23_part

def produce_imag_part_func(t,v1,v2,amp_list, s12i, s23i,a,b,mean,std,dim):
    
    m,n = s12i.shape

    s12_part = zeros((m,m))
    s23_part = zeros((m,m))
    for i in range(0,len(v1)):
        s12_part = s12_part -v1[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s12i
        s23_part = s23_part -v2[i]*scaled_trun_func(t,a,b,mean,std,amp_list[i])*s23i
    return s12_part+s23_part
    

    
        



   

 def produce_real_and_imag_ham_funcs_general(time_step,v_list,amp_list,a,b,mean,std,dim):
    global mydic
    time_step = mydic["dt"]
    v_list = mydic["v"]
    
    S = lambda t:produce_real_part_func_general(t,time_step,v_list,amp_list,s_real_list,dim,time_list)
    K = lambda t:produce_imag_part_func_general(t,time_step,v_list,amp_list,s_real_list,dim,time_list)
 
    return (S,K)

def produce_real_and_imag_ham_funcs(time_step,v_list,amp_list,a,b,mean,std,dim):
    s_real,s_imag = ham_helper_general()
    S = lambda t: produce_real_part_func_general(t,time_step,v_list,amp_list,s_real,a,b,mean,std,dim)
    K = lambda t: produce_imag_part_func_general(t,time_step,v_list,amp_list,s_imag,a,b,mean,std,dim)

    return (S,K)




def stromer_verlet(S_func,K_func,T_max,nt,gu,gv,dim,h_plank,fu_s_func,fv_s_func):
    S = lambda t : S_func(t)/h_plank
    K = lambda t : K_func(t)/h_plank

    t_list,dt= np.linspace(0,T_max,nt,retstep=True)
    fu_s  = lambda t :fu_s_func(t,h_plank)
    fv_s  = lambda t :fv_s_func(t,h_plank)
    solution_list =np.zeros((dim,dim,len(t_list)), dtype=complex)
    solution_list[:,:,0] = gu-1j*gv 
    print("plank constant:" +str(h_plank))
    h=1.0
    for i in range(1,len(t_list)):
        u_n = solution_list[:,:,i-1].real
        v_n = -solution_list[:,:,i-1].imag
        tn_1=t_list[i]
        #print(tn_1)
        tn_p5 = t_list[i] - (dt/2.0)
        tn = t_list[i-1]
        
        Sn = S(tn)
        Sn5 = S(tn_p5)
        Sn1 = S(tn_1)
        
        Kn = K(tn)
        Kn5 = K(tn_p5)
        Kn1 = K(tn_1)

        U_n1 = u_n
        V_n1 = linalg.solve(np.eye(dim) - (dt/2.0*h)*Sn5, v_n +(dt/2.0)*((1.0/h)*Kn5*U_n1+ fv_s(tn_p5)))
        kap_n1 = (1.0/h)*(Sn*U_n1 - Kn*V_n1) + fu_s(tn)
        l_n1 = (1.0/h)*(Kn5*U_n1 + Sn5*V_n1) + fv_s(tn_p5)
        V_n2 = v_n +(dt/2.0)*l_n1
        U_n2 = linalg.solve(np.eye(dim) -(dt/2.0*h)*Sn1, u_n +(dt/2.0)*(kap_n1 - (1.0/h)*Kn1*V_n2 +fu_s(tn_1)))

        kap_n2 = (1.0/h)*(Sn1*U_n2 - Kn1*V_n2) + fu_s(tn_1)
        l_n2 = (1.0/h)*(Kn5*U_n2 + Sn5*V_n2) +fv_s(tn_p5)

        u_np1 = u_n + (dt/2.0)*(kap_n1+kap_n2)
        v_np1 = v_n + (dt/2.0)*(l_n1+l_n2)
        solution_list[:,:,i] = u_np1-1j*v_np1 
    return (solution_list,solution_list[:,:,-1])

def ten_pen(k):
    return 10.0**k

def fid_leak_obj(U,V,basis_list):

    size = len(basis_list)
    U_hat = np.zeros((size,size))
    bad_states=[]
    for i in range(0,size):
        for j in range(0,size):

            U_hat[i,j] = np.transpose(basis_list[i])*U*basis_list[j]
            if(V[i,j]==0.0):
                bad_states.append((basis_list[i],basis_list[j]))
    M = (V.H)*U_hat
    TrM = np.trace(M)
    TrMr = TrM.real
    TrMi = TrM.imag
    mod_squared = TrMr**2 + TrMi**2
    fid = (1.0)/(size*(size+1))*(np.trace(M*M.H) + mod_squared )
    leak = 0 
    # for k in range(0,len(bad_states)):
    #     leak = leak + np.abs(np.transpose(bad_states[k][0]*U*bad_states[k][1]))**2
    return fid+leak
    #pederson


# Python3 implementation of the
# above approach
 
# Function to print the output
def printTheArray(arr, n):
    tem = []
    for i in range(0, n):
        print(arr[i], end = " ")
        tem.append(arr[i])
    accum_list.append(tem)
     
    print()
 
# Function to generate all binary strings
def generateAllBinaryStrings(n, arr, i):
    global accum_list
    if i == n:
        printTheArray(arr, n)
        
        return
     
    # First assign "0" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1)
 
    # And then assign "1" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1)
 
# # Driver Code
# if __name__ == "__main__":
 
#     n = 4
#     arr = [None] * n
 
#     # Print all binary strings
#     generateAllBinaryStrings(n, arr, 0)

def generate_basis_list(n_e):
    n = int(n_e/3)
    arr = [None] * n
    generateAllBinaryStrings(n, arr, 0)

def generate_pen_term(v,weight,Nc,Nt,Np,neighbor_list)

    mysum = 0.0
    for i in range(0,Nt):
        
        for k in range(0,Nc):
            this_sum =0.0
            neighbors = neighbor_list[k]
            for n in neighbors:
                this_sum = this_sum + np.sum(v[n][i,:])
                
            this_sum = max(0,this_sum-1)
        mysum = mysum + weight*this_sum
    return mysum


            
                

def GA_helper(solution, solution_idx):
    global mydic
    v=solution
    Nc=mydic["Nc"] 
    Np=mydic["Np"] 
    Nt=mydic["Nt"] 
    basis = mydic["basis"]
    amp_list = mydic["amp"]
    gate=mydic["gate"] 
    nl  = mydic["nl"]
    v_list= creat_matrix_from_vector(v,Nc,Nt,Np)

    pen_term = generate_pen_term(v,1000,Nc,Nt,Np,nl)

    
    pass



def GA_penalty():

    global mydic

    Nc=mydic["Nc"] 
    Np=mydic["Np"] 
    Nt=mydic["Nt"] 

    
    
    ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=2,
                       sol_per_pop=10,
                       num_genes=Nc*Np*Nt,
                       fitness_func=GA_helper,gene_space=[0, 1],save_best_solutions=True)
    ga.run()
    a=ga_instance.best_solutions
    b=ga_instance.best_solutions_fitness
    my_list=[]
    for i in range(0,len(a)):
        #print(b[i],a[i])
        my_list.append((b[i],a[i]))

    my_list.sort(key=lambda tup: tup[0]) 
    best_obj = my_list[0][0]
    counter=0
    best_sols=[]
    still_collecting=True
    while(still_collecting):
        if(best_obj == my_list[counter][0]):
            best_sols.append(my_list[counter][1])
        else:
            still_collecting=False
        counter+=1
    return (best_obj,best_sols)


def GA_routine(number_e,gate,T,dt,amp_list):
    global accum_list
    global mydic
    mydic={}
    Nt = int(T/dt)
    Np = len(amp_list)
    Nc = number_e-1
    pen_fun = lambda k: ten_pen(k)
    sigma_x,sigma_y,sigma_z=generate_sigmas_xyz()
    real_list=[]
    imag_list=[]
    time_list=[]
    for i in range(0,Nt):
        time_list.append(i*dt,(i+1)*dt)
    for i in range(1,number_e):
        s_iip1 = s_one(i,i+1,sigma_x,sigma_y,sigma_z,number_e)
        real_list.append(s_iip1.real)
        imag_list.append(s_iip1.imag)

    neighbor_list=[]
    for i in range(1,number_e+1):
        temp=[]
        lef = i-1
        cen = i 
        right = i+1
        if(1 >= lef and lef <=number_e ):
            temp.append(lef)
        if(1 >= cen and cen <=number_e ):
            temp.append(cen)
        if(1 >= right and right <=number_e ):
            temp.append(right)

        neighbor_list.append(temp)
    
    mydic["nl"] = neighbor_list


    x=fong_gen_single()
    accum_list=[]
    arr = [None] * number_e
    generateAllBinaryStrings(number_e, arr, 0)
    gamma=1
    sigma=0
    e0,e1=dfs_0_1(gamma,sigma,x)
    basis_list=ket_gen_dfs(e0,e1,number_e,accum_list)
    mydic["basis"] = basis_list
    mydic["amp"] = amp_list
    mydic["gate"] = gate
    mydic["rl"] = real_list
    mydic["il"] = imag_list
    mydic["Nc"] = Nc
    mydic["Np"] = Np
    mydic["Nt"] = Nt 
    mydic["T"] = T 
    mydic["dt"] = dt 

    GA_penalty()



T=500
dt = 10
amp_list=[math.pi/2.0,math.pi,(3.0/4.0)*math.pi]

arr = [None] * 2
generateAllBinaryStrings(2, arr, 0)
print(accum_list)
#convergence_test()
# x=creat_matrix_from_vector(np.asarray(list(range(1,13))),3,2,2)
# print(x)
#main
