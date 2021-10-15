from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
from kron import *
from gates import *
import pygad
from matplotlib import cm
from matplotlib import colors
from scoop import futures
import multiprocessing
import os

from deap import base
from deap import creator
from deap import tools
import random, numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

accum_list = []
mydic = {}
temp = None

diconedtothreed={}
dicthreedtooned={}

def build_dics(Nc,Nt,Np):
    global diconedtothreed
    global dicthreedtooned
    diconedtothreed={}
    dicthreedtooned={}
    counter=0
    for i in range(0,Nc):
        for j in range(0,Nt):
            for k in range(0,Np):
                diconedtothreed[(counter,)] = (i,j,k)
                dicthreedtooned[(i,j,k)] = counter


def scaled_trun_func(input, a, b, mean, std, scale):
    myclip_a = a
    myclip_b = b
    my_mean = mean
    my_std = std

    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    return scale * truncnorm.pdf(input, a, b, loc=my_mean, scale=my_std)


def plot_trunc(a, b, mean, std, scale):
    x_range = np.linspace(a, b, 1000)
    output = []
    for i in range(0, len(x_range)):
        output.append(scaled_trun_func(x_range[i], a, b, mean, std, scale))
    plt.plot(x_range, output)
    plt.savefig("trun.pdf")


# plot_trunc(0,3,1.5,1,math.pi)

def S_test(t):
    n = 5
    return math.sin(t) * np.eye(n)


def K_test(t):
    n = 5
    return math.cos(t) * np.eye(n)


def f_u_s_test(t, h):
    n = 5
    u = np.eye(n) * math.cos(t)
    v = np.eye(n) * math.sin(t)
    return -np.eye(n) * math.sin(t) - (S_test(t) * u) / h + (K_test(t) * v) / h


def f_v_s_test(t, h):
    n = 5
    u = np.eye(n) * math.cos(t)
    v = np.eye(n) * math.sin(t)
    return np.eye(n) * math.cos(t) - (K_test(t) * u) / h - (S_test(t) * v) / h


def true_u(t):
    n = 5
    return np.eye(n) * math.cos(t)


def true_v(t):
    n = 5
    return -np.eye(n) * math.sin(t)


def creat_matrix_from_vector(vector, p, m, n):
    matrix_list = []
    for i in range(0, p):
        v_i = vector[i * m * n:(i + 1) * m * n]
        v_i_mat = np.reshape(v_i, (m, n))
        matrix_list.append(v_i_mat)
    return matrix_list


def convergence_test():
    convergence_list_re = []
    maxerror_list_re = []

    convergence_list_im = []
    maxerror_list_im = []

    # n_list =[100000,200000,400000,800000,1600000]
    n_dim = 5
    n_list = []
    T_max = 10.0
    for i in range(5, 17):
        n_list.append(2 ** i)
    h_plank = 10.0
    S = S_test
    K = K_test
    for i in range(0, len(n_list)):
        (sol_list, final_sol) = stromer_verlet(S, K, T_max, n_list[i], np.eye(n_dim), np.zeros((n_dim, n_dim)), n_dim,
                                               h_plank, f_u_s_test, f_v_s_test)
        final_sol_re = final_sol.real
        final_sol_im = final_sol.imag
        max_re = np.linalg.norm(true_u(T_max) - final_sol_re, np.inf)
        max_im = np.linalg.norm(true_v(T_max) - final_sol_im, np.inf)
        maxerror_list_re.append(max_re)
        maxerror_list_im.append(max_im)

    for i in range(1, len(n_list) - 1):
        convergence_list_re.append(math.log(maxerror_list_re[i] / maxerror_list_re[i + 1]) / math.log(2.0))
        convergence_list_im.append(math.log(maxerror_list_im[i] / maxerror_list_im[i + 1]) / math.log(2.0))

    print("Error re:" + str(maxerror_list_re))
    print("Error im:" + str(maxerror_list_im))
    print("Convergence re:" + str(convergence_list_re))
    print("Convergence im:" + str(convergence_list_im))


def state_solver(v1, v2, a, b, mean, std, dim, T_max, nt, g_u, g_v):
    (S, K) = produce_real_and_imag_ham_funcs(v1, v2, amp_list, dim, a, b, mean, std, dim)
    (sol_history, final_sol) = stromer_verlet(S, K, T_max, nt, g_u, g_v, dim)
    return (sol, final_sol)


def ham_helper():
    sigma_x, sigma_y, sigma_z = generate_sigmas_xyz()
    S12 = s_one(1, 2, sigma_x, sigma_y, sigma_z)
    S23 = s_one(2, 3, sigma_x, sigma_y, sigma_z)

    return (S12.real, S12.imag, S23.real, S23.imag)


def ham_helper_general(ne):
    S_list_real = []
    S_list_imag = []
    sigma_x, sigma_y, sigma_z = generate_sigmas_xyz()
    for i in range(0, ne - 1):
        s_iip1 = s_one(i + 1, i + 2, sigma_x, sigma_y, sigma_z)
        S_list_real.append(s_iip1.real)
        S_list_imag.append(s_iip1.imag)

    return S_list_real, S_list_imag


def produce_real_part_func_general(t, time_step, v_list, amp_list, s_real_list, dim, time_list):
    global mydic
    m, n = s_real_list[0].shape
    total_sum_real = np.zeros((m, m))
    k_val = int(math.floor(t / time_step))
 
    if(k_val==int(mydic["T"]/time_step)):
        k_val = k_val-1
 
    a = time_list[k_val][0]
    b = time_list[k_val][1]
    mean = (a + b) / 2.0
    std = 1.0
    for i in range(0, len(s_real_list)):
        u_i = 0
        for n in range(0, len(amp_list)):
            u_i = u_i + scaled_trun_func(t, a, b, mean, std, amp_list[n]) * v_list[i][k_val, n]
        total_sum_real = total_sum_real - u_i * s_real_list[i]

    return total_sum_real


def produce_imag_part_func_general(t, time_step, v_list, amp_list, s_imag_list, dim, time_list):
    global mydic
    m, n = s_imag_list[0].shape
    total_sum_imag = np.zeros((m, m))
    k_val = int(math.floor(t / time_step))
    if(k_val==int(mydic["T"]/time_step)):
        k_val = k_val-1
    a = time_list[k_val][0]
    b = time_list[k_val][1]
    mean = (a + b) / 2.0
    std = 1.0
    for i in range(0, len(s_imag_list)):
        u_i = 0
        for n in range(0, len(amp_list)):
            u_i = u_i + scaled_trun_func(t, a, b, mean, std, amp_list[n]) * v_list[i][k_val, n]
        total_sum_imag = total_sum_imag - u_i * s_imag_list[i]

    return total_sum_imag


def produce_real_part_func(t, v1, v2, amp_list, s12r, s23r, a, b, mean, std, dim):
    m, n = s12r.shape

    s12_part = zeros((m, m))
    s23_part = zeros((m, m))
    for i in range(0, len(v1)):
        s12_part = s12_part - v1[i] * scaled_trun_func(t, a, b, mean, std, amp_list[i]) * s12r
        s23_part = s23_part - v2[i] * scaled_trun_func(t, a, b, mean, std, amp_list[i]) * s23r
    return s12_part + s23_part


def produce_imag_part_func(t, v1, v2, amp_list, s12i, s23i, a, b, mean, std, dim):
    m, n = s12i.shape

    s12_part = zeros((m, m))
    s23_part = zeros((m, m))
    for i in range(0, len(v1)):
        s12_part = s12_part - v1[i] * scaled_trun_func(t, a, b, mean, std, amp_list[i]) * s12i
        s23_part = s23_part - v2[i] * scaled_trun_func(t, a, b, mean, std, amp_list[i]) * s23i
    return s12_part + s23_part


def produce_real_and_imag_ham_funcs_general(time_step, v_list, amp_list, dim):
    global mydic
    time_step = mydic["dt"]
    v_list = mydic["v"]
    amp_list = mydic["amp"]
    s_real_list = mydic["rl"]
    s_imag_list = mydic["il"]
    time_step = mydic["dt"]
    time_list = mydic["tl"]

    S = lambda t: produce_real_part_func_general(t, time_step, v_list, amp_list, s_real_list, dim, time_list)
    K = lambda t: produce_imag_part_func_general(t, time_step, v_list, amp_list, s_imag_list, dim, time_list)

    return (S, K)


def produce_real_and_imag_ham_funcs(time_step, v_list, amp_list, a, b, mean, std, dim):
    s_real, s_imag = ham_helper_general()
    S = lambda t: produce_real_part_func_general(t, time_step, v_list, amp_list, s_real, a, b, mean, std, dim)
    K = lambda t: produce_imag_part_func_general(t, time_step, v_list, amp_list, s_imag, a, b, mean, std, dim)

    return (S, K)



def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  

def stromer_verlet(S_func, K_func, T_max, nt, gu, gv, dim, h_plank, fu_s_func, fv_s_func):
    S = lambda t: S_func(t) / h_plank
    K = lambda t: K_func(t) / h_plank

    t_list, dt = np.linspace(0, T_max, nt, retstep=True)
    # fu_s = lambda t: fu_s_func(t, h_plank)
    # fv_s = lambda t: fv_s_func(t, h_plank)
    fu_s = fu_s_func
    fv_s = fv_s_func
    solution_list = np.zeros((dim, dim, len(t_list)), dtype=complex)
    solution_list[:, :, 0] = gu - 1j * gv
    print("plank constant:" + str(h_plank))
    h = 1.0
    for i in range(1, len(t_list)):
        u_n = solution_list[:, :, i - 1].real
        v_n = -solution_list[:, :, i - 1].imag
        tn_1 = t_list[i]
        # print(tn_1)
        tn_p5 = t_list[i] - (dt / 2.0)
        tn = t_list[i - 1]
        #print(tn)
        Sn = S(tn)
        matprint(Sn)
        Sn5 = S(tn_p5)
        Sn1 = S(tn_1)

        Kn = K(tn)
        
        Kn5 = K(tn_p5)
        Kn1 = K(tn_1)

        U_n1 = u_n
        V_n1 = linalg.solve(np.eye(dim) - (dt / 2.0 * h) * Sn5,
                            v_n + (dt / 2.0) * ((1.0 / h) * Kn5 * U_n1 + fv_s(tn_p5)))
        kap_n1 = (1.0 / h) * (Sn * U_n1 - Kn * V_n1) + fu_s(tn)
        l_n1 = (1.0 / h) * (Kn5 * U_n1 + Sn5 * V_n1) + fv_s(tn_p5)
        V_n2 = v_n + (dt / 2.0) * l_n1
        U_n2 = linalg.solve(np.eye(dim) - (dt / 2.0 * h) * Sn1,
                            u_n + (dt / 2.0) * (kap_n1 - (1.0 / h) * Kn1 * V_n2 + fu_s(tn_1)))

        kap_n2 = (1.0 / h) * (Sn1 * U_n2 - Kn1 * V_n2) + fu_s(tn_1)
        l_n2 = (1.0 / h) * (Kn5 * U_n2 + Sn5 * V_n2) + fv_s(tn_p5)

        u_np1 = u_n + (dt / 2.0) * (kap_n1 + kap_n2)
        v_np1 = v_n + (dt / 2.0) * (l_n1 + l_n2)
        solution_list[:, :, i] = u_np1 - 1j * v_np1
    print(solution_list[:, :, -1])
    return (solution_list, solution_list[:, :, -1])


def ten_pen(k):
    return 10.0 ** k


def produce_state_constant_pulse(amp_list,plank,Nc,Nt,dt,v_list,H_list,dim):
    global mydic
    sol = np.eye(dim)
    Np = mydic["Np"]
    for k in range(0,Nt):
        H_temp = np.zeros((dim,dim))
        for i in range(0,Nc):
            for p in range(0,Np):
            
                H_temp = H_temp + v_list[i][k,p]*H_list[i]*((amp_list[p])/dt)
        cur_sol = scipy.linalg.expm(-1j*dt*(H_temp)/plank)
        sol =  np.matmul(cur_sol,sol)
    return sol





def fid_leak_obj(U, V, basis_list):
    size = len(basis_list)
    U_hat = np.zeros((size, size),dtype = 'complex_')
    bad_states = []
    for i in range(0, size):
        for j in range(0, size):
            # print(basis_list[i])
            # print(basis_list[j])
            U_hat[i, j] = np.dot(basis_list[i], U @ basis_list[j])
            # print(U_hat[i,j])
            #U_hat[i,j] = (basis_list[i].T.dot(U)*basis_list[j].T).sum(axis=1)
            if (V[i, j] == 0.0):
                bad_states.append((basis_list[i], basis_list[j]))
    M = (V.conjugate()) * U_hat
    TrM = np.trace(M)
    TrMr = TrM.real
    TrMi = TrM.imag
    mod_squared = TrMr ** 2 + TrMi ** 2
    fid = (1.0) / (size * (size + 1)) * (np.trace(M * M.conjugate()) + mod_squared)
    leak = 0
    # for k in range(0,len(bad_states)):
    #     leak = leak + np.abs(np.transpose(bad_states[k][0]*U*bad_states[k][1]))**2
    return fid + leak
    # pederson


# Python3 implementation of the
# above approach

# Function to print the output
def printTheArray(arr, n):
    tem = []
    for i in range(0, n):
        print(arr[i], end=" ")
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
    n = int(n_e / 3)
    arr = [None] * n
    generateAllBinaryStrings(n, arr, 0)



def fid_routine_constant(v,plank):
    dim = mydic["dim"]
    Nc = mydic["Nc"]
    Np = mydic["Np"]
    Nt = mydic["Nt"]
    basis = mydic["basis"]
    mydic["v"] = v
    amp_list = mydic["amp"]
    gate = mydic["gate"]
    nl = mydic["nl"]
    plank = mydic["p"]
    T_max = mydic["T"]
    dt = mydic["dt"]
    v_list=v
    H_list = mydic["H"]
    U = produce_state_constant_pulse(amp_list,plank,Nc,Nt,dt,v_list,H_list,dim)
    obj = fid_leak_obj(U, gate, basis)
    return obj


def fid_routine_time(v, plank):
    dim = mydic["dim"]
    Nc = mydic["Nc"]
    Np = mydic["Np"]
    Nt = mydic["Nt"]
    basis = mydic["basis"]
    mydic["v"] = v
    amp_list = mydic["amp"]
    gate = mydic["gate"]
    nl = mydic["nl"]
    plank = mydic["p"]
    T_max = mydic["T"]
    time_step = mydic["dt"]
    nt = int(T/(10**(-3)))
    v_list =v
    (S_func, K_func) = produce_real_and_imag_ham_funcs_general(time_step, v_list, amp_list,dim)
    gu = np.eye(dim)
    gv = np.zeros((dim,dim))
    U,U_final = stromer_verlet(S_func, K_func, T_max, nt, gu, gv, dim, plank, lambda t: np.zeros((dim, dim)),
                   lambda t: np.zeros((dim, dim)))
    obj = fid_leak_obj(U_final, gate, basis)
    return obj


def generate_pen_term(v, weight, Nc, Nt, Np, neighbor_list):
    mysum = 0.0
    for i in range(0, Nt):

        for k in range(0, Nc):
            this_sum = 0.0
            neighbors = neighbor_list[k]
            for n in neighbors:
                this_sum = this_sum + np.sum(v[n][i, :])

            this_sum = max(0, this_sum - 1)
            mysum = mysum + weight * this_sum
    return mysum



# def GA_helper_constant_deap(solution, solution_idx):
#     global mydic
#     v = solution
#     Nc = mydic["Nc"]
#     Np = mydic["Np"]
#     Nt = mydic["Nt"]
#     basis = mydic["basis"]
#     amp_list = mydic["amp"]
#     gate = mydic["gate"]
#     nl = mydic["nl"]
#     plank = mydic["p"]
#     v_list = creat_matrix_from_vector(v, Nc, Nt, Np)

#     pen_term = generate_pen_term(v_list, 1000, Nc, Nt, Np, nl)
#     fid = fid_routine_constant(v_list, plank)

#     print((fid,pen_term))
#     return fid


def GA_helper_constant(solution, solution_idx):
    global mydic
    v = solution
    Nc = mydic["Nc"]
    Np = mydic["Np"]
    Nt = mydic["Nt"]
    basis = mydic["basis"]
    amp_list = mydic["amp"]
    gate = mydic["gate"]
    nl = mydic["nl"]
    plank = mydic["p"]
    v_list = creat_matrix_from_vector(v, Nc, Nt, Np)

    pen_term = generate_pen_term(v_list, 1000, Nc, Nt, Np, nl)
    fid = fid_routine_constant(v_list, plank)

    print((fid,pen_term))
    return (fid - pen_term)

def GA_penalty_constant_deap():
    global mydic
    global temp
    Nc = mydic["Nc"]
    Np = mydic["Np"]
    Nt = mydic["Nt"]
    no_of_generations = 1000 # decide, iterations

    # decide, population size or no of individuals or solutions being considered in each generation
    population_size = 1000

    # chromosome (also called individual) in DEAP
    # length of the individual or chrosome should be divisible by no. of variables 
    # is a series of 0s and 1s in Genetic Algorithm

    # here, one individual may be 
    # [1,0,1,1,1,0,......,0,1,1,0,0,0,0] of length 100
    # each element is called a gene or allele or simply a bit
    # Individual in bit form is called a Genotype and is decoded to attain the Phenotype i.e. the 
    size_of_individual = Nc*Nt*Np

    # above, higher the better but uses higher resources

    # we are using bit flip as mutation,
    # probability that a gene or allele will mutate or flip, 
    # generally kept low, high means more random jumps or deviation from parents, which is generally not desired
    probability_of_mutation = 0.05 

    # no. of participants in Tournament selection
    # to implement strategy to select parents which will mate to produce offspring
    tournSel_k = 10 

    # no, of variables which will vary,here we have x and y
    # this is so because both variables are of same length and are represented by one individual
    # here first 50 bits/genes represent x and the rest 50 represnt y.
    no_of_variables = Nc*Nt*Np

    # bounds = [(-6,6),(-6,6)] # one tuple or pair of lower bound and upper bound for each variable
    # # same for both variables in our problem 
    bounds =[]
    for i in range(0,no_of_variables):
        bounds.append((0,1))
    # CXPB  is the probability with which two individuals
    #       are crossed or mated
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2


    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # an Individual is a list with one more attribute called fitness
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("map", futures.map)

    # Attribute generator, generation function 
    # toolbox.attr_bool(), when called, will draw a random integer between 0 and 1
    # it is equivalent to random.randint(0,1)
    toolbox.register("attr_bool", random.randint, 0, 1)

    # here give the no. of bits in an individual i.e. size_of_individual, here 100
    # depends upon decoding strategy, which uses precision
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, size_of_individual) 
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    def check_feasiblity(individual):
        #print(individual)
        global mydic
        Nc = mydic["Nc"]
        Np = mydic["Np"]
        Nt = mydic["Nt"]
        individual = decode_all_x(individual,no_of_variables,bounds)
        v = creat_matrix_from_vector(individual,Nc,Nt,Np)
        nl = mydic["nl"]
        
        mysum = 0.0
        for i in range(0, Nt):

            for k in range(0, Nc):
                this_sum = 0.0
                neighbors = nl[k]
                for n in neighbors:
                    this_sum = this_sum + np.sum(v[n][i, :])

                this_sum =  this_sum - 1
                mysum = mysum +this_sum
        if(mysum<=Nt*Nc):
            return True
        else:
            return False
    def decode_all_x(individual,no_of_variables,bounds):
        '''
        returns list of decoded x in same order as we have in binary format in chromosome
        bounds should have upper and lower limit for each variable in same order as we have in binary format in chromosome 
        '''
        # print(individual)
        len_chromosome = len(individual)
        # print(len_chromosome)
        # print(no_of_variables)
        len_chromosome_one_var = int(len_chromosome/no_of_variables)
        bound_index = 0
        x = []
        
        # one loop each for x(first 50 bits of individual) and y(next 50 of individual)
        for i in range(0,len_chromosome,len_chromosome_one_var):
            # converts binary to decimial using 2**place_value
            chromosome_string = ''.join((str(xi) for xi in  individual[i:i+len_chromosome_one_var]))
            binary_to_decimal = int(chromosome_string,2)
            
            # this formula for decoding gives us two benefits
            # we are able to implement lower and upper bounds for each variable
            # gives us flexibility to choose chromosome of any length, 
            #      more the no. of bits for a variable, more precise the decoded value can be
            lb = bounds[bound_index][0]
            ub = bounds[bound_index][1]
            precision = (ub-lb)/((2**len_chromosome_one_var)-1)
            decoded = (binary_to_decimal*precision)+lb
            x.append(decoded)
            bound_index +=1
        
        # returns a list of solutions in phenotype o, here [x,y]
        return x
    def fid_routine_constant_deap(individual):
        individual = decode_all_x(individual,no_of_variables,bounds)
        dim = mydic["dim"]
        Nc = mydic["Nc"]
        Np = mydic["Np"]
        Nt = mydic["Nt"]
        v = creat_matrix_from_vector(individual,Nc,Nt,Np)
        
        basis = mydic["basis"]
        mydic["v"] = v
        amp_list = mydic["amp"]
        gate = mydic["gate"]
        nl = mydic["nl"]
        plank = mydic["p"]
        T_max = mydic["T"]
        dt = mydic["dt"]
        v_list=v
        plank = mydic["p"]
        H_list = mydic["H"]
        U = produce_state_constant_pulse(amp_list,plank,Nc,Nt,dt,v_list,H_list,dim)
        obj = fid_leak_obj(U, gate, basis)
        print("fid" + str(obj))
        return [-obj]

    def penalty_fxn(individual):
        global mydic
        Nc = mydic["Nc"]
        Np = mydic["Np"]
        Nt = mydic["Nt"]
        individual = decode_all_x(individual,no_of_variables,bounds)
        v = creat_matrix_from_vector(individual,Nc,Nt,Np)
        nl = mydic["nl"]
        '''
        Penalty function to be implemented if individual is not feasible or violates constraint
        It is assumed that if the output of this function is added to the objective function fitness values,
        the individual has violated the constraint.
        '''
        mysum = 0.0
        for i in range(0, Nt):

            for k in range(0, Nc):
                this_sum = 0.0
                neighbors = nl[k]
                for n in neighbors:
                    this_sum = this_sum + np.sum(v[n][i, :])

                this_sum =  this_sum - 1
                mysum = mysum +this_sum
        return mysum**2



    def GA_helper_constant_deap(individual):
        global mydic
        individual =xdecode_all_x(individual,no_of_variables,bounds)
        
        Nc = mydic["Nc"]
        Np = mydic["Np"]
        Nt = mydic["Nt"]
        v = creat_matrix_from_vector(individual,Nc,Nt,Np)
        basis = mydic["basis"]
        amp_list = mydic["amp"]
        gate = mydic["gate"]
        nl = mydic["nl"]
        plank = mydic["p"]
        v_list = creat_matrix_from_vector(v, Nc, Nt, Np)

        #pen_term = generate_pen_term(v_list, 1000, Nc, Nt, Np, nl)
       

        print((fid,pen_term))
        return fid

    toolbox.register("evaluate", fid_routine_constant_deap) # privide the objective function here
    #toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 10, penalty_fxn)) # constraint on our objective function

    # registering basic processes using bulit in functions in DEAP
    toolbox.register("mate", tools.cxTwoPoint) # strategy for crossover, this classic two point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=probability_of_mutation) # mutation strategy with probability of mutation
    toolbox.register("select", tools.selTournament, tournsize=tournSel_k)
    # create poppulation as coded in population class
    # no. of individuals can be given as input
    print(multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    toolbox.register("map", pool.map)
    stats = tools.Statistics()

    # registering the functions to which we will pass the list of fitness's of a gneration's offspring
    # to ge the results
    stats.register('Min', np.min)
    stats.register('Max', np.max)
    stats.register('Avg', np.mean)
    stats.register('Std', np.std)

    logbook = tools.Logbook()
    hall_of_fame = tools.HallOfFame(1)
    pop = toolbox.population(n=population_size)

    # The next thing to do is to evaluate our brand new population.

    # use map() from python to give each individual to evaluate and create a list of the result
    fitnesses = list(map(toolbox.evaluate, pop)) 

    # ind has individual and fit has fitness score
    # individual class in deap has fitness.values attribute which is used to store fitness value
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # evolve our population until we reach the number of generations

    # Variable keeping track of the number of generations
    g = 0
    # clearing hall_of_fame object as precaution before every run
    hall_of_fame.clear()

    # Begin the evolution
    while g < no_of_generations:
        # A new generation
        g = g + 1
        
        #The evolution itself will be performed by selecting, mating, and mutating the individuals in our population.
        
        # the first step is to select the next generation.
        # Select the next generation individuals using select defined in toolbox here tournament selection
        # the fitness of populations is decided from the individual.fitness.values[0] attribute
        #      which we assigned earlier to each individual
        # these are best individuals selected from population after selection strategy
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals, this needs to be done to create copy and avoid problem of inplace operations
        # This is of utter importance since the genetic operators in toolbox will modify the provided objects in-place.
        offspring = list(map(toolbox.clone, offspring))
        
        # Next, we will perform both the crossover (mating) and the mutation of the produced children with 
        #        a certain probability of CXPB and MUTPB. 
        # The del statement will invalidate the fitness of the modified offspring as they are no more valid 
        #       as after crossover and mutation, the individual changes
        
        # Apply crossover and mutation on the offspring
        # note, that since we are not cloning, the changes in child1, child2 and mutant are happening inplace in offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values 
                    
                    
        # Evaluate the individuals with an invalid fitness (after we use del to make them invalid)
        # again note, that since we did not use clone, each change happening is happening inplace in offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        
        # To check the performance of the evolution, we will calculate and print the 
        # minimal, maximal, and mean values of the fitnesses of all individuals in our population 
        # as well as their standard deviations.
        # Gather all the fitnesses in one list and print the stats
        #this_gen_fitness = [ind.fitness.values[0] for ind in offspring]
        this_gen_fitness = [] # this list will have fitness value of all the offspring
        for ind in offspring:
            this_gen_fitness.append(ind.fitness.values[0])            
        
        
        #### SHORT METHOD
        
        # will update the HallOfFame object with the best individual 
        #   according to fitness value and weight (while creating base.Fitness class)
        hall_of_fame.update(offspring)
        
        # pass a list of fitnesses 
        # (basically an object on which we want to perform registered functions)
        # will return a dictionary with key = name of registered function and value is return of the registered function
        stats_of_this_gen = stats.compile(this_gen_fitness)
        
        # creating a key with generation number
        stats_of_this_gen['Generation'] = g
        
        # printing for each generation
        print(stats_of_this_gen)
        
        # recording everything in a logbook object
        # logbook is essentially a list of dictionaries
        logbook.append(stats_of_this_gen)
        
        
        # now one generation is over and we have offspring from that generation
        # these offspring wills serve as population for the next generation
        # this is not happening inplace because this is simple python list and not a deap framework syntax
        pop[:] = offspring
    # print the best solution using HallOfFame object
    best_obj = None
    best_sol = None
    
    for best_indi in hall_of_fame:
        # using values to return the value and
        # not a deap.creator.FitnessMin object
        best_obj_val_overall = best_indi.fitness.values[0]
        best_obj = best_obj_val_overall

        print('Minimum value for function: ',best_obj_val_overall)
        print('Optimum Solution: ',decode_all_x(best_indi,no_of_variables,bounds))
        val =decode_all_x(best_indi,no_of_variables,bounds)
        v = creat_matrix_from_vector(val, Nc,Nt,Np)
        best_sol = v
        
        
    # finding the fitness value of the fittest individual of the last generation or 
    # the solution at which the algorithm finally converges
    # we find this from logbook

    # select method will return value of all 'Min' keys in the order they were logged,
    # the last element will be the required fitness value since the last generation was logged last
    best_obj_val_convergence = logbook.select('Min')[-1]

    # plotting Generations vs Min to see convergence for each generation

    plt.figure(figsize=(20, 10))

    # using select method in logbook object to extract the argument/key as list
    plt.plot(logbook.select('Generation'), logbook.select('Min'))

    plt.title("Minimum values of f(x,y) Reached Through Generations",fontsize=20,fontweight='bold')
    plt.xlabel("Generations",fontsize=18,fontweight='bold')
    plt.ylabel("Value of Himmelblau's Function",fontsize=18,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')


    # the red line at lowest value of f(x,y) in the last generation or the solution at which algorithm converged
    plt.axhline(y=best_obj_val_convergence,color='r',linestyle='--')

    # the red line at lowest value of f(x,y)
    plt.axhline(y=best_obj_val_overall,color='r',linestyle='--')


    #
    if best_obj_val_convergence > 2:
        k = 0.8
    elif best_obj_val_convergence > 1:
        k = 0.5
    elif best_obj_val_convergence > 0.5:
        k = 0.3
    elif best_obj_val_convergence > 0.3:
        k = 0.2
    else:
        k = 0.1

    # location of both text in terms of x and y coordinate
    # k is used to create height distance on y axis between the two texts for better readability


    # for best_obj_val_convergence
    xyz1 = (no_of_generations/2.4,best_obj_val_convergence) 
    xyzz1 = (no_of_generations/2.2,best_obj_val_convergence+(k*3)) 

    plt.annotate("At Convergence: %0.5f" % best_obj_val_convergence,xy=xyz1,xytext=xyzz1,
                arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
                fontsize=18,fontweight='bold')

    # for best_obj_val_overall
    xyz2 = (no_of_generations/6,best_obj_val_overall)
    xyzz2 = (no_of_generations/5.4,best_obj_val_overall+(k/0.1))

    plt.annotate("Minimum Overall: %0.5f" % best_obj_val_overall,xy=xyz2,xytext=xyzz2,
                arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
                fontsize=18,fontweight='bold')

    plt.show()
    # a= best_sol
    # a = ga_instance.best_solutions
    # b = ga_instance.best_solutions_fitness
    print(type(best_obj))
    my_list = [(best_obj.real,best_sol)]
    # for i in range(0, len(a)):
    #     # print(b[i],a[i])
    #     my_list.append((b[i], a[i]))

    my_list.sort(key=lambda tup: tup[0])
    best_obj = my_list[0][0]
    counter = 0
    best_sols = []
    # still_collecting = True
    # while (still_collecting):
    #     if (best_obj == my_list[counter][0]):
    #         best_sols.append(my_list[counter][1])
    #     else:
    #         still_collecting = False
    #     counter += 1
    #print(best_sols)
    #print(best_sol[0].shape)
    v_list = best_sol
    
    #print(v_list)
    make_movie(v_list,Nc,Nt,Np,1)
    pool.close()
    return (best_obj, best_sol)




def GA_penalty_constant():
    global mydic
    global temp

    Nc = mydic["Nc"]
    Np = mydic["Np"]
    Nt = mydic["Nt"]

    ga_instance = pygad.GA(num_generations=1000,
                           num_parents_mating=2,
                           sol_per_pop=10,
                           num_genes=Nc * Np * Nt,
                           fitness_func=GA_helper_constant, gene_space=[0, 1], save_best_solutions=True)
    ga_instance.run()
    a = ga_instance.best_solutions
    b = ga_instance.best_solutions_fitness
    my_list = []
    for i in range(0, len(a)):
        # print(b[i],a[i])
        my_list.append((b[i], a[i]))

    my_list.sort(key=lambda tup: tup[0])
    best_obj = my_list[0][0]
    counter = 0
    best_sols = []
    still_collecting = True
    while (still_collecting):
        if (best_obj == my_list[counter][0]):
            best_sols.append(my_list[counter][1])
        else:
            still_collecting = False
        counter += 1
    #print(best_sols)
    v_list = creat_matrix_from_vector(best_sols[0],Nc,Nt,Np)
    temp=v_list
    #print(v_list)
    make_movie(v_list,Nc,Nt,Np,1)
    return (best_obj, best_sols)


def make_movie(v,Nc,Nt,Np,plot_time_each_step):
    duration = Nt*plot_time_each_step


    global temp
    temp=v
    fig, ax = plt.subplots()
    def make_frame(t):
        global temp
        global mydic 
        amp_list=mydic["amp"]
        Nc=mydic["Nc"]
        Np=mydic["Np"]
        #plt.imshow(matrix, cmap = cm.Greys_r)
        #print(t)
        my_t = int(t)
        v_list=temp
        matrix = np.zeros((1, Nc))
        ax = plt.gca()
        ax.clear()
        current_level = []
        for i in range(0,len(v_list)):
            cur_ent = v_list[i][my_t,:]
            current_level.append(cur_ent)
        # print(len(current_level))
        # print(matrix.shape)
        # print(current_level)
        # print(Nc)
        for k in range(0,len(current_level)):
            
                       #print(matrix)
            indx = np.argmax(current_level[k])
            matrix[0, k ] = 100*current_level[k][indx]
            #print(str(current_level[k][indx]*amp_list[indx]))
            ax.text(k,0, str(current_level[k][indx]*amp_list[indx]), va='center', ha='center',color="k", weight='bold')
    
        #print(matrix)
        cmap = colors.ListedColormap(['white', 'red'])
        bounds=[0,90,110]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(matrix, cmap = cmap,norm=norm)
            
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration = duration)
    
    # displaying animation with auto play and looping
    animation.ipython_display(fps = plot_time_each_step, loop = True, autoplay = True)

def GA_routine_constant(number_e, gate, T, dt, amp_list, plank):
    global accum_list
    global mydic
    mydic = {}
    Nt = int(T / dt)
    Np = len(amp_list)
    Nc = number_e - 1
    pen_fun = lambda k: ten_pen(k)
    sigma_x, sigma_y, sigma_z = generate_sigmas_xyz()
    real_list = []
    imag_list = []
    time_list = []
    H_list=[]
    for i in range(0, Nt):
        time_list.append((i * dt, (i + 1) * dt))
    for i in range(1, number_e):
        s_iip1 = s_one(i, i + 1, sigma_x, sigma_y, sigma_z, 1,number_e)
        H_list.append(s_iip1)
    mydic["tl"] = time_list
    neighbor_list = []
    for i in range(1, number_e + 1):
        temp = []
        lef = i - 1
        cen = i
        right = i + 1
        if (1 >= lef and lef <= number_e):
            temp.append(lef)
        if (1 >= cen and cen <= number_e):
            temp.append(cen)
        if (1 >= right and right <= number_e):
            temp.append(right)

        neighbor_list.append(temp)

    mydic["nl"] = neighbor_list

    x = fong_gen_single()
    
    arr = [None] * number_e
    generateAllBinaryStrings(int(number_e/3), arr, 0)
    accum_list = arr
    gamma = 1
    sigma = 0
    e0, e1 = dfs_0_1(gamma, sigma, x)
    basis_list = ket_gen_dfs(e0, e1, int(number_e/3), accum_list)
    print(basis_list[0])
    print(basis_list[1])
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
    mydic["p"] = plank
    mydic["dim"] = 2 ** number_e
    mydic["H"] = H_list

    # e0 = [1, 0]
    # e1 = [0,1]

    # H = math.pi*H_list[0]
    # matprint(H)
    # sol = scipy.linalg.expm(-1j*H)
    # e100 = kron(e1,kron(e0,e0))
    # e010 = kron(e0,kron(e1,e0))
    # print(e100)
    # print(e010)
    # matprint(sol)
    # print(matprint(sol.conjugate()@sol))
    # print("-")
    # print(e100.shape)
  
    # res1 = sol@e100
    # res2 = sol@e010
    # print("---------------")
    # print(res1)
    # print(res2)
    # print("---------------")


    

    #return GA_penalty_constant()
    return GA_penalty_constant_deap()

def GA_routine_time(number_e, gate, T, dt, amp_list, plank):
    global accum_list
    global mydic
    mydic = {}
    Nt = int(T / dt)
    Np = len(amp_list)
    Nc = number_e - 1
    pen_fun = lambda k: ten_pen(k)
    sigma_x, sigma_y, sigma_z = generate_sigmas_xyz()
    real_list = []
    imag_list = []
    time_list = []
    for i in range(0, Nt):
        time_list.append((i * dt, (i + 1) * dt))
    for i in range(1, number_e):
        s_iip1 = s_one(i, i + 1, sigma_x, sigma_y, sigma_z, 1,number_e)
        real_list.append(s_iip1.real)
        imag_list.append(s_iip1.imag)
    mydic["tl"] = time_list
    neighbor_list = []
    for i in range(1, number_e + 1):
        temp = []
        lef = i - 1
        cen = i
        right = i + 1
        if (1 >= lef and lef <= number_e):
            temp.append(lef)
        if (1 >= cen and cen <= number_e):
            temp.append(cen)
        if (1 >= right and right <= number_e):
            temp.append(right)

        neighbor_list.append(temp)

    mydic["nl"] = neighbor_list

    x = fong_gen_single()
    
    arr = [None] * number_e
    generateAllBinaryStrings(int(number_e/3), arr, 0)
    accum_list = arr
    gamma = 1
    sigma = 0
    e0, e1 = dfs_0_1(gamma, sigma, x)
    basis_list = ket_gen_dfs(e0, e1, int(number_e/3), accum_list)
    print(basis_list[0])
    print(basis_list[1])
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
    mydic["p"] = plank
    mydic["dim"] = 2 ** number_e

    GA_penalty()


T = 500
dt = 10
amp_list = [math.pi / 2.0, math.pi, (3.0 / 4.0) * math.pi]
number_e = 3
gate_list = generate_gate_lists_one()
plank = 1.0
gate = gate_list[0]
if __name__ == '__main__':
    
    build_dics(number_e-1,int(T/dt),len(amp_list))
    print(GA_routine_constant(number_e, gate, T, dt, amp_list, plank))
# arr = [None] * 2
# generateAllBinaryStrings(2, arr, 0)
# print(accum_list)
# convergence_test()
# x=creat_matrix_from_vector(np.asarray(list(range(1,13))),3,2,2)
# print(x)
# main
