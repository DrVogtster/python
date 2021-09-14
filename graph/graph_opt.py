import random
import numpy as np
import glob, os
import pandas as pd
import numpy as np
import pickle
import pylab as plt
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats.distributions import norm
import math
import asyncio
import os
from sklearn.model_selection import GridSearchCV

from pyomo.core import *
from pyomo.environ import *


def parser(file):
    sequence_list=[]
    file1 = open(file, 'r')
    Lines = file1.readlines()
    
    for line in Lines:
        line=line.replace("\n", "")
        first = line.split("|")
        #print(first)
        seq_piece=[]
        for ent in first:
            end = ent.split(" ")
            print(end)
            seq_piece.append([int(end[0]),int(end[1])])
        sequence_list.append(seq_piece)
    return sequence_list

def random_coeff_matrix(m,n):
    C = np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            C[i,j] = random.uniform(.1, 10.0)
    return C










def mat_func(input_mat):
    # H = np.zeros((len(xk),len(xk)))

    # for i in range(0,len(xk)):
    # 	H[i,i] = ((-1)**i)*2.0
    # return H

    (y_size,x_size) = input_mat.shape
    H = {}
    for i in range(1,y_size+1):
        for j in range(1,x_size+1):
            #print((i,j))

                H[(i,j)] = input_mat[i-1,j-1]

    return H
def vec_func(self,input_vec):
    # H = np.zeros((len(xk),len(xk)))

    # for i in range(0,len(xk)):
    # 	H[i,i] = ((-1)**i)*2.0
    # return H

    xsize = len(input_vec)
    v = {}

    for i in range(1,xsize+1):

        v[i] =input_vec[i-1]

    return v

def build_sos1_constraint_one(self, M, i):
    return (M.ni[i] - M.n_max[1]*M.bi[i] <= 0.0)
def build_sos1_constraint_two(self, M, i):
    return (M.ni[i] - M.bi[i] >= 0.0)

def greater_percent(self, M, i):
    return M.wi[i]>=M.mp[1]
def number_stock_constraint(self, model):
    return sum(model.bi[i] for i in model.number_stocks) <= model.ns[1]

def satisfy_allocation(self,model):
    return((1-model.buffer[1])*model.p[1],sum(model.buy_price[i]*model.wi[i] for i in model.number_stocks),(1+model.buffer[1])*model.p[1])

def num_con(self,model):


    # return (model.min_stock[1],counter,model.max_stock[1])
    return sum(model.bi[i] for i in model.number_stocks) ==model.max_stock[1]

def bound_consd(self, M, i):
    return M.bi[i]<=M.wi[i]

def percentage_constraint(self, model):
    return sum(model.wi[i] for i in model.number_stocks) == 1.0

def allocation_constraint(self, model):
    return (0.0,sum(model.ni[i] * model.buy_price[i] for i in model.number_stocks),model.p[1])



def sharp_optimization_and_get_buy_sell_prices(m,n,k):
    
    M = AbstractModel()
    M.n = RangeSet(1,n)
    M.m = RangeSet(1,m)
    M.k = RangeSet(1,k)

    inital_zero = [0]*len(self.portfolio.keys())
    inital_one = [1]*len(self.portfolio.keys())
    max_percent = 1.0/len(self.portfolio.keys())
    buffer_val=.1
    # for i in range(0,max_amount_of_stocks):
    #     inital_one[i] = max_percent

    init_vec0 = self.vec_func((inital_zero))
    init_vec1 = self.vec_func((inital_one))
    print(max_percent)
    #M.bi = Var(M.number_stocks, domain=Binary,initialize=init_vec0)
    #M.ni = Var(M.number_stocks, domain=NonNegativeReals,initialize=init_vec1)
    M.wi = Var(M.number_stocks, domain=NonNegativeReals,initialize=init_vec1)
    M.n_set = RangeSet(1, 1)
    M.p = Param(M.n_set, initialize={1: portfolio_allocation_amount})
    M.buffer = Param(M.n_set, initialize={1: buffer_val})

    M.mp = Param(M.n_set, initialize={1: max_percent})
    M.ns = Param(M.n_set, initialize={1: max_amount_of_stocks})
    buy_vec = self.vec_func(buy_list)
    M.buy_price = Param(M.number_stocks, initialize=buy_vec)
    n_max = round(portfolio_allocation_amount/min(buy_list))
    M.n_max = Param(M.n_set, initialize={1: n_max})
    print((np.amin(self.avg_mat),np.amax(self.avg_mat)))
    print((np.amin(self.cov_mat),np.amax(self.cov_mat)))
    print((np.amin(self.cor_mat),np.amax(self.cor_mat)))

    avg_vec = self.vec_func(self.avg_mat)
    cov_mat = self.mat_func(self.cov_mat)
    cor_mat = self.mat_func(self.cor_mat)
    buy_vec= self.vec_func(buy_list)


    M.expected_price = Param(M.number_stocks, initialize=avg_vec)

    M.cov_mat = Param(M.number_stocks, M.number_stocks, initialize=cov_mat)
    M.cor_mat = Param(M.number_stocks, M.number_stocks, initialize=cor_mat)
    M.i = RangeSet(len(self.portfolio.keys()))
    #M.buffer = Param(M.n_set, initialize={1: buffer_val})
    M.C1 = Constraint(M.i, rule=self.greater_percent)
    #M.C1 = Constraint(M.i, rule=self.build_sos1_constraint_one)
    #M.C11 = Constraint(M.i, rule=self.build_sos1_constraint_two)
    M.C2 = Constraint(rule=self.percentage_constraint)
    #M.C4 = Constraint(rule=self.satisfy_allocation)
    # M.C3 = Constraint(rule=self.allocation_constraint)
    M.obj = Objective(rule=self.sharp_obj, sense=minimize)



    instance = M.create_instance()

    #results = SolverFactory('mindtpy').solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)
    print("solving optimization problem")
    # results=SolverFactory('mindtpy').solve(instance,strategy='ECP',
    #                            time_limit=3600, mip_solver='cplex', nlp_solver='ipopt',tee=True)
    results=SolverFactory('ipopt').solve(instance,tee=True)
    #results.options['max_iter']= 10000 #number of iterations you wish
    instance.solutions.store_to(results)

    new_bi = []

    new_ni = []
    new_wi=[]
    for p in instance.number_stocks:
        # print(instance.v[p].value)
        # new_ni.append(instance.ni[p].value)
        # new_bi.append(instance.bi[p].value)
        new_wi.append(instance.wi[p].value)
    #print(new_bi)
    #print(new_ni)
    print(new_wi)
    res_dic={}
    counter=0
    real_num_list=[]
    for k in range(0,len(new_wi)):
            num = new_wi[k]*portfolio_allocation_amount/buy_list[k]
            real_num_list.append(num)

    print("integer piece")
    #real_num_list=[]
    V = AbstractModel()
    V.number_stocks = RangeSet(1, len(self.portfolio.keys()))
    V.buy_vector = Param(V.number_stocks, initialize=buy_vec)
    V.n_set = RangeSet(1, 1)
    V.p = Param(V.n_set, initialize={1: portfolio_allocation_amount})
    num_vec  =self.vec_func(real_num_list)
    V.buff_val = Param(V.n_set, initialize={1: buffer_val})

    round_val = []
    for k in range(0,len(real_num_list)):
        round_val.append(round(real_num_list[k]))
    print(round_val)
    round_vec = self.vec_func(round_val)
    V.n_real = Param(V.number_stocks, initialize=num_vec)
    V.ni = Var(V.number_stocks, domain=NonNegativeIntegers,initialize=round_vec)
    V.C1 = Constraint(rule=self.n_c1)
    V.obj = Objective(rule=self.obj_n, sense=minimize)

    instance = V.create_instance()
    results=SolverFactory('mindtpy').solve(instance,strategy='OA',
                                time_limit=3600, mip_solver='glpk', nlp_solver='ipopt',tee=True)



    instance.solutions.store_to(results)

    new_n = []
    for p in instance.number_stocks:
        # print(instance.v[p].value)
        # new_ni.append(instance.ni[p].value)
        # new_bi.append(instance.bi[p].value)
        new_n.append(instance.ni[p].value)
def X_i(M,n, i,k):
    return sum(M.X[n,k,i,j] for j in M.nm)==1.0
def X_j(M,n,j,k):
    return sum(M.X[n,k,i,j] for i in M.nn)==1.0
def V_no_overlap(M,n, i,j):
    return sum(M.V[n,k,i,j]+M.V[n,k,j,i]  for k in M.k)<=2
def V_paths(M,n,k,i):
    # M.X = Var(M.n_steps,M.k,M.nn,M.nm, domain=Binary)
    for p in M.nn:
        if M.X[n,k,p,]

    return sum(M.V[n,k,i,j] +M.V[n,k,j,i]for j in M.mn) == M.L[i,k]
def J(model):
    return sum(
            model.V[n,k,i,j]*model.C[i,j] for
            i in model.nm for j in model.nm for k in model.k for n in model.nn)

def loc_matrix(step,Nq,seq_list):
    loc_mat = np.zeros((Nq,step))
    counter=0
    for seq in seq_list:
        for piece in seq:
            print(piece)
            loc_mat[piece[0]-1,counter]=1
            loc_mat[piece[1]-1,counter]=-1

        counter+=1
    return loc_mat
def graph_opt_fun(m,n,k,n_steps,number_nodes,file):
    seq_list = parser(file)
    C = random_coeff_matrix(m*n,m*n)
    L =loc_matrix(n_steps,number_nodes,seq_list)
    py_c = mat_func(C)
    M = AbstractModel()
    M.n = RangeSet(1,n)
    M.m = RangeSet(1,m)
    M.nm = RangeSet(1,n*m)
    M.k = RangeSet(1,k)
    M.nn = RangeSet(1,number_nodes)
    M.C = Param(M.nm,M.nm, initialize=py_c)
    
    M.n_steps = RangeSet(1,n_steps)
    M.L = Param(M.nn,M.n_steps, initialize=L)

    M.V = Var(M.n_steps,M.k,M.nm,M.nm, domain=Binary)
    M.X = Var(M.n_steps,M.k,M.nn,M.m,M.n, domain=Binary)



out=parser("seq.txt")
# print(random_coeff_matrix(5,5))

file_name="seq.txt"
graph_opt_fun(5,5,5,5,5,file_name)
print(loc_matrix(3,4,out))
#matrix (i-1)*j+j