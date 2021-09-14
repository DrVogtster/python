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

dic_mn={}
dic_m_n={}



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
            C[i,j] = -random.uniform(.1, 10.0)
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




def X_i(M,n,nn):
    print(n)
    print(type(n))
    return sum(M.X[n,k,nn,i,j] for i in M.m for j in M.n for k in M.k)==1.0
def X_j(M,n,k,i,j,):
    return sum(M.X[n,k,v,i,j] for v in M.nn for k in M.k)==1.0
def V_no_overlap(M,n, i,j):
    return sum(M.V[n,k,i,j]+M.V[n,k,j,i]  for k in M.k)<=2
def V_paths(M,n,k,i):
    # M.X = Var(M.n_steps,M.k,M.nn,M.nm, domain=Binary)
    res = get_dic_mn()

    return sum(M.V[n,k,i,j] +M.V[n,k,j,i]for j in M.mn) == M.V[m,k,]
def generate_mn_m_n_dics(m,n):
    global dic_mn
    global dic_m_n
    counter=0
    for i in range(0,m):
        for j in range(0,n):
            dic_mn[str(counter)] = str(i)+","+str(j)
            dic_m_n[str(i)+","+str(j)] = str(counter)
            counter+=1
    return dic_m_n,dic_mn

def get_dic_mn_ele(dic_mn,mn_ele_string):

    return dic_mn[mn_ele_string]

def get_dic_m_n_ele(dic_m_n,m_n_ele_string):
    return dic_m_n[m_n_ele_string]


def J(model):
    return sum(
            model.V[n,k,i,j]*model.C[i,j] for
            i in model.mn for j in model.mn for k in model.k for n in model.n_steps)
    

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
    M.mn = RangeSet(1,n*m)
    M.k = RangeSet(1,k)
    M.nn = RangeSet(1,number_nodes)
    M.C = Param(M.mn,M.mn, initialize=py_c)
    
    M.n_steps = RangeSet(1,n_steps)
    M.L = Param(M.nn,M.n_steps, initialize=L)

    M.V = Var(M.n_steps,M.k,M.mn,M.mn, domain=Binary)
    M.X = Var(M.n_steps,M.k,M.nn,M.m,M.n, domain=Binary)

    M.C1 = Constraint(M.n_steps,M.nn, rule=X_i)
    M.C2 = Constraint(M.n_steps,M.k,M.m,M.n, rule=X_j)

    M.obj = Objective(rule=J, sense=minimize)

    instance = M.create_instance()
    # results=SolverFactory('mindtpy').solve(instance,strategy='OA',
    #                             time_limit=3600, mip_solver='glpk', nlp_solver='ipopt',tee=True)
    opt = SolverFactory("glpk")
    results=opt.solve(instance,tee=True)

    instance.solutions.store_to(results)

    new_n = []
    # for i in instance.mn:
    #    for j in instance.mn:
      
    #      print(instance.V[1,1,i,j].value)



out=parser("seq.txt")
# print(random_coeff_matrix(5,5))

file_name="seq.txt"
m=5
n=3
k=2
n_steps=3
number_nodes=4

graph_opt_fun(m,n,k,n_steps,number_nodes,file_name)
print(loc_matrix(3,4,out))
#matrix (i-1)*j+j