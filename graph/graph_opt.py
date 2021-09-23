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
dic_i_j_neighbors={}
dic_node_neighbors={}



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

def check_in_grid(cory,corx,m,n):
    low_val =1
    m_upper = m
    n_upper = n

    if( corx >=low_val and corx <= n_upper and cory>=low_val and cory <= m_upper ):
        return True
    else:
        return False 



def generate_neighbors_for_nodes(dic_i_j):

    global dic_m_n
    global dic_node_neighbors
    print(dic_m_n)
    for key in dic_i_j.keys():
        print(key)
        node_list=[]
        i_j_list = dic_i_j[key]
        
        node_ent = dic_m_n[key]
        print((node_ent,i_j_list))
        for ent in i_j_list:
            node_list.append(dic_m_n[ent])
        dic_node_neighbors[(node_ent,)] = node_list
        

def generate_neighbors_for_graph_i_j(m,n):
    global dic_i_j_neighbors
  
    for i in range(1,m+1):
        for j in range(1,n+1):
            tuple_list=[]
            i_t = i
            j_t = j

        
            north = (i_t +1,j_t)
            east = (i_t,j_t+1)
            south = (i_t -1,j_t)
            west = (i_t,j_t-1)
            dir_list = [north,east,south,west]
            for ent in dir_list:
                if(check_in_grid(ent[0],ent[1],m,n)):
                    tuple_list.append(ent)
            dic_i_j_neighbors[(i_t,j_t)]= tuple_list
  

    return dic_i_j_neighbors



def mat_func3d(input_mat):
    # H = np.zeros((len(xk),len(xk)))

    # for i in range(0,len(xk)):
    # 	H[i,i] = ((-1)**i)*2.0
    # return H

    (y_size,x_size,z_size) = input_mat.shape
    print(input_mat.shape)
    H = {}
    for i in range(1,y_size+1):
        for j in range(1,x_size+1):
                for k in range(1,z_size+1):
                #print((i,j))

                    H[(i,j,k)] = input_mat[i-1,j-1,k-1]

    return H


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
    global dic_mn
    global dic_node_neighbors
    # M.X = Var(M.n_steps,M.k,M.nn,M.nm, domain=Binary)
    res2 = get_dic_mn_ele(dic_mn,(i,))
    node_list = dic_node_neighbors[(i,)]
    # for v in range(0,len(res2)):
    #     res2[v] = int(int(res2[v])) 
    return sum(M.V[n,k,i,j] -M.V[n,k,j,i]for j in node_list) == sum(M.X[n,k,p,i]*M.L[n,k,p] for p in M.nn)

def X_place_nodes(M,n, k,nn):
    return sum(M.X[n,k,nn,i]  for i in M.mn)== M.L[n,k,nn]*M.L[n,k,nn]
def X_place_nodes_no_double_placement(M,n,i):
    return (0.0,sum(M.X[n,k,nn,i]  for nn in M.nn for k in M.k),1.0)
def generate_mn_m_n_dics(m,n):
    global dic_mn
    global dic_m_n
    counter=1
    for i in range(1,m+1):
        for j in range(1,n+1):
            dic_mn[(counter,)] = (i,j)
            dic_m_n[(i,j)] = counter
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
    

def loc_matrix(step,pairs,nn,seq_list):
    loc_mat = np.zeros((step,pairs,nn))
    for n in range(0,len(seq_list)):
        seq= seq_list[n]
        for k in range(0,len(seq)):
            piece = seq[k]
            loc_mat[n,k,piece[0]-1]=1
            loc_mat[n,k,piece[1]-1]=-1
    return loc_mat
def graph_opt_fun(m,n,k,n_steps,number_nodes,file):
    seq_list = parser(file)
    generate_mn_m_n_dics(m,n)
    C = random_coeff_matrix(m*n,m*n)
    L =loc_matrix(n_steps,k,number_nodes,seq_list)
    print(L)
    L =mat_func3d(L)

    my_dic_i_j = generate_neighbors_for_graph_i_j(m,n)
    generate_neighbors_for_nodes(my_dic_i_j)
    py_c = mat_func(C)
    M = AbstractModel()
    M.n = RangeSet(1,n)
    M.m = RangeSet(1,m)
    M.mn = RangeSet(1,n*m)
    M.k = RangeSet(1,k)
    M.nn = RangeSet(1,number_nodes)
    M.C = Param(M.mn,M.mn, initialize=py_c)
    
    M.n_steps = RangeSet(1,n_steps)
    M.L = Param(M.n,M.k,M.nn, initialize=L)

    M.V = Var(M.n_steps,M.k,M.mn,M.mn, domain=Binary)
    M.X = Var(M.n_steps,M.k,M.nn,M.mn, domain=Binary)

    M.C1 = Constraint(M.n_steps,M.k,M.nn, rule=X_place_nodes)
    M.C2 = Constraint(M.n_steps,M.k,M.mn, rule=V_paths)
    M.C3 =Constraint(M.n_steps,M.mn,rule=X_place_nodes_no_double_placement)

    M.obj = Objective(rule=J, sense=minimize)
    #M.C2.pprint()
  

    instance = M.create_instance()
    instance.pprint()
   
    # results=SolverFactory('mindtpy').solve(instance,strategy='OA',
    #                             time_limit=3600, mip_solver='glpk', nlp_solver='ipopt',tee=True)
    opt = SolverFactory("glpk")
    results=opt.solve(instance,tee=True)

    instance.solutions.store_to(results)

    new_n = []
    # for n in instance.n_steps:
    #    print("-----------")
    #    for k in instance.mn:
    #        print(instance.V[n,k,:,:].value)
    #     print("-----------")
    
    # print(instance.X[1,1,1,1,1].value)
    # print(instance.X[1,1,2,1,1].value)
    # print(instance.X[1,1,1,1,2].value)
    # print(instance.X[1,1,2,1,2].value)
    #print(sum(instance.X[1,1,k,i,j].value for k in M.nn for i in M.m for j in M.n ))
    #print(sum(instance.V[1,2,i,j].value for i in M.m for j in M.n ))
    print(results)
 

out=parser("seq.txt")
# print(random_coeff_matrix(5,5))

file_name="seq.txt"
m=5
n=3
k=2
n_steps=3
number_nodes=4



graph_opt_fun(m,n,k,n_steps,number_nodes,file_name)
#print(loc_matrix(3,4,out))
#matrix (i-1)*j+j