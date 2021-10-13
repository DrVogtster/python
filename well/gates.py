import numpy as np
import math
#CHPT gates
def H_gate():
    H = np.zeros((2,2))
    H[0,0] = 1
    H[0,1] = 1
    H[1,0] = 1
    H[1,1] = -1
    H = (1.0/math.sqrt(2))*H
    return H 
def Z_theta(theta):
    Z = np.zeros((2,2))
    itheta = 1j*theta
    Z[0,0]  =1
    Z[1,1] = np.exp(itheta)

def P_gate():
    return Z_theta(math.pi/2.0)
def T_gate():
    return Z_theta(math.pi/4.0)

def C_gate():
    C = np.zeros((4,4))
    C[0,0]=1
    C[1,1]=1
    C[3,2]=1
    C[2,3]=1
    return C


def generate_gate_lists_one():
    gate_list=[]
    funcs = [H_gate,P_gate,T_gate]
    for f in funcs:
        gate_list.append(f())

    return gate_list

