import numpy as np
from numpy.lib.shape_base import kron
import math

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  

def split(word):
    return [char for char in word]
def convert_str_to_list(mes):
    string_list = split(mes)
    int_list = [int(char) for char in string_list]
    return int_list

def ret_vec(val,e0,e1):
    if(val==0):
        return e0
    else:
        return e1

def ket_gen(order):
    e0=np.asarray([1,0])
    e1=np.asarray([0,1])
    k = len(order)-3
    temp =np.kron(ret_vec(order[-2],e0,e1),ret_vec(order[-1],e0,e1)) 
    while(k>=0):
        if(order[k]==0):
            temp = np.kron(e0,temp)
        else:
            temp = np.kron(e1,temp)

        k=k-1    
    return temp

def get_vec(string):
    con_s = convert_str_to_list(string)
    print(con_s)
    val = ket_gen(con_s)
    print(val)
    return val

def fong_gen_single():
    f_mat = np.zeros((8,8))
    e_000 = get_vec('000')
    e_100 = get_vec('100')
    e_010 = get_vec('010')
    e_001 = get_vec('001')
    e_111 = get_vec('111')
    e_101 = get_vec('101')
    e_011 = get_vec('011')
    e_110 = get_vec('110')

    f_mat[:,0] = (1.0/math.sqrt(2))*(e_010-e_100)
    f_mat[:,1] = (1.0/math.sqrt(2))*(e_011-e_101)
    f_mat[:,2] = (math.sqrt(2.0/3.0))*(e_001) - math.sqrt(1.0/6.0)*(e_010) -math.sqrt(1.0/6.0)*(e_100)
    f_mat[:,3] = (math.sqrt(1.0/6.0))*(e_011) +math.sqrt(1.0/6.0)*(e_101) -math.sqrt(2.0/3.0)*(e_110)
    f_mat[:,4] = e_000
    f_mat[:,5] = (math.sqrt(1.0/3.0))*(e_001 +e_010+e_100)
    f_mat[:,6] = (math.sqrt(1.0/3.0))*(e_011 +e_101+e_110)
    f_mat[:,7] = e_111

    return f_mat
def dfs_0_1(gamma,sigma,f_mat):

    f1= f_mat[:,0]
    f2 = f_mat[:,1]
    f3 = f_mat[:,2]
    f4 = f_mat[:,3]

    dfs_0 = (gamma*f1 + sigma*f2)
    dfs_1= (gamma*f3 + sigma*f4)
    return dfs_0,dfs_1

#dont scale by .25
def s_one(m,n,sigma_x,sigma_y,sigma_z):
    k_max=3
    dim=2
    return (sigmas_n(m,sigma_x,k_max,dim)*sigmas_n(n,sigma_x,k_max,dim) + sigmas_n(m,sigma_y,k_max,dim)*sigmas_n(n,sigma_y,k_max,dim) + sigmas_n(m,sigma_z,k_max,dim)*sigmas_n(n,sigma_z,k_max,dim))

def generate_sigmas_xyz():
    sigma_x = np.zeros((2,2), dtype=complex)
    sigma_x[0,1] =1
    sigma_x[1,0] =1
    
    sigma_y = np.zeros((2,2), dtype=complex)
    sigma_y[0,1]=-1j
    sigma_y[1,0]=1j

    sigma_z = np.zeros((2,2), dtype=complex)
    sigma_z[0,0]=1.0
    sigma_z[1,1] = -1.0
    return sigma_x,sigma_y,sigma_z





def sigmas_n(n,sigma,k_max,dim):
    eye = np.eye(dim)
    start=None
    if(k_max==n):
        start = np.kron(eye,sigma)
    elif(k_max-1==n):
        start = np.kron(sigma, eye)
    else:
        start=np.kron(eye,eye)    


    k_max=k_max-2
    while(k_max>=1):
        if(k_max==n):
            start = np.kron(sigma,start)
        else:
            start = np.kron(eye, start)


        
        k_max=k_max-1
        print(start)
    return start



x=fong_gen_single()
matprint(x)
for i in range(0,8):
    print(np.linalg.norm(x[:,i]))

# gamma=.5
# sigma=.5
# y=dfs_0_1(gamma,sigma,x)
# print(y)
# print(10)

# sigma_x,sigma_y,sigma_z = generate_sigmas_xyz()


# H=s_one(1,1,sigma_x,sigma_y,sigma_z)
# print(H)
#comment