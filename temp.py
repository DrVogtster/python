import os
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.colors as colors


from pyomo.core import *
from pyomo.environ import *
#from plot_graph import *

dic_mn={}
dic_m_n={}
@@ -213,6 +223,7 @@ def loc_matrix(step,pairs,nn,seq_list):
            loc_mat[n,k,piece[1]-1]=-1
    return loc_mat
def graph_opt_fun(m,n,k,n_steps,number_nodes,file):
    global dic_mn
    seq_list = parser(file)
    generate_mn_m_n_dics(m,n)
    C = random_coeff_matrix(m*n,m*n)
@@ -232,7 +243,7 @@ def graph_opt_fun(m,n,k,n_steps,number_nodes,file):
    M.C = Param(M.mn,M.mn, initialize=py_c)

    M.n_steps = RangeSet(1,n_steps)
    M.L = Param(M.n,M.k,M.nn, initialize=L)
    M.L = Param(M.n_steps,M.k,M.nn, initialize=L)

    M.V = Var(M.n_steps,M.k,M.mn,M.mn, domain=Binary)
    M.X = Var(M.n_steps,M.k,M.nn,M.mn, domain=Binary)
@@ -269,14 +280,142 @@ def graph_opt_fun(m,n,k,n_steps,number_nodes,file):
    #print(sum(instance.X[1,1,k,i,j].value for k in M.nn for i in M.m for j in M.n ))
    #print(sum(instance.V[1,2,i,j].value for i in M.m for j in M.n ))
    print(results)
    print(seq_list)
    plot_time_each_step=1
    plot_graph_time_opt(seq_list,m,n,plot_time_each_step,instance.X,instance.V,dic_mn,number_nodes)


def plot_graph_time_opt(seq_list,m,n,plot_time_each_step,X,V,dic_mn,nn):
    n_step = len(seq_list)
    nn = []



    number_steps = len(seq_list)


    duration = number_steps*plot_time_each_step


    fig, ax = plt.subplots()
    x = np.linspace(-2, 2, 200)
    # method to get frames
    def make_frame(t):
        global dic_mn
        #plt.imshow(matrix, cmap = cm.Greys_r)
        #print(t)
        my_t = int(t+1)
        matrix = np.zeros((m, n))
        ax = plt.gca()
        ax.clear()
        current_level = seq_list[my_t-1]
        for k in range(0,len(current_level)):
            for i in range(0,len(current_level[k])):

                    val = current_level[k][i]
                    node_loc_nm = None
                    for j in range(1,m*n+1):
                        if(X[my_t,k+1,val,j]==1):
                            node_loc_nm=j

                            break
                    i_j_loc = dic_mn[(node_loc_nm,)]
                    print(i_j_loc )
                    matrix[i_j_loc[0]-1, i_j_loc[1]-1] = 100.0
                    ax.text(i_j_loc[1]-1, i_j_loc[0]-1, "n_" + str(val), va='center', ha='center')
        print(matrix)
        plt.imshow(matrix, cmap = cm.Reds,aspect='auto')
        plt.title('Step: ' +str(int(t)+1))
        plt.xlabel('n=' +str(n))
        plt.ylabel('m='+str(m))
        ax = plt.gca()
        for k in range(0,len(current_level)):
                start_n = current_level[k][0]
                end_n = current_level[k][1]
                start_loc=None
                end_loc=None
                for i in range(1,m*n+1):
                    if(X[my_t,k+1,start_n,i]==1):
                        start_loc =i
                for i in range(1,m*n+1):
                    if(X[my_t,k+1,end_n,i]==1):
                        end_loc =i
                # print("-------------")
                # print(current_level[k])
                # print(start_n)
                # print(end_n)
                # print("-------------")
                start = start_loc
                end= end_loc
                node_path =[]
                i_j_path = []
                while(start!=end):

                    start_i_j = dic_mn[(start,)]
                    next_loc = None
                    for i in range(1,m*n+1):
                        if(V[my_t,k+1,start,i]==1):
                            next_loc=i
                            break
                    next_loc_i_j = dic_mn[(next_loc,)]
                    y_temp = [start_i_j[0]-1,next_loc_i_j[0]-1]
                    x_temp = [start_i_j[1]-1,next_loc_i_j[1]-1]
                    x,y = np.array([x_temp, y_temp])
                    print(x,y)
                    line = plt.Line2D(x, y, lw=5., color='r', alpha=0.4)
                    line.set_clip_on(False)
                    ax.add_line(line)
                    node_path.append((start,next_loc))
                    i_j_path.append((start_i_j,next_loc_i_j))
                    start = next_loc

                print("NODE PATH " +str(node_path))
                print("i_j PATH " +str(i_j_path))




        # plotting line




        # for connections in current_connections:
        #     print(connections)
        #     x_temp = [connections[0][0],connections[1][0]]
        #     y_temp = [connections[0][1],connections[1][1]]

        #     x,y = np.array([y_temp, x_temp])
        #     print(x,y)
        #     line = plt.Line2D(x, y, lw=5., color='r', alpha=0.4)
        #     line.set_clip_on(False)
        #     ax.add_line(line)


        # returning mumpy image
        return mplfig_to_npimage(fig)

    # creating animation
    animation = VideoClip(make_frame, duration = duration)

    # displaying animation with auto play and looping
    animation.ipython_display(fps = plot_time_each_step, loop = True, autoplay = True)


    # x,y = np.array([[50, 150], [30, 80]])
    # line = plt.Line2D(x, y, lw=5., color='k', alpha=0.4)
    # line.set_clip_on(False)
    # ax.add_line(line)

    # plt.savefig("ex.pdf")
    print("done making movie")

out=parser("seq.txt")
# print(random_coeff_matrix(5,5))

file_name="seq.txt"
m=5
n=5
k=2
n_steps=3
number_nodes=4
