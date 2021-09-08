from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.colors as colors


#for fix node placement show connections in time on a mxn grib
def plot_graph_time(node_locs,connections_in_time,m,n,plot_time_each_step):
    matrix = np.zeros((m, n))
    for node in node_locs:
        matrix[node[0], node[1]] = 100.0

    number_steps = len(connections_in_time)
   

    duration = number_steps*plot_time_each_step


    fig, ax = plt.subplots()
    x = np.linspace(-2, 2, 200)
    # method to get frames
    def make_frame(t):
        #plt.imshow(matrix, cmap = cm.Greys_r)
        print(t)
        ax = plt.gca()
        ax.clear()
        plt.imshow(matrix, cmap = cm.Reds,aspect='auto')
        plt.title('Step: ' +str(int(t)+1))
        plt.xlabel('n=' +str(n))
        plt.ylabel('m='+str(m))
        ax = plt.gca()
        # plotting line
        current_connections = connections_in_time[int(t)]
        for connections in current_connections:
            print(connections)
            x_temp = [connections[0][0],connections[1][0]]
            y_temp = [connections[0][1],connections[1][1]]
            
            x,y = np.array([y_temp, x_temp])
            print(x,y)
            line = plt.Line2D(x, y, lw=5., color='r', alpha=0.4)
            line.set_clip_on(False)
            ax.add_line(line)

        
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


m=3
n=100
node_locs = [[0,0],[1,10],[2,20],[0,50],[2,99]]

#connections c[k][j][l] - at time k multiple connections if j>0, and l has connections 
connections_in_time=[[[[0,0],[2,99]],[[0,50],[2,20]]],[[[1,10],[2,99]],[[0,50],[0,0]]],[[[2,99],[2,20]],[[1,10],[0,50]]]]
plot_time_each_step=1
plot_graph_time(node_locs,connections_in_time,m,n,plot_time_each_step)


