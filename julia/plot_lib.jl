ENV["GKSwstype"]="nul"

using Plots
using Random

Nt=13
T=130.0
dt=T/Nt
p1 = acos(-1.0/sqrt(3))/pi
p2 = asin(1.0/3)/pi
amp_list = [p1,p2,.5,3.0/2.0,1,2-p1,1-p2]
for k=1:length(amp_list)
amp_list[k] = pi*amp_list[k]
end
Np = length(amp_list)
Nc=5
v_list = zeros((Nc,Nt,Np))
shift=1
v_list[2,5,5]=1.0
v_list[2,7,4]=1.0
v_list[2,9,5]=1.0

v_list[3,1+shift,2+shift]=1.0
v_list[3,3+shift,3+shift]=1.0
v_list[3,5+shift,3+shift]=1.0
v_list[3,7+shift,3+shift]=1.0
v_list[3,9+shift,3+shift]=1.0
v_list[3,11+shift,2+shift]=1.0

v_list[4,0+shift,0+shift]=1.0
v_list[4,2+shift,4+shift]=1.0
v_list[4,4+shift,3+shift]=1.0
v_list[4,6+shift,2+shift]=1.0
v_list[4,8+shift,3+shift]=1.0
v_list[4,10+shift,4+shift]=1.0
v_list[4,12+shift,5+shift]=1.0

v_list[5,1+shift,1+shift]=1.0
v_list[5,3+shift,3+shift]=1.0
v_list[5,5+shift,4+shift]=1.0
v_list[5,7+shift,4+shift]=1.0
v_list[5,9+shift,3+shift]=1.0
v_list[5,11+shift,6+shift]=1.0







plot_time_each_step=1
rng = MersenneTwister(1234);

name="plot"
random_time_data = bitrand(rng, Nt)
println(random_time_data)

t_list = []
for k=1:Nt
append!(t_list,[k*dt])
end
fontsize = 15

anim = @animate for i in 1:Nt
    ann =  Tuple{Int64,Int64,Plots.PlotText}[]
    println("plotting step $i")
    cur_dev = zeros(Nc)
    
    p_spec = -1
    for k=1:Nc
        p_spec = -1
        for p=1:Np
            if(v_list[k,i,p] == 1)
                p_spec = p
                println("hit!")
                break
            end
       
        end
        if(p_spec>0)
            cur_dev[k]=100.0
            append!(ann,[(k,1,text("$(round(amp_list[p_spec],digits=3))",fontsize, :white, :center))])
        
        else
            append!(ann,[(k,1,text("0.00",fontsize, :white, :center))])
        end
    end
    println(cur_dev)
    matVec(vector) = reshape(vector,1,length(vector))
    p1=heatmap(matVec(cur_dev),aspect_ratio=1,yticks=false,size=(750,100),title="Time = $(i*dt)",color=:coolwarm,colorbar = false)
    #print(ann)
    annotate!(ann,linecolor=:white)
    p2=plot(t_list[1:i],random_time_data[1:i])
    plot(p1,p2, layout = (2, 1))


end


gif(anim, "ex.gif",fps=plot_time_each_step)
print(amp_list)
#=
x = collect(range(0, 2, length= 100))
y1 = exp.(x)
y2 = exp.(1.3 .* x)

plot(x, y1, fillrange = y2, fillalpha = 0.35, c = 1, label = "Confidence band", legend = :topleft)
Let's scatter y1 and y2 on top of the plot, just to make sure we're filling in the right region.

plot!(x,y1, line = :scatter, msw = 0, ms = 2.5, label = "Lower bound")
plot!(x,y2, line = :scatter, msw = 0, ms = 2.5, label = "Upper bound")
=#
