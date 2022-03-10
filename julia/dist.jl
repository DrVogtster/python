using Distributions
using LinearAlgebra
using KernelDensity
using Plots
N = 3
# mu =[1.0,1.0,1.0]
# cov = Matrix(1.0I, N, N)  

function generate_kde(data)

    U=kde(data)

    return (U,U.x,U.density)

end

function generate_plots(file_name,chain_history,number_param,burn_in)
   
    for k=1:number_param
        data = chain_history[k,:]
        burn_data = data[burn_in:end]
        Plots.histogram(burn_data, normed=true,bins=20,label="Samples")
        (U,x,y) = generate_kde(burn_data)
        lb= minimum(data)
        ub = maximum(data)
        println((lb,ub))
        ticks = round.(range(lb,stop=ub,length = 4),digits=7)
        println(ticks)
        Plots.plot!(x,y, xticks=ticks ,formatter = identity,label="Distribution")

        Plots.savefig("$(file_name)q$k.png")


        # Plots.plot(burn_data)
        # Plots.savefig("$(file_name)chain$k.png")


    end




end


mu=1
cov=1
println(cov)
d = MvNormal(mu,cov)
x =rand(d,100)
println(x[1])

N=1

bnd_arr= [[0.0,1.0]]
num_samples_in_round=100
samples_wanted=100
global counter=1
good_sample_list=zeros((N,samples_wanted))



while counter<=samples_wanted
    println(counter)
    x =rand(d,100)
    for k=1:length(x)
        cur_sam = x[:,k]
        println(cur_sam)
        good_sample=true
        for j=1:length(bnd_arr)
            lb = bnd_arr[j][1]
            ub = bnd_arr[j][2]
            if cur_sam[j]< lb || cur_sam[j] > ub
                good_sample=false
                
            end

        end
        if(good_sample &&counter<=samples_wanted )
            good_sample_list[:,counter] = cur_sam
            global counter+=1
            
        end
        if(counter >samples_wanted)
            break
        end
    end



end

print(good_sample_list)


generate_plots("ex",good_sample_list,N,1)

