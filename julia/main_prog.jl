

using DiffEqFlux
using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using ForwardDiff,Calculus,Tracker
using ReverseDiff
using LinearAlgebra

include("kron_lib.jl")

function build_hams(ordering)


end

function fid_func(U,G)

end

function runner(rotation_list,G,T,dt)


end

function gen_right_neigh(Nc)

    buddy_list=[]
    for i=1:Nc-1
        append!(buddy_list,[[i,i+1]])
    end
    return buddy_list
end




T = 130
dt = 10
p1 = acos(-1.0/sqrt(3))/pi
p2 = asin(1.0/3)/pi
amp_list = [p1,p2,.5,3.0/2.0,1,2-p1,1-p2]
number_e=9
for i=1:length(amp_list)
    amp_list[i] = amp_list[i]*pi
end

 arr = zeros(9)
 res=generateAllBinaryStrings(convert(Int64, round(number_e/3.0, digits=0)), arr, 1)
 println(get_acum())