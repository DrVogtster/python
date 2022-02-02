

using DiffEqFlux
using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using ForwardDiff,Calculus,Tracker
using ReverseDiff
using LinearAlgebra

function build_hams(ordering)


end

function fid_func(U,G)

end

function runner(rotation_list,G,T,dt)


end





T = 130
dt = 10
p1 = acos(-1.0/sqrt(3))/pi
p2 = asin(1.0/3)/pi
amp_list = [p1,p2,.5,3.0/2.0,1,2-p1,1-p2]

for i=1:length(amp_list)
    amp_list[i] = amp_list[i]*pi
end

